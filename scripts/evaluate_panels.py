"""LAB10 · Paso 7: evaluación con paneles no seleccionados.

Usa los modelos guardados por scripts/train_ae_vae.py (autoencoder.keras, encoder.keras y
decoder.keras) para:
  1. Elegir 16 ejemplos de prueba con semilla fija (sin selección manual).
  2. Mostrar originales, reconstrucciones AE y VAE y 16 muestras nuevas.
  3. Buscar el vecino más cercano de entrenamiento de cada muestra generada.
  4. Medir el error de reconstrucción de ambos modelos en las 10,000 imágenes de prueba.

Uso:
    uv run python scripts/evaluate_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inf8239_u03_gen.data import normalize_images, validate_images

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N = 16
CLASSES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]


def load_models():
    models = ROOT / "models"
    paths = {name: models / f"{name}.keras" for name in ("autoencoder", "encoder", "decoder")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Faltan modelos: {missing}. Ejecute primero scripts/train_ae_vae.py")
    return {name: tf.keras.models.load_model(path, compile=False) for name, path in paths.items()}


def predict(model, values, batch_size=1024):
    return model.predict(values, batch_size=batch_size, verbose=0)


def per_image_errors(originals, reconstructions):
    flat_x = originals.reshape(len(originals), -1)
    flat_r = np.clip(reconstructions.reshape(len(originals), -1), 1e-7, 1 - 1e-7)
    mse = np.mean((flat_x - flat_r) ** 2, axis=1)
    bce = -np.mean(flat_x * np.log(flat_r) + (1 - flat_x) * np.log(1 - flat_r), axis=1)
    return mse, bce


def nearest_neighbors(queries, reference, chunk=10000):
    """Distancia euclidiana en píxeles (784 valores entre 0 y 1) al vecino más cercano."""
    q = queries.reshape(len(queries), -1).astype("float64")
    q_norm = (q ** 2).sum(axis=1)
    best_distance = np.full(len(q), np.inf)
    best_index = np.zeros(len(q), dtype=int)
    for start in range(0, len(reference), chunk):
        r = reference[start:start + chunk].reshape(-1, q.shape[1]).astype("float64")
        squared = q_norm[:, None] + (r ** 2).sum(axis=1)[None, :] - 2 * q @ r.T
        distance = np.sqrt(np.maximum(squared, 0))
        index = distance.argmin(axis=1)
        value = distance[np.arange(len(q)), index]
        better = value < best_distance
        best_distance[better] = value[better]
        best_index[better] = index[better] + start
    return best_index, best_distance


def mean_pairwise_distance(images):
    flat = images.reshape(len(images), -1).astype("float64")
    distances = np.sqrt(((flat[:, None, :] - flat[None, :, :]) ** 2).sum(axis=2))
    upper = distances[np.triu_indices(len(images), k=1)]
    return float(upper.mean()), float(upper.min())


def save_panel(path, rows, titles, labels=None):
    figure, axes = plt.subplots(len(rows), N, figsize=(N * 0.95, len(rows) * 1.25))
    for r, (images, title) in enumerate(zip(rows, titles)):
        for c in range(N):
            axis = axes[r, c]
            axis.imshow(images[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if c == 0:
                axis.set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center")
            if labels is not None and labels[r] is not None:
                axis.set_title(labels[r][c], fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = normalize_images(x_train), normalize_images(x_test)
    validate_images(x_test)
    models = load_models()

    # 1-2. Panel de 16 ejemplos elegidos con semilla, sin selección manual
    selection = np.sort(np.random.default_rng(SEED).choice(len(x_test), size=N, replace=False))
    originals = x_test[selection]
    ae_rec = predict(models["autoencoder"], originals)
    mean, _ = predict(models["encoder"], originals)
    vae_rec = predict(models["decoder"], mean)
    latent_samples = np.random.default_rng(SEED + 1).normal(size=(N, 2))
    generated = predict(models["decoder"], latent_samples)
    save_panel(reports / "panel_16.png", [originals, ae_rec, vae_rec, generated],
               ["Original", "AE", "VAE", "Generada"],
               [[CLASSES[y] for y in y_test[selection]], None, None, None])

    # 3. Vecino más cercano en entrenamiento
    nn_index, nn_distance = nearest_neighbors(generated, x_train)
    _, test_nn_distance = nearest_neighbors(originals, x_train)
    save_panel(reports / "nearest_neighbors.png", [generated, x_train[nn_index]],
               ["Generada", "Vecino train"],
               [[f"d={d:.1f}" for d in nn_distance],
                [f"{CLASSES[y_train[i]]}" for i in nn_index]])

    # 4. Error de reconstrucción en las 10,000 imágenes de prueba
    ae_all = predict(models["autoencoder"], x_test)
    mean_all, log_var_all = predict(models["encoder"], x_test)
    vae_all = predict(models["decoder"], mean_all)
    ae_mse, ae_bce = per_image_errors(x_test, ae_all)
    vae_mse, vae_bce = per_image_errors(x_test, vae_all)
    kl = -0.5 * np.sum(1 + log_var_all - mean_all ** 2 - np.exp(log_var_all), axis=1)

    per_class = []
    for label, name in enumerate(CLASSES):
        mask = y_test == label
        per_class.append({"clase": name, "ae_mse": float(ae_mse[mask].mean()),
                          "vae_mse": float(vae_mse[mask].mean())})

    gen_mean, gen_min = mean_pairwise_distance(generated)
    orig_mean, orig_min = mean_pairwise_distance(originals)
    nn_classes = [CLASSES[y_train[i]] for i in nn_index]
    metrics = {
        "seed": SEED,
        "test_indices": selection.tolist(),
        "reconstruction_test": {
            "images": int(len(x_test)),
            "ae_mse": float(ae_mse.mean()), "vae_mse": float(vae_mse.mean()),
            "ae_bce_per_pixel": float(ae_bce.mean()), "vae_bce_per_pixel": float(vae_bce.mean()),
            "ae_bce_per_image": float(ae_bce.mean() * 784), "vae_bce_per_image": float(vae_bce.mean() * 784),
            "vae_kl_test": float(kl.mean()),
        },
        "reconstruction_mse_by_class": per_class,
        "generation": {
            "samples": N,
            "nearest_train_distance_mean": float(nn_distance.mean()),
            "nearest_train_distance_min": float(nn_distance.min()),
            "test_to_train_distance_mean": float(test_nn_distance.mean()),
            "pairwise_distance_generated_mean": gen_mean,
            "pairwise_distance_generated_min": gen_min,
            "pairwise_distance_test_mean": orig_mean,
            "pairwise_distance_test_min": orig_min,
            "nearest_neighbor_classes": {name: nn_classes.count(name) for name in CLASSES if name in nn_classes},
        },
    }
    (reports / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    r = metrics["reconstruction_test"]
    g = metrics["generation"]
    print(f"Ejemplos de prueba (semilla {SEED}): {selection.tolist()}")
    print(f"Reconstrucción en {r['images']} imágenes de prueba")
    print(f"  AE : MSE {r['ae_mse']:.4f} | BCE por píxel {r['ae_bce_per_pixel']:.4f} ({r['ae_bce_per_image']:.1f} por imagen)")
    print(f"  VAE: MSE {r['vae_mse']:.4f} | BCE por píxel {r['vae_bce_per_pixel']:.4f} ({r['vae_bce_per_image']:.1f} por imagen) | KL {r['vae_kl_test']:.2f}")
    print("MSE por clase (AE / VAE):")
    for row in per_class:
        print(f"  {row['clase']:<10} {row['ae_mse']:.4f} / {row['vae_mse']:.4f}")
    print("Generación (16 muestras):")
    print(f"  Distancia al vecino de entrenamiento: media {g['nearest_train_distance_mean']:.2f}, mínima {g['nearest_train_distance_min']:.2f}")
    print(f"  Referencia: imagen de prueba real a su vecino de entrenamiento: media {g['test_to_train_distance_mean']:.2f}")
    print(f"  Distancia entre muestras: media {g['pairwise_distance_generated_mean']:.2f}, mínima {g['pairwise_distance_generated_min']:.2f}")
    print(f"  Referencia entre imágenes reales: media {g['pairwise_distance_test_mean']:.2f}, mínima {g['pairwise_distance_test_min']:.2f}")
    print(f"  Clases de los vecinos: {g['nearest_neighbor_classes']}")
    print("Archivos: reports/panel_16.png, reports/nearest_neighbors.png, reports/evaluation_metrics.json")


if __name__ == "__main__":
    main()
