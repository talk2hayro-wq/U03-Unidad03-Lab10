# INF-8239 · Unidad 03 · Modelos generativos

Autor académico: Edwin Ramón José Nolasco
Estudiante: Hairo Fermin – AG4949 · LAB10 · 19 de octubre de 2026

Autoencoder y VAE entrenados sobre Fashion-MNIST y comparados por reconstrucción, generación, espacio latente y costo, con una aplicación local para muestrear el espacio latente del VAE.

> Una imagen sintética no es automáticamente privada, original, verdadera ni libre de sesgo. No afirme privacidad por el solo hecho de generar datos sintéticos.

## Requisitos
- [uv](https://docs.astral.sh/uv/) y Python 3.12 (`requires-python = ">=3.12,<3.13"`).
- Conexión a internet la primera vez: `tf.keras.datasets.fashion_mnist` descarga unos 31 MB y los guarda en caché.
- CPU es suficiente: ambos modelos entrenan en unos 15 s.

## CPU
```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
uv run python scripts/evaluate_panels.py
uv run pytest -q
```
`check_runtime.py` debe mostrar `GPU: []` en Windows: TensorFlow no usa GPU en Windows nativo desde la versión 2.11.

## GPU
Solo para WSL2/Linux con NVIDIA correctamente configurada:
```bash
uv sync --extra gpu
```
`--extra gpu` instala `tensorflow[and-cuda]`, que solo funciona con NVIDIA. Con GPU AMD (por ejemplo, Radeon RX 6700) use `--extra cpu`: ROCm en WSL no soporta la serie RX 6000 y el plugin DirectML está pausado y no admite Python 3.12.

## Aplicación
```bash
uv run streamlit run app/streamlit_app.py
```
Dos controles (−3 a 3) eligen un punto del espacio latente y el decodificador del VAE genera la imagen. Requiere `models/decoder.keras`; si no existe, la aplicación pide entrenar primero.

Para una nube con memoria limitada:
```bash
uv export --format requirements.txt --extra cpu --output-file requirements-cloud.txt
```
- La aplicación solo hace inferencia; entrene en local o en Colab (`requirements-colab.txt`), nunca en cada visita.
- `.gitignore` excluye `models/*.keras`: para publicar la aplicación incluya `models/decoder.keras` (416 KB) con `git add -f`.
- El proyecto no usa secretos; si existieran, se configuran en la plataforma y nunca en Git.

## Estructura
```
src/inf8239_u03_gen/   data.py (normalización y validación) y models.py (Autoencoder, codificador y decodificador)
scripts/               check_runtime.py, train_ae_vae.py, evaluate_panels.py
app/                   streamlit_app.py
tests/                 contratos de datos y de modelos
models/                autoencoder.keras, encoder.keras, decoder.keras
reports/               paneles, interpolación y métricas
MODEL_CARD.md          ficha del modelo
AI_USE_DECLARATION.md  declaración de uso de IA
```

## Cambios respecto del proyecto base
- `scripts/train_ae_vae.py` guarda también `models/encoder.keras` (codificador del VAE). No cambia el entrenamiento: con semilla 42 las pérdidas se reproducen exactamente.
- `scripts/evaluate_panels.py` (nuevo) carga los modelos guardados, sin entrenar, y genera:
  - `reports/panel_16.png`: 16 ejemplos de prueba elegidos con semilla 42 (originales, Autoencoder, VAE) y 16 muestras nuevas.
  - `reports/nearest_neighbors.png`: cada muestra junto a su imagen más cercana de entrenamiento.
  - `reports/evaluation_metrics.json`: error de reconstrucción en las 10,000 imágenes de prueba, por clase, KL, distancias a vecinos y entre muestras.

## Modelos
| | Autoencoder | VAE |
| --- | --- | --- |
| Arquitectura | 784 → 16 → 784 | 784 → 128 → 2 (media y log-varianza); 2 → 128 → 784 |
| Parámetros | 25,888 | 202,516 (decodificador 101,520) |
| Entrenamiento | 6 épocas, 10 % de validación, EarlyStopping | 8 épocas, sin validación |
| Genera muestras nuevas | No | Sí |

## Hallazgos
Resultados en CPU (Windows 11, TensorFlow 2.21.0), semilla 42.

| Métrica (10,000 imágenes de prueba) | Autoencoder | VAE |
| --- | --- | --- |
| MSE | 0.0203 | 0.0330 |
| Entropía cruzada por imagen | 240.5 | 267.0 |
| KL | — | 6.30 |
| Tiempo de entrenamiento | 5.5–5.9 s | 9.0–9.1 s |
| Archivo | 331 KB | 416 KB (decodificador) |

- **Reconstrucción:** el Autoencoder es mejor en las 10 clases. Las más difíciles son bolso y sandalia; el VAE reconstruye los bolsos como prendas de torso y las sandalias como botines. No hay sobreajuste: prueba y entrenamiento dan casi lo mismo.
- **Generación:** las muestras del VAE son siluetas borrosas; algunas mezclan calzado y torso.
- **Originalidad:** ninguna muestra copia un ejemplo de entrenamiento. Su distancia al vecino más cercano (2.79) es menor que la de una imagen real (3.89) porque son promedios suaves, no por memorización.
- **Diversidad:** baja. Las muestras están entre sí a la mitad de distancia que las imágenes reales (6.35 frente a 11.66) y 13 de 16 se parecen a prendas de torso.
- **Espacio latente:** organizado sin etiquetas. El eje Y separa prendas de torso (arriba) de calzado (abajo); el eje X va de mangas largas y botines a mangas cortas, zapatillas y pantalones. La interpolación es gradual, pero los puntos intermedios no siempre son prendas plausibles.
- **Costo:** el VAE tarda 1.6 veces más y tiene 7.8 veces más parámetros; a cambio, puede generar e interpolar.
- **Pruebas:** `uv run pytest -q` → 4 passed.

Los valores exactos dependen del hardware, la semilla y las épocas.

## Uso previsto y límites
- Autoencoder: compresión y reconstrucción. VAE: demostración académica de generación y exploración del espacio latente.
- No usar las muestras como datos sintéticos para otros modelos ni presentarlas como privadas u originales.
- No aplicar los modelos fuera de Fashion-MNIST (28×28, escala de grises) sin reentrenar y evaluar.
- La comparación no es entre iguales: 16 dimensiones latentes frente a 2 y arquitecturas distintas.
- La comprobación de originalidad cubre 16 muestras con distancia en píxeles, que no mide similitud perceptual.

Detalle en `MODEL_CARD.md` y `AI_USE_DECLARATION.md`.

## Dataset
Fashion-MNIST (Zalando Research), 60,000 imágenes de entrenamiento y 10,000 de prueba, 10 clases. Licencia MIT. Xiao, H., Rasul, K. y Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
