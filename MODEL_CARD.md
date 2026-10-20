# Model Card · Autoencoder y VAE

## Modelos y versiones

|Modelo|Archivo|Descripción|
|-|-|-|
|Autoencoder|`models/autoencoder.keras`|Densa 784 → 16 (ReLU) → 784 (sigmoide); 25,888 parámetros|
|VAE · codificador|`models/encoder.keras`|784 → 128 (ReLU) → media y log-varianza (2 dimensiones); 100,996 parámetros|
|VAE · decodificador|`models/decoder.keras`|2 → 128 (ReLU) → 784 (sigmoide); 101,520 parámetros|

* Proyecto: `inf8239-u03-generative` 0.1.0 (INF-8239, Unidad 03, LAB10).
* Entorno: Python 3.12.15, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3 (fijados en `uv.lock`).
* Entrenamiento: `scripts/train\_ae\_vae.py --epochs-ae 6 --epochs-vae 8`, semilla 42, operaciones deterministas (`TF\_DETERMINISTIC\_OPS=1`). Dos corridas produjeron pérdidas idénticas en cada época.
* Cambio respecto del proyecto base: `train\_ae\_vae.py` guarda también `encoder.keras`, necesario para la evaluación de `scripts/evaluate\_panels.py`.

## Uso previsto y usuarios

Demostración académica para el curso INF-8239: comparar compresión (Autoencoder) y generación (VAE), explorar un espacio latente de 2 dimensiones con la aplicación `app/streamlit\_app.py` y discutir costo y responsabilidad. Usuarios: estudiantes y docentes.

## Usos fuera de alcance

* Diseño, catálogo o venta de prendas reales; las muestras son siluetas borrosas de 28×28 en escala de grises.
* Generar datos sintéticos presentados como privados, originales o representativos de una población.
* Clasificar, detectar defectos o tomar decisiones sobre productos o personas.
* Aplicar los modelos a otras imágenes (fotos, color, otras resoluciones) sin reentrenar y validar.

## Dataset, licencia y particiones

* **Dataset:** Fashion-MNIST (Zalando Research): 70,000 imágenes de 28×28 en escala de grises de 10 clases de prendas y calzado. Se descarga con `tf.keras.datasets.fashion\_mnist` (30.9 MB).
* **Licencia y cita:** licencia MIT. Xiao, H., Rasul, K. y Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
* **Particiones:** 60,000 de entrenamiento y 10,000 de prueba (oficiales). El Autoencoder usa el 10 % del entrenamiento como validación (54,000 / 6,000); el VAE entrena con las 60,000. La prueba solo se usa en la evaluación final.
* **Preprocesamiento:** valores escalados a \[0, 1] y canal agregado (`normalize\_images`, validado por `validate\_images`). Las etiquetas no se usan en el entrenamiento; solo para describir resultados.

## Arquitectura y espacio latente

* **Autoencoder:** espacio latente de 16 dimensiones sin regularizar; pérdida de entropía cruzada binaria por píxel. Reconstruye, pero decodificar un punto arbitrario no garantiza una imagen plausible.
* **VAE:** espacio latente de 2 dimensiones; pérdida = reconstrucción (entropía cruzada sumada sobre 784 píxeles) + divergencia KL respecto de una normal estándar. Permite generar muestreando z \~ N(0, I).
* **Organización del latente** (aplicación, puntos de −3 a 3): el eje Y separa prendas de torso (arriba) de calzado (abajo); el eje X ordena de mangas largas y botines (izquierda) a mangas cortas, zapatillas y pantalones (derecha). La organización surge sin etiquetas.
* **Interpolación** de (−2, −1) a (2, 1): transición gradual botín → torso → pantalón; los puntos intermedios superponen calzado y camisa y no son prendas plausibles.

## Métricas de reconstrucción y generación

Reconstrucción en las 10,000 imágenes de prueba (`reports/evaluation\_metrics.json`):

|Modelo|MSE|Entropía cruzada por píxel|Por imagen|KL|
|-|-|-|-|-|
|Autoencoder|0.0203|0.3068|240.5|—|
|VAE|0.0330|0.3405|267.0|6.30|

* Entrenamiento: Autoencoder 0.3060 (validación 0.3071); VAE reconstrucción 267.3 y KL 6.26 en la época 8. Los valores de prueba son casi iguales: no hay sobreajuste. Ninguno convergió por completo (las pérdidas seguían bajando).
* MSE por clase: el Autoencoder es mejor en las 10 clases. Las más difíciles son bolso (AE 0.0295, VAE 0.0621) y sandalia (0.0297, 0.0382); las más fáciles, pantalón (0.0139, 0.0230) y zapatilla (0.0153, 0.0195).
* Fallos del VAE en el panel de 16 ejemplos elegidos con semilla 42: los bolsos se reconstruyen como prendas de torso y las sandalias como botines.
* Generación: no hay una métrica de calidad perceptual; la evaluación es visual y por distancias (sección siguiente).

## Diversidad y vecinos cercanos

16 muestras del VAE (z \~ N(0, I), semilla 43) comparadas con las 60,000 imágenes de entrenamiento por distancia euclidiana en píxeles:

|Medida|Muestras|Referencia con imágenes reales|
|-|-|-|
|Distancia al vecino de entrenamiento|media 2.79, mínima 1.61|3.89 (prueba a entrenamiento)|
|Distancia entre las 16 imágenes|media 6.35, mínima 0.73|media 11.66, mínima 6.25|
|Clases de los vecinos|Camisa 9, Suéter 3, Camiseta 1, Zapatilla 1, Bolso 1, Botín 1|10 clases|

* **No se observaron copias:** cada muestra es una silueta borrosa sin los detalles de su vecino. La menor distancia frente a la referencia se explica porque las muestras son promedios suaves y oscuros, no por memorización.
* **Diversidad baja:** las muestras están a la mitad de distancia entre sí que las imágenes reales y hay un par casi duplicado.
* **Sesgo de clases:** 13 de 16 muestras se parecen a prendas de torso; no aparecen pantalones, vestidos, abrigos ni sandalias. Con 12 muestras de otra semilla predominaba el calzado: la proporción es inestable con pocas muestras.
* Límite del método: la distancia en píxeles no mide similitud perceptual; una muestra podría parecerse a un ejemplo de forma no capturada por esta medida.

## Hardware, tiempo y tamaño

|Concepto|Autoencoder|VAE|
|-|-|-|
|Hardware|CPU, Windows 11 (22621), TensorFlow 2.21.0 sin GPU|Igual|
|Tiempo de entrenamiento|5.5–5.9 s (6 épocas)|9.0–9.1 s (8 épocas)|
|Parámetros|25,888|202,516 (decodificador 101,520)|
|Archivo|331 KB (incluye estado del optimizador Adam)|`decoder.keras` 416 KB; `encoder.keras` guardado aparte|

La GPU AMD Radeon RX 6700 del equipo no se usó: `--extra gpu` instala CUDA (solo NVIDIA), TensorFlow no usa GPU en Windows nativo desde la versión 2.11, ROCm en WSL no soporta la serie RX 6000 y el plugin DirectML está pausado y no admite Python 3.12. Los tiempos varían con el equipo y la carga.

## Limitaciones, riesgos y supervisión

* **Comparación no equivalente:** el Autoencoder tiene 16 dimensiones latentes y ninguna capa oculta; el VAE, 2 dimensiones y capas de 128. Parte de la diferencia se debe a la arquitectura.
* **Calidad:** muestras borrosas, sin texturas, con mezclas ambiguas en zonas intermedias del latente; la aplicación permite explorar puntos extremos (±3) que la normal estándar casi nunca muestrea.
* **Privacidad y originalidad:** una imagen sintética no es automáticamente privada, original, verdadera ni libre de sesgo. Aquí no se observaron copias, pero la evaluación cubre solo 16 muestras.
* **Sesgo:** Fashion-MNIST representa productos de un catálogo europeo, centrados y sin fondo; las muestras heredan esa selección y no representan prendas de otros contextos.
* **Supervisión:** cualquier uso de las muestras fuera del aula requiere revisión humana, declarar el origen de los datos y repetir la comparación con vecinos cercanos.
* **Publicación:** `.gitignore` excluye `models/\*.keras`; para desplegar la aplicación hay que incluir `decoder.keras` en el repositorio o generarlo antes de iniciar la aplicación, nunca en cada visita.

