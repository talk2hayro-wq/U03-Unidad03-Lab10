# Declaración de uso de inteligencia artificial

## Herramientas utilizadas

* **Claude (Anthropic)**, en la aplicación de Claude; identificador del modelo configurado: `claude-opus-5-5`.
* No se usaron otras herramientas de IA. Los modelos del laboratorio (Autoencoder y VAE) se entrenaron localmente con TensorFlow.

## Propósito de cada uso

|Uso|Propósito|
|-|-|
|Entorno|Explicar por qué la GPU AMD Radeon RX 6700 no puede usarse con TensorFlow 2.21 y elegir `--extra cpu`|
|Código|Agregar el guardado de `encoder.keras` en `scripts/train\_ae\_vae.py` y escribir `scripts/evaluate\_panels.py`|

## Prompts o decisiones relevantes

* **Ejecución en CPU:** se descartó la opción GPU tras verificar que CUDA solo admite NVIDIA y que ROCm y DirectML no son compatibles con este equipo o con Python 3.12.
* **Paso 7 (paneles no seleccionados):** el script de la guía usaba los 12 primeros ejemplos y no comparaba con vecinos cercanos. Se decidió agregar un script de evaluación en lugar de seleccionar ejemplos a mano.
* **Criterio de originalidad:** comparar la distancia de cada muestra a su vecino de entrenamiento con la distancia de una imagen real de prueba al suyo, y la distancia entre muestras con la distancia entre imágenes reales.

## Elementos verificados por el equipo

* Todos los comandos se ejecutaron localmente (Windows 11, `C:\\Users\\Sobeit\\lab10`) y las cifras del Model Card provienen de sus salidas.
* El reentrenamiento con el script modificado reprodujo exactamente las pérdidas de la primera corrida.
* El script de evaluación se probó antes con las mismas versiones de TensorFlow, Keras y NumPy; las pruebas del proyecto (`uv run pytest -q`) siguieron pasando.

## Cambios realizados sobre resultados generados

* Los resultados numéricos no se modificaron; se reportan tal como los produjeron los scripts.
* Se corrigió la lectura inicial de las muestras: en el panel de 12 de la guía predominaba el calzado y en el de 16 (otra semilla) predominan prendas de torso; se reporta como inestabilidad con pocas muestras.
* Se aclaró que una distancia menor al entrenamiento no indica copia: las muestras son promedios borrosos y la distancia en píxeles favorece imágenes suaves.

## Responsabilidad asumida

El estudiante revisó el código agregado, ejecutó todos los comandos y verificó las cifras y las interpretaciones antes de incluirlas. Asume la responsabilidad por el contenido entregado, por las limitaciones declaradas y por no presentar las muestras sintéticas como privadas, originales ni representativas.

