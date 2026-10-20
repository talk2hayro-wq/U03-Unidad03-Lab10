# Declaración de uso de inteligencia artificial

## Herramientas utilizadas
- **Claude (Anthropic)**, en la aplicación de Claude; identificador del modelo configurado: `claude-opus-5-5`.
- No se usaron otras herramientas de IA. Los modelos del laboratorio (Autoencoder y VAE) se entrenaron localmente con TensorFlow.

## Propósito de cada uso
| Uso | Propósito |
| --- | --- |
| Revisión del proyecto base | Describir estructura, arquitecturas y salidas del script, y anticipar limitaciones antes de ejecutar |
| Entorno | Explicar por qué la GPU AMD Radeon RX 6700 no puede usarse con TensorFlow 2.21 y elegir `--extra cpu` |
| Interpretación de salidas | Leer pérdidas, tiempos, tamaños, mensajes de consola, paneles, interpolación y capturas de la aplicación |
| Código | Agregar el guardado de `encoder.keras` en `scripts/train_ae_vae.py` y escribir `scripts/evaluate_panels.py` |
| Documentación | Redactar este documento, `MODEL_CARD.md` y el informe del laboratorio |

## Prompts o decisiones relevantes
- Se trabajó paso a paso con la guía del laboratorio: después de cada comando se compartieron la salida o las capturas y se pidió su interpretación.
- **Ejecución en CPU:** se descartó la opción GPU tras verificar que CUDA solo admite NVIDIA y que ROCm y DirectML no son compatibles con este equipo o con Python 3.12.
- **Paso 7 (paneles no seleccionados):** el script de la guía usaba los 12 primeros ejemplos y no comparaba con vecinos cercanos. Se decidió agregar un script de evaluación en lugar de seleccionar ejemplos a mano.
- **Criterio de originalidad:** comparar la distancia de cada muestra a su vecino de entrenamiento con la distancia de una imagen real de prueba al suyo, y la distancia entre muestras con la distancia entre imágenes reales.

## Elementos verificados por el equipo
- Todos los comandos se ejecutaron localmente (Windows 11, `C:\Users\Sobeit\lab10`) y las cifras del Model Card provienen de sus salidas.
- El reentrenamiento con el script modificado reprodujo exactamente las pérdidas de la primera corrida.
- El script de evaluación se probó antes con las mismas versiones de TensorFlow, Keras y NumPy; las pruebas del proyecto (`uv run pytest -q`) siguieron pasando.
- Las interpretaciones visuales (panel, vecinos, interpolación, capturas de la aplicación) se contrastaron con las imágenes generadas.

## Cambios realizados sobre resultados generados
- Los resultados numéricos no se modificaron; se reportan tal como los produjeron los scripts.
- Se corrigió la lectura inicial de las muestras: en el panel de 12 de la guía predominaba el calzado y en el de 16 (otra semilla) predominan prendas de torso; se reporta como inestabilidad con pocas muestras.
- Se aclaró que una distancia menor al entrenamiento no indica copia: las muestras son promedios borrosos y la distancia en píxeles favorece imágenes suaves.
- Las secciones del Model Card sin evidencia directa (por ejemplo, calidad perceptual de la generación) se declaran como limitación en lugar de completarse con supuestos.

## Responsabilidad asumida
El estudiante revisó el código agregado, ejecutó todos los comandos y verificó las cifras y las interpretaciones antes de incluirlas. Asume la responsabilidad por el contenido entregado, por las limitaciones declaradas y por no presentar las muestras sintéticas como privadas, originales ni representativas.
