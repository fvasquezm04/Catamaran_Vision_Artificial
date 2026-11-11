"""
Entrenamiento del detector de pelotas con YOLO11n

Antes de ejecutar (solo una vez):
    pip install ultralytics

Uso:
    python scripts/yolo/entrenar_pelotas.py
"""

import torch
from ultralytics import YOLO

# ---------------- CONFIGURACIÓN ----------------
# Ruta completa a tu data.yaml (usa / en vez de \)
from pathlib import Path
DATA_YAML = str(Path(__file__).resolve().parents[2] / "configuracion" / "data.yaml")

EPOCHS = 50        # vueltas completas sobre el dataset
IMG_SIZE = 640     # tamaño al que se redimensionan las imágenes
BATCH = 4          # bájalo a 4 si te sale error de memoria
NOMBRE = "resultado_pelotas"  # nombre de la carpeta donde se guardan los resultados
# -----------------------------------------------

def main():
    # Usa GPU si hay una NVIDIA disponible, si no, CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {'GPU' if device == 0 else 'CPU (va a ser más lento)'}")

    # Modelo base preentrenado (se descarga solo la primera vez)
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        name=NOMBRE,
        patience=15,   # para si no mejora en 15 épocas seguidas
        workers=0,     # en Windows es mejor dejarlo bajo
    )

    # Evaluación final sobre test (la carpeta 'val' del yaml)
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")


# En Windows este 'if' es obligatorio, si no da error de multiprocessing
if __name__ == "__main__":
    main()