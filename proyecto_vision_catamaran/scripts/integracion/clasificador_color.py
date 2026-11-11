from pathlib import Path

import cv2
import joblib
import numpy as np

# ============================================================
# PARÁMETROS (idénticos a los del entrenamiento, no cambiar)
# ============================================================
TAMANO_NORMALIZADO = 128
FACTOR_ELIPSE = 0.86
SATURACION_MINIMA = 30
VALOR_MINIMO = 20
SATURACION_REFLEJO = 55
VALOR_REFLEJO = 235
BINS_H = 18
BINS_S = 12
BINS_V = 12
BINS_LAB = 12


def crear_mascara_eliptica(alto, ancho):
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    centro = (ancho // 2, alto // 2)
    ejes = (
        max(1, int(ancho * FACTOR_ELIPSE / 2)),
        max(1, int(alto * FACTOR_ELIPSE / 2)),
    )
    cv2.ellipse(mascara, centro, ejes, 0, 0, 360, 255, -1)
    return mascara


def histograma_normalizado(canal, mascara, bins, rango):
    histograma = cv2.calcHist([canal], [0], mascara, [bins], rango)
    histograma = histograma.flatten().astype(np.float32)
    suma = histograma.sum()
    if suma > 0:
        histograma /= suma
    return histograma


def estadisticas_robustas(valores):
    if len(valores) == 0:
        return np.zeros(7, dtype=np.float32)

    percentiles = np.percentile(valores, [10, 25, 50, 75, 90])
    return np.array(
        [
            np.mean(valores),
            np.std(valores),
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            percentiles[4],
        ],
        dtype=np.float32,
    )


def extraer_features(imagen):
    """Recibe un recorte en BGR y devuelve el vector de 124 features."""
    if imagen is None or imagen.size == 0:
        raise ValueError("Imagen vacía")

    imagen = cv2.resize(
        imagen,
        (TAMANO_NORMALIZADO, TAMANO_NORMALIZADO),
        interpolation=cv2.INTER_AREA,
    )
    imagen_suave = cv2.GaussianBlur(imagen, (5, 5), 0)

    hsv = cv2.cvtColor(imagen_suave, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen_suave, cv2.COLOR_BGR2LAB)

    alto, ancho = imagen.shape[:2]
    mascara_elipse = crear_mascara_eliptica(alto, ancho)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    mascara_color = (s >= SATURACION_MINIMA) & (v >= VALOR_MINIMO)
    mascara_reflejo = (s <= SATURACION_REFLEJO) & (v >= VALOR_REFLEJO)

    mascara_valida = (mascara_elipse > 0) & mascara_color & (~mascara_reflejo)

    cantidad_elipse = np.count_nonzero(mascara_elipse)
    cantidad_valida = np.count_nonzero(mascara_valida)

    if cantidad_valida < max(100, int(cantidad_elipse * 0.10)):
        mascara_valida = (
            (mascara_elipse > 0) & (v >= 10) & (~mascara_reflejo)
        )

    mascara_uint8 = mascara_valida.astype(np.uint8) * 255

    features = []

    for canal in [h, s, v, l, a, b]:
        features.extend(estadisticas_robustas(canal[mascara_valida]))

    features.extend(histograma_normalizado(h, mascara_uint8, BINS_H, [0, 180]))
    features.extend(histograma_normalizado(s, mascara_uint8, BINS_S, [0, 256]))
    features.extend(histograma_normalizado(v, mascara_uint8, BINS_V, [0, 256]))

    for canal in [l, a, b]:
        features.extend(
            histograma_normalizado(canal, mascara_uint8, BINS_LAB, [0, 256])
        )

    total_elipse = max(1, np.count_nonzero(mascara_elipse))

    proporcion_valida = np.count_nonzero(mascara_valida) / total_elipse
    proporcion_reflejo = (
        np.count_nonzero((mascara_elipse > 0) & mascara_reflejo) / total_elipse
    )
    proporcion_baja_saturacion = (
        np.count_nonzero((mascara_elipse > 0) & (s < SATURACION_MINIMA))
        / total_elipse
    )
    proporcion_oscura = (
        np.count_nonzero((mascara_elipse > 0) & (v < 40)) / total_elipse
    )

    features.extend(
        [
            proporcion_valida,
            proporcion_reflejo,
            proporcion_baja_saturacion,
            proporcion_oscura,
        ]
    )

    return np.asarray(features, dtype=np.float32)


class ClasificadorColor:
    def __init__(self, ruta_joblib):
        paquete = joblib.load(Path(ruta_joblib))
        self.modelo = paquete["modelo"]
        self.cantidad_features = paquete["cantidad_features"]

    def predecir(self, recorte_bgr):
        """Devuelve (color, probabilidad) para un recorte BGR."""
        vector = extraer_features(recorte_bgr)

        if vector.shape[0] != self.cantidad_features:
            raise ValueError(
                f"Se esperaban {self.cantidad_features} features "
                f"y se obtuvieron {vector.shape[0]}"
            )

        probabilidades = self.modelo.predict_proba(vector.reshape(1, -1))[0]
        indice = int(np.argmax(probabilidades))

        return str(self.modelo.classes_[indice]), float(probabilidades[indice])