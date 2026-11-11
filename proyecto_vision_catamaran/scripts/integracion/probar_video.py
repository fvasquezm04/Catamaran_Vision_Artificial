import argparse
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from clasificador_color import ClasificadorColor


# ============================================================
# RUTAS Y CONFIGURACION
# ============================================================

# Este archivo debe estar en:
# sistema_pelotas/scripts/integracion/probar_video.py
RAIZ = Path(__file__).resolve().parents[2]

RUTA_YOLO = RAIZ / "modelos" / "yolo" / "best.pt"
RUTA_COLOR = RAIZ / "modelos" / "color" / "clasificador_color.joblib"
CARPETA_VIDEOS = RAIZ / "videos_prueba"
CARPETA_RESULTADOS = RAIZ / "resultados" / "videos"

UMBRAL_YOLO = 0.50
UMBRAL_COLOR = 0.60
MARGEN_CAJA = 0.03
TAMANO_MINIMO = 15

# Estabilizacion temporal del color por track.
HISTORIAL_MAXIMO = 12
MIN_PREDICCIONES_COLOR = 5
PROPORCION_MINIMA_VOTOS = 0.60

# Un track se elimina del estado visible si lleva demasiados frames sin aparecer.
MAX_FRAMES_AUSENTE = 30

COLORES_VALIDOS = ["amarillo", "azul", "naranja", "rojo", "verde"]

COLORES_DIBUJO = {
    "amarillo": (0, 255, 255),
    "azul": (255, 0, 0),
    "naranja": (0, 140, 255),
    "rojo": (0, 0, 255),
    "verde": (0, 200, 0),
    "desconocido": (160, 160, 160),
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def ajustar_caja(x1, y1, x2, y2, ancho_img, alto_img):
    """Agrega margen y limita la caja a los bordes del frame."""
    margen_x = (x2 - x1) * MARGEN_CAJA
    margen_y = (y2 - y1) * MARGEN_CAJA

    x1 = int(np.floor(x1 - margen_x))
    y1 = int(np.floor(y1 - margen_y))
    x2 = int(np.ceil(x2 + margen_x))
    y2 = int(np.ceil(y2 + margen_y))

    x1 = max(0, min(x1, ancho_img - 1))
    y1 = max(0, min(y1, alto_img - 1))
    x2 = max(x1 + 1, min(x2, ancho_img))
    y2 = max(y1 + 1, min(y2, alto_img))

    return x1, y1, x2, y2


def color_estable(historial):
    """Devuelve (color, proporcion_votos, probabilidad_media)."""
    predicciones_validas = [
        (color, prob)
        for color, prob in historial
        if color in COLORES_VALIDOS and prob >= UMBRAL_COLOR
    ]

    if len(predicciones_validas) < MIN_PREDICCIONES_COLOR:
        return "desconocido", 0.0, 0.0

    votos = Counter(color for color, _ in predicciones_validas)
    ganador, cantidad = votos.most_common(1)[0]
    proporcion = cantidad / len(predicciones_validas)

    probabilidades_ganador = [
        prob for color, prob in predicciones_validas if color == ganador
    ]
    probabilidad_media = float(np.mean(probabilidades_ganador))

    if proporcion < PROPORCION_MINIMA_VOTOS:
        return "desconocido", proporcion, probabilidad_media

    return ganador, proporcion, probabilidad_media


def dibujar_panel(frame, visibles, unicas, fps_actual):
    """Dibuja un panel compacto con los contadores."""
    lineas = [
        f"FPS: {fps_actual:.1f}",
        f"Visibles: {sum(visibles.values())}",
        f"Unicas: {sum(unicas.values())}",
    ]

    for color in COLORES_VALIDOS:
        lineas.append(
            f"{color.capitalize()}: visible {visibles[color]} | unica {unicas[color]}"
        )

    ancho_panel = 360
    alto_panel = 18 + 28 * len(lineas)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + ancho_panel, 10 + alto_panel), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for indice, texto in enumerate(lineas):
        y = 38 + indice * 28
        cv2.putText(
            frame,
            texto,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def guardar_resumen(ruta, datos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(datos, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Prueba YOLO + clasificador de color con tracking."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="video_prueba.mp4",
        help="Nombre dentro de videos_prueba o ruta completa del video.",
    )
    parser.add_argument(
        "--no-mostrar",
        action="store_true",
        help="Procesa sin abrir una ventana de OpenCV.",
    )
    parser.add_argument(
        "--conf-yolo",
        type=float,
        default=UMBRAL_YOLO,
        help="Confianza minima de YOLO. Valor predeterminado: 0.50.",
    )
    return parser.parse_args()


def resolver_video(valor):
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return CARPETA_VIDEOS / ruta


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    args = obtener_argumentos()
    video_entrada = resolver_video(args.video)

    if not RUTA_YOLO.is_file():
        raise FileNotFoundError(f"No se encontro el modelo YOLO: {RUTA_YOLO}")
    if not RUTA_COLOR.is_file():
        raise FileNotFoundError(f"No se encontro el modelo de color: {RUTA_COLOR}")
    if not video_entrada.is_file():
        raise FileNotFoundError(f"No se encontro el video: {video_entrada}")

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    video_salida = CARPETA_RESULTADOS / f"{video_entrada.stem}_procesado.mp4"
    resumen_salida = CARPETA_RESULTADOS / f"{video_entrada.stem}_resumen.json"

    print("Cargando modelos...")
    detector = YOLO(str(RUTA_YOLO))
    clasificador = ClasificadorColor(RUTA_COLOR)

    captura = cv2.VideoCapture(str(video_entrada))
    if not captura.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el video: {video_entrada}")

    fps_video = captura.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video <= 0:
        fps_video = 30.0

    escritor = None
    historial_color = defaultdict(lambda: deque(maxlen=HISTORIAL_MAXIMO))
    ultimo_frame_track = {}
    tracks_contados = set()
    color_contado_track = {}
    conteo_unicas = Counter({color: 0 for color in COLORES_VALIDOS})

    total_frames = 0
    total_detecciones = 0
    inicio_total = time.perf_counter()
    instante_fps = inicio_total
    frames_ventana_fps = 0
    fps_actual = 0.0

    while True:
        ok, frame = captura.read()
        if not ok:
            break

        total_frames += 1
        frames_ventana_fps += 1
        alto, ancho = frame.shape[:2]

        if escritor is None:
            escritor = cv2.VideoWriter(
                str(video_salida),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_video,
                (ancho, alto),
            )
            if not escritor.isOpened():
                raise RuntimeError(f"No se pudo crear el video: {video_salida}")

        # ByteTrack asigna un ID persistente a cada pelota.
        resultado = detector.track(
            frame,
            conf=args.conf_yolo,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        visibles = Counter({color: 0 for color in COLORES_VALIDOS})
        ids_presentes = set()

        if resultado.boxes is not None and len(resultado.boxes) > 0:
            cajas = resultado.boxes.xyxy.cpu().numpy()
            confianzas = resultado.boxes.conf.cpu().numpy()

            if resultado.boxes.id is None:
                ids = np.full(len(cajas), -1, dtype=int)
            else:
                ids = resultado.boxes.id.int().cpu().numpy()

            for (x1, y1, x2, y2), conf_yolo, track_id in zip(
                cajas, confianzas, ids
            ):
                track_id = int(track_id)
                x1, y1, x2, y2 = ajustar_caja(
                    x1, y1, x2, y2, ancho, alto
                )

                if (x2 - x1) < TAMANO_MINIMO or (y2 - y1) < TAMANO_MINIMO:
                    continue

                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue

                color_instantaneo, prob_instantanea = clasificador.predecir(recorte)
                total_detecciones += 1

                if prob_instantanea < UMBRAL_COLOR:
                    color_instantaneo = "desconocido"

                if track_id >= 0:
                    ids_presentes.add(track_id)
                    ultimo_frame_track[track_id] = total_frames
                    historial_color[track_id].append(
                        (color_instantaneo, prob_instantanea)
                    )
                    color, proporcion_votos, prob_color = color_estable(
                        historial_color[track_id]
                    )
                else:
                    color = color_instantaneo
                    proporcion_votos = 1.0
                    prob_color = prob_instantanea

                # La pelota se cuenta una sola vez cuando el color ya es estable.
                if track_id >= 0 and color in COLORES_VALIDOS:
                    visibles[color] += 1

                    if track_id not in tracks_contados:
                        tracks_contados.add(track_id)
                        color_contado_track[track_id] = color
                        conteo_unicas[color] += 1
                elif color in COLORES_VALIDOS:
                    visibles[color] += 1

                tono = COLORES_DIBUJO.get(color, COLORES_DIBUJO["desconocido"])
                id_texto = f"ID {track_id}" if track_id >= 0 else "sin ID"
                etiqueta = (
                    f"{id_texto} | {color} | "
                    f"color {prob_color:.2f} | YOLO {float(conf_yolo):.2f}"
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), tono, 2)
                cv2.putText(
                    frame,
                    etiqueta,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    tono,
                    2,
                    cv2.LINE_AA,
                )

        # Limpieza de historiales de tracks que ya desaparecieron.
        tracks_expirados = [
            track_id
            for track_id, ultimo in ultimo_frame_track.items()
            if total_frames - ultimo > MAX_FRAMES_AUSENTE
        ]
        for track_id in tracks_expirados:
            historial_color.pop(track_id, None)
            ultimo_frame_track.pop(track_id, None)

        ahora = time.perf_counter()
        lapso = ahora - instante_fps
        if lapso >= 1.0:
            fps_actual = frames_ventana_fps / lapso
            frames_ventana_fps = 0
            instante_fps = ahora

        dibujar_panel(frame, visibles, conteo_unicas, fps_actual)
        escritor.write(frame)

        if not args.no_mostrar:
            cv2.imshow("Pelotas con tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    captura.release()
    if escritor is not None:
        escritor.release()
    cv2.destroyAllWindows()

    duracion = time.perf_counter() - inicio_total
    fps_medio = total_frames / duracion if duracion > 0 else 0.0

    resumen = {
        "video_entrada": str(video_entrada),
        "video_salida": str(video_salida),
        "frames_procesados": total_frames,
        "detecciones_por_frame_acumuladas": total_detecciones,
        "pelotas_unicas_por_color": {
            color: int(conteo_unicas[color]) for color in COLORES_VALIDOS
        },
        "pelotas_unicas_total": int(sum(conteo_unicas.values())),
        "fps_video_original": float(fps_video),
        "fps_procesamiento_medio": float(fps_medio),
        "umbral_yolo": float(args.conf_yolo),
        "umbral_color": float(UMBRAL_COLOR),
        "nota": (
            "El conteo unico depende de los IDs de ByteTrack. Si una pelota "
            "sale durante mucho tiempo y vuelve a entrar, puede recibir otro ID."
        ),
    }

    guardar_resumen(resumen_salida, resumen)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Frames procesados: {total_frames}")
    print(f"Detecciones acumuladas: {total_detecciones}")
    print(f"FPS de procesamiento: {fps_medio:.1f}")
    print("Pelotas unicas por color:")
    for color in COLORES_VALIDOS:
        print(f"  {color}: {conteo_unicas[color]}")
    print(f"Video procesado: {video_salida}")
    print(f"Resumen JSON: {resumen_salida}")


if __name__ == "__main__":
    main()
