import os
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parents[1]
RUTA_INTEGRACION = RAIZ / "scripts" / "integracion"

if str(RUTA_INTEGRACION) not in sys.path:
    sys.path.insert(0, str(RUTA_INTEGRACION))

from clasificador_color import ClasificadorColor


COLORES = ("amarillo", "azul", "naranja", "rojo", "verde")

COLORES_BGR = {
    "amarillo": (0, 255, 255),
    "azul": (255, 0, 0),
    "naranja": (0, 140, 255),
    "rojo": (0, 0, 255),
    "verde": (0, 200, 0),
    "desconocido": (160, 160, 160),
}


class VisionService:
    """Ejecuta YOLO, ByteTrack y el clasificador de color."""

    def __init__(
        self,
        ruta_yolo=None,
        ruta_color=None,
        confianza_yolo=None,
        confianza_color=None,
        historial_maximo=12,
        predicciones_minimas=5,
        proporcion_votos_minima=0.60,
        margen_caja=0.03,
        tamano_minimo=15,
        max_frames_ausente=30,
        tamano_yolo=None,
        dispositivo=None,
    ):
        self.ruta_yolo = Path(
            ruta_yolo or RAIZ / "modelos" / "yolo" / "best.pt"
        )
        self.ruta_color = Path(
            ruta_color
            or RAIZ / "modelos" / "color" / "clasificador_color.joblib"
        )

        if not self.ruta_yolo.is_file():
            raise FileNotFoundError(
                f"No se encontro YOLO: {self.ruta_yolo}"
            )

        if not self.ruta_color.is_file():
            raise FileNotFoundError(
                f"No se encontro el clasificador: {self.ruta_color}"
            )

        # Valores configurables desde .env sin tocar codigo.
        self.confianza_yolo = float(
            confianza_yolo or os.getenv("YOLO_CONFIANZA", "0.50")
        )
        self.confianza_color = float(
            confianza_color or os.getenv("COLOR_CONFIANZA", "0.60")
        )
        self.historial_maximo = historial_maximo
        self.predicciones_minimas = predicciones_minimas
        self.proporcion_votos_minima = proporcion_votos_minima
        self.margen_caja = margen_caja
        self.tamano_minimo = tamano_minimo
        self.max_frames_ausente = max_frames_ausente
        # El detector se entreno con imgsz=640. Con 320 las pelotas quedan
        # a la mitad de tamano y YOLO deja de detectarlas en la GoPro.
        self.tamano_yolo = int(tamano_yolo or os.getenv("YOLO_IMGSZ", "640"))

        # Seleccion automatica de GPU NVIDIA si PyTorch tiene CUDA.
        if dispositivo is None:
            try:
                import torch

                self.dispositivo = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                self.dispositivo = "cpu"
        else:
            self.dispositivo = dispositivo

        self.usar_fp16 = self.dispositivo != "cpu"

        print(
            f"Cargando YOLO en {self.dispositivo}, "
            f"imgsz={self.tamano_yolo}, fp16={self.usar_fp16}..."
        )
        self.detector = YOLO(str(self.ruta_yolo))

        print("Cargando clasificador de color...")
        self.clasificador = ClasificadorColor(self.ruta_color)

        # Lock corto para leer y actualizar el estado desde FastAPI.
        self._lock_estado = threading.RLock()

        # Impide ejecutar dos inferencias simultaneas sobre el mismo YOLO.
        self._lock_inferencia = threading.Lock()

        self._historial = defaultdict(
            lambda: deque(maxlen=self.historial_maximo)
        )
        self._ultimo_frame_track = {}
        self._tracks_contados = set()
        self._color_por_track = {}
        self._conteo_unicas = Counter(
            {color: 0 for color in COLORES}
        )
        self._conteo_visibles = Counter(
            {color: 0 for color in COLORES}
        )
        self._detecciones_visibles = []

        self._numero_frame = 0
        self._ultimo_frame_procesado = None
        self._fps_procesamiento = 0.0
        self._latencia_ms = 0.0
        self._ultimo_error = ""
        self._inicio_sesion = time.time()

        self._calentar_modelo()

    def _calentar_modelo(self):
        """Primera inferencia al arrancar el servidor.

        Asi la primera imagen de la GoPro no tarda varios segundos y, si
        falta alguna dependencia del tracker (paquete lap), el error sale
        en la consola al iniciar y no en silencio durante el stream.
        """
        print("Calentando YOLO + ByteTrack...")
        inicio = time.perf_counter()
        imagen = np.zeros((480, 848, 3), dtype=np.uint8)
        self.detector.track(
            imagen,
            conf=self.confianza_yolo,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.tamano_yolo,
            device=self.dispositivo,
            half=self.usar_fp16,
            verbose=False,
        )
        self.clasificador.predecir(np.full((40, 40, 3), 128, dtype=np.uint8))
        self.reiniciar_conteos()
        print(f"Modelos listos en {time.perf_counter() - inicio:.1f} s")

    def _ajustar_caja(self, caja, ancho, alto):
        x1, y1, x2, y2 = map(float, caja)

        margen_x = (x2 - x1) * self.margen_caja
        margen_y = (y2 - y1) * self.margen_caja

        x1 = max(
            0,
            min(int(np.floor(x1 - margen_x)), ancho - 1),
        )
        y1 = max(
            0,
            min(int(np.floor(y1 - margen_y)), alto - 1),
        )
        x2 = max(
            x1 + 1,
            min(int(np.ceil(x2 + margen_x)), ancho),
        )
        y2 = max(
            y1 + 1,
            min(int(np.ceil(y2 + margen_y)), alto),
        )

        return x1, y1, x2, y2

    def _obtener_color_estable(self, track_id):
        predicciones = [
            (color, probabilidad)
            for color, probabilidad in self._historial[track_id]
            if color in COLORES
            and probabilidad >= self.confianza_color
        ]

        if len(predicciones) < self.predicciones_minimas:
            return "desconocido", 0.0, 0.0

        votos = Counter(color for color, _ in predicciones)
        ganador, cantidad = votos.most_common(1)[0]
        proporcion = cantidad / len(predicciones)

        probabilidades = [
            probabilidad
            for color, probabilidad in predicciones
            if color == ganador
        ]
        probabilidad_media = float(np.mean(probabilidades))

        if proporcion < self.proporcion_votos_minima:
            return "desconocido", proporcion, probabilidad_media

        return ganador, proporcion, probabilidad_media

    def _limpiar_tracks_expirados(self, numero_frame):
        expirados = [
            track_id
            for track_id, ultimo in self._ultimo_frame_track.items()
            if numero_frame - ultimo > self.max_frames_ausente
        ]

        for track_id in expirados:
            self._historial.pop(track_id, None)
            self._ultimo_frame_track.pop(track_id, None)

    def _dibujar_panel(
        self,
        frame,
        fps,
        latencia_ms,
        visibles,
        unicas,
    ):
        lineas = [
            f"FPS vision: {fps:.1f}",
            f"Latencia: {latencia_ms:.1f} ms",
            f"Visibles: {sum(visibles.values())}",
            f"Unicas: {sum(unicas.values())}",
        ]

        for color in COLORES:
            lineas.append(
                f"{color.capitalize()}: "
                f"{visibles[color]} visibles | "
                f"{unicas[color]} unicas"
            )

        ancho_panel = 410
        alto_panel = 18 + 27 * len(lineas)
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + ancho_panel, 10 + alto_panel),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for indice, texto in enumerate(lineas):
            cv2.putText(
                frame,
                texto,
                (22, 38 + indice * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def procesar_frame(self, frame):
        """Procesa un frame BGR y devuelve una copia anotada."""
        if frame is None or frame.size == 0:
            raise ValueError("El frame esta vacio")

        inicio = time.perf_counter()
        alto, ancho = frame.shape[:2]
        salida = frame.copy()

        # La inferencia no usa el lock del estado. La API puede seguir
        # respondiendo mientras YOLO procesa un frame lento.
        with self._lock_inferencia:
            try:
                resultado = self.detector.track(
                    frame,
                    conf=self.confianza_yolo,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=self.tamano_yolo,
                    device=self.dispositivo,
                    half=self.usar_fp16,
                    verbose=False,
                )[0]
            except Exception as error:
                with self._lock_estado:
                    self._ultimo_error = str(error)
                raise

        with self._lock_estado:
            self._numero_frame += 1
            numero_frame = self._numero_frame

        visibles = Counter({color: 0 for color in COLORES})
        detecciones = []

        if resultado.boxes is not None and len(resultado.boxes) > 0:
            cajas = resultado.boxes.xyxy.cpu().numpy()
            confianzas = resultado.boxes.conf.cpu().numpy()

            if resultado.boxes.id is not None:
                ids = resultado.boxes.id.int().cpu().numpy()
            else:
                ids = np.full(len(cajas), -1, dtype=int)

            for caja, conf_yolo, track_id in zip(
                cajas,
                confianzas,
                ids,
            ):
                track_id = int(track_id)
                x1, y1, x2, y2 = self._ajustar_caja(
                    caja,
                    ancho,
                    alto,
                )

                if (
                    x2 - x1 < self.tamano_minimo
                    or y2 - y1 < self.tamano_minimo
                ):
                    continue

                recorte = frame[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue

                color_instantaneo, prob_instantanea = (
                    self.clasificador.predecir(recorte)
                )
                prob_instantanea = float(prob_instantanea)

                if prob_instantanea < self.confianza_color:
                    color_instantaneo = "desconocido"

                with self._lock_estado:
                    if track_id >= 0:
                        self._ultimo_frame_track[track_id] = (
                            numero_frame
                        )
                        self._historial[track_id].append(
                            (
                                color_instantaneo,
                                prob_instantanea,
                            )
                        )
                        color, proporcion, prob_color = (
                            self._obtener_color_estable(track_id)
                        )
                    else:
                        color = color_instantaneo
                        proporcion = 1.0
                        prob_color = prob_instantanea

                    if color in COLORES:
                        visibles[color] += 1

                        if (
                            track_id >= 0
                            and track_id
                            not in self._tracks_contados
                        ):
                            self._tracks_contados.add(track_id)
                            self._color_por_track[track_id] = color
                            self._conteo_unicas[color] += 1

                deteccion = {
                    "track_id": track_id,
                    "color": color,
                    "probabilidad_color": round(
                        float(prob_color),
                        4,
                    ),
                    "proporcion_votos": round(
                        float(proporcion),
                        4,
                    ),
                    "confianza_yolo": round(
                        float(conf_yolo),
                        4,
                    ),
                    "caja": [x1, y1, x2, y2],
                }
                detecciones.append(deteccion)

                tono = COLORES_BGR.get(
                    color,
                    COLORES_BGR["desconocido"],
                )
                etiqueta = (
                    f"ID {track_id} | {color} | "
                    f"C {prob_color:.2f} | "
                    f"Y {float(conf_yolo):.2f}"
                )

                cv2.rectangle(
                    salida,
                    (x1, y1),
                    (x2, y2),
                    tono,
                    2,
                )
                cv2.putText(
                    salida,
                    etiqueta,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    tono,
                    2,
                    cv2.LINE_AA,
                )

        lapso = time.perf_counter() - inicio
        latencia_ms = lapso * 1000.0
        fps = 1.0 / lapso if lapso > 0 else 0.0

        with self._lock_estado:
            self._conteo_visibles = visibles
            self._detecciones_visibles = detecciones
            self._limpiar_tracks_expirados(numero_frame)
            self._fps_procesamiento = fps
            self._latencia_ms = latencia_ms
            self._ultimo_error = ""
            unicas_panel = self._conteo_unicas.copy()

        self._dibujar_panel(
            salida,
            fps,
            latencia_ms,
            visibles,
            unicas_panel,
        )

        with self._lock_estado:
            self._ultimo_frame_procesado = salida.copy()

        return salida

    def obtener_ultimo_frame(self):
        with self._lock_estado:
            if self._ultimo_frame_procesado is None:
                return None
            return self._ultimo_frame_procesado.copy()

    def obtener_estado(self):
        with self._lock_estado:
            return {
                "numero_frame": self._numero_frame,
                "fps_procesamiento": round(
                    self._fps_procesamiento,
                    2,
                ),
                "latencia_ms": round(self._latencia_ms, 2),
                "pelotas_visibles": {
                    color: int(self._conteo_visibles[color])
                    for color in COLORES
                },
                "pelotas_visibles_total": int(
                    sum(self._conteo_visibles.values())
                ),
                "pelotas_unicas": {
                    color: int(self._conteo_unicas[color])
                    for color in COLORES
                },
                "pelotas_unicas_total": int(
                    sum(self._conteo_unicas.values())
                ),
                "detecciones": list(
                    self._detecciones_visibles
                ),
                "segundos_sesion": round(
                    time.time() - self._inicio_sesion,
                    1,
                ),
                "ultimo_error": self._ultimo_error,
                "dispositivo": str(
                    getattr(self, "dispositivo", "cpu")
                ),
                "tamano_yolo": int(
                    getattr(self, "tamano_yolo", 320)
                ),
            }

    def reiniciar_conteos(self):
        """Reinicia conteos e historiales de la sesion."""
        with self._lock_estado:
            self._historial.clear()
            self._ultimo_frame_track.clear()
            self._tracks_contados.clear()
            self._color_por_track.clear()
            self._conteo_unicas = Counter(
                {color: 0 for color in COLORES}
            )
            self._conteo_visibles = Counter(
                {color: 0 for color in COLORES}
            )
            self._detecciones_visibles = []
            self._numero_frame = 0
            self._inicio_sesion = time.time()

        # Ultralytics crea su predictor en la primera inferencia.
        # Eliminamos el atributo trackers para que se cree de nuevo en la
        # siguiente llamada a track(). Se hace bajo el lock de inferencia:
        # borrarlo mientras otro hilo ejecuta track() lanza AttributeError.
        with self._lock_inferencia:
            predictor = getattr(self.detector, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                try:
                    delattr(predictor, "trackers")
                except Exception:
                    pass
