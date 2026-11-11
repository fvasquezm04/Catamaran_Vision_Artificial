import threading
import time
import traceback
from pathlib import Path

import cv2

from services.camera_service import GoProCameraService, puerto_udp_libre
from services.vision_service import VisionService


RAIZ = Path(__file__).resolve().parents[1]
CARPETA_VIDEOS = RAIZ / "videos_prueba"

# Si YOLO tarda mas que esto en entregar un frame procesado (por ejemplo la
# primera inferencia), se muestra el frame crudo para que la web no quede
# en negro. Mientras YOLO responda a tiempo, SOLO se publican frames
# procesados (antes se alternaban crudo/procesado y las cajas casi no se veian).
SEGUNDOS_MAX_SIN_PROCESADO = 1.5


class PipelineService:
    """Lee GoPro o video local, ejecuta la vision y publica JPEG."""

    def __init__(self):
        self.vision = VisionService()

        self._captura = None
        self._camara_gopro = None
        self._hilo = None
        self._detener = threading.Event()
        self._lock = threading.RLock()
        self._lock_control = threading.Lock()

        self._ultimo_jpeg = None
        self._id_jpeg = 0
        self._instante_procesado = 0.0
        self._fuente = ""
        self._tipo_fuente = "detenida"
        self._ultimo_error = ""
        self._ejecutando = False
        self._repetir_video = True
        self._ultimo_numero_gopro = -1
        self._intervalo_video = 0.0

    # ------------------------------------------------------------------
    # Fuentes
    # ------------------------------------------------------------------
    def _resolver_video(self, video):
        ruta = Path(video)
        if not ruta.is_absolute():
            ruta = CARPETA_VIDEOS / ruta
        return ruta

    def iniciar_video(self, video, repetir=True):
        with self._lock_control:
            ruta = self._resolver_video(video)
            if not ruta.is_file():
                raise FileNotFoundError(f"No se encontro el video: {ruta}")

            self._detener_interno()
            self.vision.reiniciar_conteos()

            captura = cv2.VideoCapture(str(ruta))
            if not captura.isOpened():
                raise RuntimeError(f"OpenCV no pudo abrir el video: {ruta}")

            fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
            with self._lock:
                self._captura = captura
                self._camara_gopro = None
                self._fuente = ruta.name
                self._tipo_fuente = "video"
                self._repetir_video = repetir
                self._intervalo_video = 1.0 / fps if fps > 0 else 0.0
                self._ultimo_error = ""
                self._ultimo_jpeg = None
                self._ejecutando = True

            self._iniciar_hilo()

    def iniciar_gopro(self):
        """Conecta la GoPro, espera el primer frame y arranca la vision."""
        with self._lock_control:
            self._detener_interno()
            self.vision.reiniciar_conteos()

            camara = GoProCameraService()
            if not puerto_udp_libre(camara.puerto_udp):
                raise RuntimeError(
                    f"El puerto UDP {camara.puerto_udp} ya esta en uso. Cierra "
                    "otros programas que lean la GoPro (VLC, otra instancia de "
                    "uvicorn, el script camera_service.py) y vuelve a intentar."
                )

            camara.iniciar()

            if not camara.esperar_primer_frame(timeout=15.0):
                estado = camara.estado()
                camara.detener()
                raise RuntimeError(
                    estado.get("ultimo_error")
                    or "La GoPro no entrego frames dentro del tiempo esperado"
                )

            with self._lock:
                self._captura = None
                self._camara_gopro = camara
                self._fuente = "GoPro HERO10 (WiFi/UDP)"
                self._tipo_fuente = "gopro"
                self._repetir_video = False
                self._ultimo_numero_gopro = -1
                self._ultimo_error = ""
                self._ultimo_jpeg = None
                self._ejecutando = True

            self._iniciar_hilo()

    def _iniciar_hilo(self):
        self._detener.clear()
        self._instante_procesado = time.perf_counter()
        self._hilo = threading.Thread(
            target=self._procesar, daemon=True, name="pipeline-vision"
        )
        self._hilo.start()

    # ------------------------------------------------------------------
    # Bucle de procesamiento
    # ------------------------------------------------------------------
    def _leer_frame(self):
        """Devuelve (True, frame), (None, None) si no hay frame nuevo aun,
        o (False, None) si la fuente termino."""
        with self._lock:
            tipo = self._tipo_fuente
            captura = self._captura
            camara = self._camara_gopro
            repetir = self._repetir_video

        if tipo == "video":
            if captura is None:
                return False, None
            exito, frame = captura.read()
            if exito:
                return True, frame
            if repetir:
                captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.vision.reiniciar_conteos()
                exito, frame = captura.read()
                return exito, frame
            return False, None

        if tipo == "gopro":
            if camara is None:
                return False, None
            exito, frame, numero = camara.obtener_frame()
            if not exito or numero == self._ultimo_numero_gopro:
                time.sleep(0.005)
                return None, None
            self._ultimo_numero_gopro = numero
            return True, frame

        return False, None

    def _publicar(self, imagen, calidad, procesado):
        ok, buffer = cv2.imencode(
            ".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad]
        )
        if not ok:
            return
        with self._lock:
            self._ultimo_jpeg = buffer.tobytes()
            self._id_jpeg += 1
            if procesado:
                self._instante_procesado = time.perf_counter()

    def _procesar(self):
        error_impreso = None

        while not self._detener.is_set():
            inicio = time.perf_counter()
            exito, frame = self._leer_frame()

            if exito is None:
                # GoPro sin frame nuevo: si lleva rato sin llegar, avisar.
                with self._lock:
                    camara = self._camara_gopro
                if camara is not None:
                    estado = camara.estado()
                    if not estado["camara_conectada"]:
                        with self._lock:
                            self._ultimo_error = (
                                estado.get("ultimo_error")
                                or "La GoPro dejo de enviar frames"
                            )
                continue

            if not exito:
                with self._lock:
                    self._ultimo_error = "La fuente dejo de entregar frames"
                break

            # Solo si YOLO lleva mucho sin responder se muestra el crudo.
            if time.perf_counter() - self._instante_procesado > SEGUNDOS_MAX_SIN_PROCESADO:
                self._publicar(frame, 70, procesado=False)

            try:
                procesado = self.vision.procesar_frame(frame)
                self._publicar(procesado, 80, procesado=True)
                with self._lock:
                    self._ultimo_error = ""
                error_impreso = None
            except Exception as error:
                mensaje = f"Error en vision: {type(error).__name__}: {error}"
                with self._lock:
                    self._ultimo_error = mensaje
                if mensaje != error_impreso:
                    # Imprimir la traza completa en la consola de uvicorn.
                    traceback.print_exc()
                    error_impreso = mensaje
                self._publicar(frame, 70, procesado=False)
                time.sleep(0.2)

            # En video local se respeta su velocidad real.
            if self._tipo_fuente == "video" and self._intervalo_video:
                restante = self._intervalo_video - (time.perf_counter() - inicio)
                if restante > 0:
                    time.sleep(restante)

        with self._lock:
            self._ejecutando = False

    # ------------------------------------------------------------------
    # Consulta y control
    # ------------------------------------------------------------------
    def obtener_jpeg(self):
        """Devuelve (id, bytes) del ultimo JPEG publicado."""
        with self._lock:
            return self._id_jpeg, self._ultimo_jpeg

    def obtener_estado(self):
        estado = self.vision.obtener_estado()
        with self._lock:
            camara = self._camara_gopro
            estado.update(
                {
                    "fuente": self._fuente,
                    "tipo_fuente": self._tipo_fuente,
                    "pipeline_ejecutando": self._ejecutando,
                    "pipeline_error": self._ultimo_error,
                    "camara": camara.estado() if camara else None,
                }
            )
        return estado

    def reiniciar_conteos(self):
        self.vision.reiniciar_conteos()

    def detener(self):
        with self._lock_control:
            self._detener_interno()

    def _detener_interno(self):
        self._detener.set()

        hilo = self._hilo
        if hilo is not None and hilo.is_alive():
            hilo.join(timeout=5.0)

        with self._lock:
            captura = self._captura
            camara = self._camara_gopro
            self._captura = None
            self._camara_gopro = None
            self._hilo = None
            self._ejecutando = False
            self._tipo_fuente = "detenida"
            self._ultimo_numero_gopro = -1

        if captura is not None:
            captura.release()
        if camara is not None:
            camara.detener()
