"""Captura del stream de previsualizacion de una GoPro HERO10 por WiFi.

Flujo (API Open GoPro, camara en modo punto de acceso WiFi):
    1. El PC esta conectado a la red WiFi de la GoPro (IP de la camara 10.5.5.9).
    2. GET http://10.5.5.9:8080/gopro/camera/stream/start
       -> la camara empieza a enviar MPEG-TS (H.264) por UDP al puerto 8554
          del PC que hizo la peticion.
    3. OpenCV/FFmpeg escucha en udp://0.0.0.0:8554 y decodifica.
    4. Cada pocos segundos se envia /gopro/camera/keep_alive para que la
       camara no se duerma.

Un hilo lee el stream sin parar y guarda solo el frame mas reciente; asi el
procesamiento (YOLO) nunca trabaja con frames atrasados en un buffer.
"""

import os
import socket
import threading
import time
from typing import Optional

# Opciones de FFmpeg para baja latencia. Deben existir ANTES de abrir la
# captura. Se respeta un valor que el usuario haya definido manualmente.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|analyzeduration;2000000|probesize;2000000",
)

import cv2  # noqa: E402
import requests  # noqa: E402


class GoProCameraService:
    def __init__(
        self,
        ip_camara: Optional[str] = None,
        puerto_http: int = 8080,
        puerto_udp: Optional[int] = None,
        intervalo_keep_alive: float = 3.0,
        timeout_http: float = 5.0,
        timeout_sin_frames: float = 5.0,
    ) -> None:
        self.ip_camara = ip_camara or os.getenv("GOPRO_IP", "10.5.5.9")
        self.puerto_http = puerto_http
        self.puerto_udp = int(puerto_udp or os.getenv("GOPRO_PUERTO_UDP", "8554"))
        self.intervalo_keep_alive = intervalo_keep_alive
        self.timeout_http = timeout_http
        self.timeout_sin_frames = timeout_sin_frames

        self.url_base = f"http://{self.ip_camara}:{self.puerto_http}"
        # timeout (microsegundos) evita que read() se quede bloqueado para
        # siempre si la camara deja de enviar paquetes.
        self.url_stream = (
            f"udp://@0.0.0.0:{self.puerto_udp}"
            "?overrun_nonfatal=1&fifo_size=5000000&timeout=5000000"
        )

        # trust_env=False: ignora proxies del sistema; la GoPro es una red local.
        self._http = requests.Session()
        self._http.trust_env = False

        self._captura: Optional[cv2.VideoCapture] = None
        self._hilo_captura: Optional[threading.Thread] = None
        self._hilo_keep_alive: Optional[threading.Thread] = None
        self._detener = threading.Event()
        self._lock_frame = threading.Lock()

        self._ultimo_frame = None
        self._numero_frame = 0
        self._instante_ultimo_frame = 0.0
        self._ultimo_error = ""
        self._fps_entrada = 0.0
        self._conectada = False
        self._resolucion = None

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _get(self, ruta: str, timeout: Optional[float] = None):
        return self._http.get(
            f"{self.url_base}{ruta}", timeout=timeout or self.timeout_http
        )

    def verificar_conexion(self) -> None:
        """Lanza ConnectionError con un mensaje claro si no hay camara."""
        try:
            respuesta = self._get("/gopro/camera/state", timeout=3.0)
        except requests.RequestException as error:
            raise ConnectionError(
                f"No hay respuesta de la GoPro en {self.url_base}. "
                "Verifica que el PC este conectado a la red WiFi de la camara "
                "(ejecuta conectar_gopro_ble.py si la red no aparece) y que "
                f"la camara este encendida. Detalle: {error}"
            ) from error

        if respuesta.status_code == 404:
            raise ConnectionError(
                "La GoPro respondio 404 a la API Open GoPro. Actualiza el "
                "firmware de la HERO10 (v01.16 o superior)."
            )

    def iniciar_stream_http(self) -> None:
        """Pide a la GoPro que empiece a enviar la previsualizacion por UDP."""
        # Un stream que quedo abierto de una ejecucion anterior produce
        # HTTP 409 al pedir start. Detenerlo primero deja la camara limpia.
        try:
            self._get("/gopro/camera/stream/stop", timeout=3.0)
            time.sleep(0.5)
        except requests.RequestException:
            pass

        ultimo = None
        for intento in range(3):
            try:
                respuesta = self._get("/gopro/camera/stream/start")
            except requests.RequestException as error:
                ultimo = f"No se pudo iniciar el stream: {error}"
                time.sleep(1.0)
                continue

            if respuesta.status_code == 200:
                self._ultimo_error = ""
                return

            ultimo = (
                f"La GoPro respondio HTTP {respuesta.status_code} a "
                f"stream/start: {respuesta.text.strip()[:200]}"
            )
            if respuesta.status_code == 409:
                # Ocupada: grabando, en un menu, o con el stream ya activo.
                time.sleep(1.5)
                continue
            break

        if ultimo and "409" in ultimo:
            # Puede que el stream ya este corriendo: se intenta recibirlo.
            print(f"Advertencia: {ultimo}. Se intentara recibir el UDP igual.")
            self._ultimo_error = ultimo
            return

        raise ConnectionError(ultimo or "No se pudo iniciar el stream")

    def _enviar_keep_alive(self) -> None:
        while not self._detener.wait(self.intervalo_keep_alive):
            try:
                self._get("/gopro/camera/keep_alive", timeout=2.0)
            except requests.RequestException as error:
                if not self._conectada:
                    self._ultimo_error = f"Fallo keep_alive: {error}"

    def _detener_stream_http(self) -> None:
        try:
            self._get("/gopro/camera/stream/stop", timeout=2.0)
        except requests.RequestException:
            pass

    # ------------------------------------------------------------------
    # UDP / OpenCV
    # ------------------------------------------------------------------
    def _abrir_captura(self) -> bool:
        if self._captura is not None:
            self._captura.release()
            self._captura = None

        parametros = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            parametros += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000]
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            parametros += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]

        captura = cv2.VideoCapture(self.url_stream, cv2.CAP_FFMPEG, parametros)
        if not captura.isOpened():
            captura.release()
            return False

        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._captura = captura
        return True

    def _leer_stream(self) -> None:
        frames_ventana = 0
        inicio_ventana = time.perf_counter()
        fallos_consecutivos = 0

        while not self._detener.is_set():
            if self._captura is None:
                if not self._abrir_captura():
                    self._conectada = False
                    self._ultimo_error = (
                        f"OpenCV no pudo abrir el stream UDP en el puerto "
                        f"{self.puerto_udp}. Revisa el firewall de Windows "
                        "(ver README) y que la camara este enviando video."
                    )
                    self._detener.wait(1.0)
                    continue

            exito, frame = self._captura.read()

            if not exito or frame is None or frame.size == 0:
                fallos_consecutivos += 1
                self._conectada = False
                if fallos_consecutivos >= 30:
                    self._ultimo_error = (
                        "No llegan frames por UDP. Se reabre el stream."
                    )
                    self._captura.release()
                    self._captura = None
                    fallos_consecutivos = 0
                    # La camara pudo cortar el stream; se vuelve a pedir.
                    try:
                        self.iniciar_stream_http()
                    except ConnectionError as error:
                        self._ultimo_error = str(error)
                    self._detener.wait(0.5)
                else:
                    time.sleep(0.01)
                continue

            fallos_consecutivos = 0
            ahora = time.perf_counter()

            with self._lock_frame:
                self._ultimo_frame = frame
                self._numero_frame += 1
                self._instante_ultimo_frame = ahora
                self._resolucion = (frame.shape[1], frame.shape[0])

            if not self._conectada:
                self._ultimo_error = ""
            self._conectada = True
            frames_ventana += 1

            lapso = ahora - inicio_ventana
            if lapso >= 1.0:
                self._fps_entrada = frames_ventana / lapso
                frames_ventana = 0
                inicio_ventana = ahora

        if self._captura is not None:
            self._captura.release()
            self._captura = None

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------
    def iniciar(self) -> None:
        """Verifica la camara, pide el stream y arranca los hilos."""
        if self.esta_ejecutandose():
            return

        self._detener.clear()
        self.verificar_conexion()
        self.iniciar_stream_http()

        self._hilo_keep_alive = threading.Thread(
            target=self._enviar_keep_alive, daemon=True, name="gopro-keep-alive"
        )
        self._hilo_keep_alive.start()

        self._hilo_captura = threading.Thread(
            target=self._leer_stream, daemon=True, name="gopro-captura-udp"
        )
        self._hilo_captura.start()

    def obtener_frame(self, copiar: bool = True):
        """Devuelve (exito, frame, numero_frame)."""
        with self._lock_frame:
            if self._ultimo_frame is None:
                return False, None, self._numero_frame
            frame = self._ultimo_frame.copy() if copiar else self._ultimo_frame
            return True, frame, self._numero_frame

    def esperar_primer_frame(self, timeout: float = 15.0) -> bool:
        limite = time.perf_counter() + timeout
        while time.perf_counter() < limite:
            exito, _, _ = self.obtener_frame(copiar=False)
            if exito:
                return True
            if self._detener.wait(0.05):
                return False

        self._ultimo_error = (
            f"No llego ningun frame en {timeout:.0f} s. La camara acepto el "
            f"pedido de stream pero el PC no recibe video en UDP "
            f"{self.puerto_udp}: casi siempre es el firewall de Windows "
            "(la red de la GoPro queda como 'Publica'). Ver README."
        )
        return False

    def estado(self) -> dict:
        antiguedad = None
        if self._instante_ultimo_frame > 0:
            antiguedad = time.perf_counter() - self._instante_ultimo_frame

        conectada = bool(
            self._conectada
            and antiguedad is not None
            and antiguedad <= self.timeout_sin_frames
        )

        return {
            "camara_conectada": conectada,
            "stream_ejecutandose": self.esta_ejecutandose(),
            "numero_frame": self._numero_frame,
            "fps_entrada": round(self._fps_entrada, 2),
            "resolucion": self._resolucion,
            "segundos_desde_ultimo_frame": (
                round(antiguedad, 2) if antiguedad is not None else None
            ),
            "url_http": self.url_base,
            "url_udp": self.url_stream,
            "ultimo_error": self._ultimo_error,
        }

    def esta_ejecutandose(self) -> bool:
        return bool(
            self._hilo_captura is not None
            and self._hilo_captura.is_alive()
            and not self._detener.is_set()
        )

    def detener(self) -> None:
        self._detener.set()

        for hilo in (self._hilo_captura, self._hilo_keep_alive):
            if hilo is not None and hilo.is_alive():
                hilo.join(timeout=6.0)

        if self._captura is not None:
            self._captura.release()
            self._captura = None

        self._detener_stream_http()
        self._conectada = False

    def __enter__(self):
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.detener()


def puerto_udp_libre(puerto: int) -> bool:
    """True si nadie mas esta escuchando en el puerto UDP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", puerto))
        return True
    except OSError:
        return False
    finally:
        s.close()


if __name__ == "__main__":
    camara = GoProCameraService()
    try:
        print("Iniciando stream de la GoPro...")
        camara.iniciar()
        if not camara.esperar_primer_frame(timeout=15.0):
            raise RuntimeError(camara.estado()["ultimo_error"])

        print("Stream recibido. Presiona q para cerrar.")
        ultimo_numero = -1
        ultimo_reporte = 0.0
        while True:
            exito, frame, numero = camara.obtener_frame()
            if exito and numero != ultimo_numero:
                ultimo_numero = numero
                cv2.imshow("GoPro HERO10", frame)
            ahora = time.perf_counter()
            if ahora - ultimo_reporte >= 2.0:
                print(camara.estado())
                ultimo_reporte = ahora
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camara.detener()
        cv2.destroyAllWindows()
