"""Diagnostico paso a paso de la conexion PC <-> GoPro HERO10 por WiFi.

Uso (con el PC ya conectado a la red WiFi de la GoPro y uvicorn APAGADO,
porque ambos necesitan el puerto UDP 8554):

    python scripts/integracion/diagnostico_gopro.py

Cada paso dice OK o FALLA y que hacer. Si los 4 pasos dan OK, la web
deberia mostrar la camara.
"""

import os
import socket
import sys
import time

import requests

IP = os.getenv("GOPRO_IP", "10.5.5.9")
URL = f"http://{IP}:8080"
PUERTO_UDP = int(os.getenv("GOPRO_PUERTO_UDP", "8554"))

sesion = requests.Session()
sesion.trust_env = False


def titulo(texto):
    print("\n" + "=" * 64 + f"\n{texto}\n" + "=" * 64)


def falla(mensaje):
    print(f"[FALLA] {mensaje}")
    try:
        sesion.get(f"{URL}/gopro/camera/stream/stop", timeout=2)
    except requests.RequestException:
        pass
    sys.exit(1)


def paso_1_http():
    titulo("1) La GoPro responde por HTTP")
    try:
        r = sesion.get(f"{URL}/gopro/camera/state", timeout=3)
    except requests.RequestException as error:
        falla(
            f"No hay respuesta en {URL}.\n"
            "  - Conecta el PC a la red WiFi de la GoPro (GP........).\n"
            "  - Si la red no aparece, ejecuta conectar_gopro_ble.py.\n"
            "  - Desactiva VPNs.\n"
            f"  Detalle: {error}"
        )
    if r.status_code == 404:
        falla("API Open GoPro no disponible: actualiza el firmware (v01.16+).")
    print(f"[OK] HTTP {r.status_code}")

    try:
        ip_local = socket.gethostbyname(socket.gethostname())
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((IP, 8080))
        ip_local = s.getsockname()[0]
        s.close()
        print(f"     IP del PC en la red de la GoPro: {ip_local}")
    except OSError:
        pass


def paso_2_stream():
    titulo("2) La GoPro acepta iniciar la previsualizacion")
    try:
        sesion.get(f"{URL}/gopro/camera/stream/stop", timeout=3)
        time.sleep(0.5)
        r = sesion.get(f"{URL}/gopro/camera/stream/start", timeout=5)
    except requests.RequestException as error:
        falla(f"Error HTTP: {error}")
    if r.status_code == 409:
        falla(
            "HTTP 409: la camara esta ocupada. Deten la grabacion, sal de "
            "menus en la pantalla y deja la camara en modo Video."
        )
    if r.status_code != 200:
        falla(f"HTTP {r.status_code}: {r.text[:200]}")
    print("[OK] stream/start aceptado")


def paso_3_udp():
    titulo(f"3) Llegan paquetes UDP al puerto {PUERTO_UDP} del PC")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", PUERTO_UDP))
    except OSError:
        falla(
            f"El puerto {PUERTO_UDP} esta ocupado. Cierra uvicorn, VLC u otro "
            "programa que este leyendo la GoPro."
        )
    s.settimeout(1.0)
    paquetes = 0
    total = 0
    limite = time.time() + 5
    while time.time() < limite:
        try:
            datos, _ = s.recvfrom(65535)
            paquetes += 1
            total += len(datos)
        except socket.timeout:
            pass
    s.close()

    if paquetes == 0:
        falla(
            "La camara dice que transmite, pero NO llega nada al PC.\n"
            "  Casi siempre es el Firewall de Windows: la red de la GoPro se\n"
            "  marca como 'Publica'. En PowerShell como administrador:\n\n"
            '  New-NetFirewallRule -DisplayName "GoPro UDP 8554" '
            "-Direction Inbound -Protocol UDP -LocalPort 8554 "
            "-Action Allow -Profile Any\n\n"
            "  y permite python.exe en redes publicas cuando Windows lo pregunte."
        )
    print(f"[OK] {paquetes} paquetes, {total / 5 / 1024:.0f} KB/s")


def paso_4_opencv():
    titulo("4) OpenCV decodifica el video")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from services.camera_service import GoProCameraService  # noqa: E402

    camara = GoProCameraService()
    try:
        camara.iniciar()
        if not camara.esperar_primer_frame(timeout=15):
            falla(camara.estado()["ultimo_error"])
        time.sleep(3)
        estado = camara.estado()
        print(
            f"[OK] {estado['resolucion']} a {estado['fps_entrada']} FPS "
            f"({estado['numero_frame']} frames)"
        )
        import cv2

        _, frame, _ = camara.obtener_frame()
        cv2.imwrite("captura_gopro.jpg", frame)
        print("     Frame guardado en captura_gopro.jpg")
    finally:
        camara.detener()


if __name__ == "__main__":
    paso_1_http()
    paso_2_stream()
    paso_3_udp()
    paso_4_opencv()
    titulo("Todo OK: inicia uvicorn y pulsa 'Usar GoPro HERO10' en la web")
