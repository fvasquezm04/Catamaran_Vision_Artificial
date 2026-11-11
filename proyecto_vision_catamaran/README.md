# Proyecto de visión artificial – Catamarán clasificador de pelotas

Sistema de visión artificial para un catamarán que **detecta, rastrea, clasifica por color y cuenta** pelotas flotando en el agua. El video llega desde una **GoPro HERO10** (o desde un video de prueba), se procesa en tiempo real y se muestra en una interfaz web con conteos por color y un agente de IA al que se le puede preguntar por el estado del sistema.

Colores soportados: **amarillo, azul, naranja, rojo y verde**.

---

## ¿Cómo funciona?

```
GoPro HERO10 (UDP 8554)  ─┐
                          ├─►  Frame ─► YOLO11n ─► ByteTrack ─► Clasificador de color ─► Votación por track ─► Video anotado + conteos
Video de prueba (.mp4)   ─┘     │        (¿dónde?)    (¿cuál?)       (¿de qué color?)       (estabiliza el color)       │
                                                                                                                       ▼
                                                                                      Interfaz web (FastAPI) + Agente de IA
```

1. **Captura del video** (`services/camera_service.py`, `services/pipeline_service.py`)
   Se lee el stream de la GoPro por WiFi (UDP, puerto 8554) o un video local de `videos_prueba/`. Un hilo dedicado lee los frames y otro ejecuta la visión, para que la interfaz no se congele si la inferencia tarda.

2. **Detección con YOLO** (`modelos/yolo/best.pt`)
   Un modelo **YOLO11n** entrenado con una sola clase (`P` = pelota) encuentra la posición de cada pelota en el frame. YOLO **no decide el color**, solo dónde hay una pelota. Se entrenó con `imgsz=640`, por lo que se usa el mismo tamaño en inferencia.

3. **Seguimiento con ByteTrack** (`services/vision_service.py`)
   A cada pelota detectada se le asigna un **ID de track** que se mantiene entre frames. Esto permite contar pelotas **únicas** (no la misma pelota 100 veces) y estabilizar su color.

4. **Clasificación de color** (`modelos/color/clasificador_color.joblib`, `scripts/integracion/clasificador_color.py`)
   Para cada caja de YOLO se recorta la pelota y se clasifica su color con un modelo de machine learning clásico:
   - Se normaliza el recorte a 128×128 y se aplica una **máscara elíptica** para quedarse solo con la pelota.
   - Se descartan píxeles muy oscuros, poco saturados o con reflejo (brillo del sol sobre el agua).
   - Se extraen **histogramas de color en HSV y CIELAB** (124 características).
   - Un **Random Forest** devuelve el color y su confianza.

5. **Votación temporal por track**
   Un solo frame puede engañar al clasificador (reflejos, sombras). Por eso se guardan las últimas predicciones de cada track y el color final se decide por **votación**: se necesitan al menos 5 predicciones y que el color ganador tenga ≥ 60 % de los votos. Mientras no se cumpla, la pelota se muestra como *desconocido*.

6. **Interfaz web y agente de IA** (`app.py`, `frontend/`, `services/agent_service.py`)
   - FastAPI sirve la página y el video procesado como stream MJPEG (`/video`).
   - Muestra FPS, latencia, pelotas **visibles** ahora y pelotas **únicas** acumuladas por color.
   - El **agente** responde preguntas ("¿cuántas pelotas rojas se ven?") a partir del estado real del sistema. Usa OpenAI solo cuando hace falta; sin API key funciona con respuestas locales.

---

## Estructura del proyecto

```
proyecto_vision_catamaran/
├── app.py                      # Servidor FastAPI (endpoints + stream de video)
├── requirements.txt            # Dependencias
├── .env.example                # Plantilla de configuración (copiar a .env)
├── frontend/                   # Interfaz web (HTML + JS + CSS)
├── services/
│   ├── pipeline_service.py     # Orquesta fuente de video -> visión -> JPEG
│   ├── vision_service.py       # YOLO + ByteTrack + color + votación + dibujo
│   ├── camera_service.py       # Conexión y stream de la GoPro
│   └── agent_service.py        # Agente de IA (local u OpenAI)
├── modelos/
│   ├── yolo/best.pt            # Detector de pelotas
│   └── color/clasificador_color.joblib   # Clasificador de color
├── scripts/
│   ├── integracion/            # Clasificador, prueba con video, utilidades GoPro
│   ├── entrenamiento_color/    # 01 extraer recortes, 02 entrenar clasificador
│   └── yolo/entrenar_pelotas.py          # Entrenamiento del detector
├── configuracion/              # Parámetros del clasificador y data.yaml de YOLO
├── datos_color/                # Metadatos del dataset de recortes (CSV)
├── resultados_color/           # Métricas, reporte y matriz de confusión
└── videos_prueba/video_prueba.mp4
```

---

## Instalación (Python 3.12)

Hazlo **con internet**, antes de conectarte al WiFi de la GoPro:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Edita `.env` si quieres usar el agente con OpenAI (`OPENAI_API_KEY`). **Nunca subas `.env` al repositorio** (ya está en `.gitignore`).

> `lap` (lo usa ByteTrack) está en `requirements.txt`. Si falta y no hay internet (red de la GoPro), el tracking falla.

## Uso

### Con un video de prueba (sin cámara)

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Abre <http://localhost:8000> y pulsa **Probar con video de prueba**.

### Con la GoPro HERO10

1. Enciende la GoPro (firmware v01.16 o superior, modo Video, sin grabar) y activa su WiFi. Si la red no aparece: `python scripts/integracion/conectar_gopro_ble.py`.
2. Conecta el PC a la red WiFi de la GoPro.
3. Permite el video UDP en el firewall de Windows (una sola vez, PowerShell **como administrador**):
   ```powershell
   New-NetFirewallRule -DisplayName "GoPro UDP 8554" -Direction Inbound -Protocol UDP -LocalPort 8554 -Action Allow -Profile Any
   ```
4. Inicia el servidor (una sola instancia, ocupa el puerto UDP 8554):
   ```powershell
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
5. Abre <http://localhost:8000> y pulsa **Usar GoPro HERO10**.

Diagnóstico paso a paso si algo falla: `python scripts/integracion/diagnostico_gopro.py`.

### Procesar un video sin interfaz

```powershell
python scripts/integracion/probar_video.py
```

Genera un video anotado y un resumen JSON en `resultados/videos/`.

---

## Configuración (`.env`)

| Variable | Por defecto | Uso |
|---|---|---|
| `OPENAI_API_KEY` | *(vacío)* | Opcional. Activa las respuestas del agente con OpenAI |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Modelo usado por el agente |
| `VIDEO_PRUEBA` | `video_prueba.mp4` | Video de `videos_prueba/` para la prueba |
| `GOPRO_IP` | `10.5.5.9` | IP de la cámara en su propio WiFi |
| `GOPRO_PUERTO_UDP` | `8554` | Puerto donde llega el video |
| `YOLO_IMGSZ` | `640` | Igual al entrenamiento; bajar a 480/320 solo si el PC es lento (se pierden pelotas pequeñas) |
| `YOLO_CONFIANZA` | `0.50` | Confianza mínima de YOLO |
| `COLOR_CONFIANZA` | `0.60` | Confianza mínima del color |

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Interfaz web |
| GET | `/video` | Stream MJPEG del video procesado |
| GET | `/api/estado` | Estado actual (conteos, FPS, latencia, fuente) |
| POST | `/api/fuente/gopro` | Inicia el procesamiento desde la GoPro |
| POST | `/api/fuente/video` | Inicia el procesamiento desde un video local |
| POST | `/api/agente` | Pregunta al agente de IA |
| POST | `/api/reiniciar` | Reinicia los conteos |
| POST | `/api/detener` | Detiene el pipeline |

---

## Resultados

**Clasificador de color** (validación cruzada estratificada por grupos temporales, 652 recortes):

| Modelo | Accuracy | Macro F1 | Tiempo medio |
|---|---|---|---|
| **Random Forest** (elegido) | 1.000 | 1.000 | 3.85 ms |
| KNN | 1.000 | 1.000 | 0.16 ms |
| Regresión logística | 0.974 | 0.976 | 0.07 ms |
| SVM RBF | 0.960 | 0.961 | 0.15 ms |

Matriz de confusión y reporte completo en `resultados_color/`.

> Las métricas perfectas se obtienen sobre el dataset propio del proyecto (fondo y condiciones de luz similares). Con otra luz, otro lago o pelotas nuevas el desempeño puede bajar; conviene validar con video real de la prueba en agua.

## Reentrenamiento

- **Detector YOLO:** `python scripts/yolo/entrenar_pelotas.py`. Requiere el dataset YOLO (no incluido en el repositorio); ajusta `path` en `configuracion/data.yaml`.
- **Clasificador de color:** `scripts/entrenamiento_color/01_extraer_recortes.py` recorta las pelotas usando las anotaciones, y `02_entrenar_clasificador.py` compara modelos (Random Forest, KNN, regresión logística, SVM) y guarda el mejor. Estos scripts esperan la estructura original de carpetas del proyecto (`pelotas_juntas/` con los frames y JSON de labelme), que no está en el repositorio por su tamaño.

## Limitaciones conocidas

- El conteo de pelotas únicas depende de los IDs de ByteTrack: si una pelota sale de cuadro mucho tiempo y vuelve, puede contarse de nuevo.
- Con CPU el procesamiento ronda los 2–3 FPS; con GPU NVIDIA (CUDA) se usa automáticamente si PyTorch tiene CUDA.
- La GoPro solo transmite si el PC está en su red WiFi, por lo que mientras tanto no hay internet (el agente con OpenAI no funcionará; responde en modo local).

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| "No hay respuesta de la GoPro en http://10.5.5.9:8080" | El PC no está en el WiFi de la cámara | Conectarse a la red `GP...` o usar el script BLE |
| "No llegó ningún frame en 15 s" | El firewall bloquea UDP 8554 | Crear la regla de firewall indicada arriba |
| "El puerto UDP 8554 ya está en uso" | Otra instancia de uvicorn, VLC o un script | Cerrarlos; no usar `--workers` > 1 |
| HTTP 409 | La cámara está grabando o en un menú | Detener grabación y salir de menús |
| Se ve video pero sin cajas | Error en la visión | Revisar la consola de uvicorn (imprime la traza completa) |
