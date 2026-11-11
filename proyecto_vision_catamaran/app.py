import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.agent_service import AgentService
from services.pipeline_service import PipelineService


RAIZ = Path(__file__).resolve().parent
FRONTEND = RAIZ / "frontend"
VIDEO_PREDETERMINADO = os.getenv("VIDEO_PRUEBA", "video_prueba.mp4")

pipeline = None
agente = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, agente
    pipeline = PipelineService()
    agente = AgentService()

    yield

    if pipeline is not None:
        pipeline.detener()


app = FastAPI(
    title="Sistema de visión de pelotas",
    version="1.2.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class SolicitudVideo(BaseModel):
    video: str = VIDEO_PREDETERMINADO
    repetir: bool = True


class SolicitudAgente(BaseModel):
    pregunta: str


@app.get("/")
def inicio():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/estado")
def estado():
    return JSONResponse(pipeline.obtener_estado())


@app.post("/api/agente")
def preguntar_agente(solicitud: SolicitudAgente):
    try:
        return agente.responder(solicitud.pregunta, pipeline.obtener_estado())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/api/reiniciar")
def reiniciar():
    pipeline.reiniciar_conteos()
    return {"ok": True, "mensaje": "Conteos reiniciados"}


@app.post("/api/fuente/video")
def iniciar_video(solicitud: SolicitudVideo):
    try:
        pipeline.iniciar_video(solicitud.video, solicitud.repetir)
        return {"ok": True, "fuente": solicitud.video}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/fuente/gopro")
def iniciar_gopro():
    try:
        pipeline.iniciar_gopro()
        return {"ok": True, "fuente": "GoPro HERO10"}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/api/detener")
def detener():
    pipeline.detener()
    return {"ok": True, "mensaje": "Pipeline detenido"}


async def generar_mjpeg(request: Request):
    """Envia un JPEG solo cuando el pipeline publica uno nuevo."""
    ultimo_id = -1
    while True:
        if await request.is_disconnected():
            break
        id_jpeg, jpeg = pipeline.obtener_jpeg()
        if jpeg is None or id_jpeg == ultimo_id:
            await asyncio.sleep(0.015)
            continue
        ultimo_id = id_jpeg
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )


@app.get("/video")
async def video(request: Request):
    return StreamingResponse(
        generar_mjpeg(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )
