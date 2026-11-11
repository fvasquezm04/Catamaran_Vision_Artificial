import json
import os
import re
import unicodedata

from dotenv import load_dotenv

load_dotenv()

COLORES = ("amarillo", "azul", "naranja", "rojo", "verde")


class AgentService:
    """Responde desde el estado real y usa OpenAI solo cuando aporta valor."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.modelo = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.cliente = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.cliente = OpenAI(api_key=self.api_key)
            except Exception as error:
                print(f"No se pudo iniciar OpenAI: {error}")

    @staticmethod
    def _normalizar(texto):
        texto = unicodedata.normalize("NFD", texto.lower())
        return "".join(c for c in texto if unicodedata.category(c) != "Mn")

    @staticmethod
    def _resumen(estado):
        visibles = estado.get("pelotas_visibles", {})
        unicas = estado.get("pelotas_unicas", {})
        return {
            "pelotas_visibles": {c: int(visibles.get(c, 0)) for c in COLORES},
            "pelotas_visibles_total": int(estado.get("pelotas_visibles_total", 0)),
            "pelotas_unicas": {c: int(unicas.get(c, 0)) for c in COLORES},
            "pelotas_unicas_total": int(estado.get("pelotas_unicas_total", 0)),
            "fps_procesamiento": float(estado.get("fps_procesamiento", 0)),
            "latencia_ms": float(estado.get("latencia_ms", 0)),
            "pipeline_ejecutando": bool(estado.get("pipeline_ejecutando", False)),
            "tipo_fuente": estado.get("tipo_fuente", "desconocida"),
            "fuente": estado.get("fuente", ""),
            "error": estado.get("pipeline_error") or estado.get("ultimo_error") or "",
        }

    def _respuesta_local(self, pregunta, estado):
        """Resuelve consultas objetivas sin gastar API ni arriesgar cifras."""
        texto = self._normalizar(pregunta)
        datos = self._resumen(estado)
        visibles = datos["pelotas_visibles"]
        unicas = datos["pelotas_unicas"]

        if any(p in texto for p in ("ayuda", "que puedo preguntar", "ejemplos")):
            return (
                "Puedes preguntar por pelotas visibles, pelotas únicas detectadas, "
                "el color más frecuente, el total, los FPS, la latencia o el estado del sistema."
            )

        if "fps" in texto or "fotogramas" in texto:
            return f"El sistema está procesando aproximadamente {datos['fps_procesamiento']:.1f} FPS."

        if "latencia" in texto or "demora" in texto:
            return f"La latencia de procesamiento actual es de {datos['latencia_ms']:.1f} ms por frame."

        if any(p in texto for p in ("funcionando", "activo", "estado", "conectado")):
            estado_texto = "activo" if datos["pipeline_ejecutando"] else "detenido"
            respuesta = f"El pipeline está {estado_texto} y la fuente actual es {datos['tipo_fuente']}."
            if datos["error"]:
                respuesta += f" El último error registrado es: {datos['error']}"
            return respuesta

        quiere_visibles = any(
            p in texto for p in ("visible", "visibles", "se ven", "hay ahora", "actualmente")
        )
        quiere_unicas = any(
            p in texto for p in ("unica", "unicas", "detectado", "detectadas", "sesion", "acumulado")
        )

        color_mencionado = next((c for c in COLORES if c in texto), None)
        if color_mencionado:
            if quiere_visibles and not quiere_unicas:
                cantidad = visibles[color_mencionado]
                return f"Actualmente se ven {cantidad} pelotas de color {color_mencionado}."
            cantidad = unicas[color_mencionado]
            return (
                f"Durante la sesión se han contabilizado {cantidad} trayectorias únicas "
                f"de color {color_mencionado}."
            )

        if any(p in texto for p in ("mas frecuente", "mayor cantidad", "mas hay", "predomina")):
            fuente = visibles if quiere_visibles else unicas
            maximo = max(fuente.values(), default=0)
            ganadores = [c for c, n in fuente.items() if n == maximo]
            tipo = "visibles" if quiere_visibles else "únicas"
            if maximo == 0:
                return f"No hay pelotas {tipo} registradas en este momento."
            return f"El mayor conteo de pelotas {tipo} es {maximo}, correspondiente a: {', '.join(ganadores)}."

        if any(p in texto for p in ("cuantas pelotas", "total", "cuantos objetos")):
            if quiere_visibles and not quiere_unicas:
                return f"Actualmente hay {datos['pelotas_visibles_total']} pelotas visibles."
            return (
                f"Durante la sesión se han contabilizado {datos['pelotas_unicas_total']} "
                "trayectorias únicas de pelotas."
            )

        if any(p in texto for p in ("resumen", "reporte", "conteo")):
            detalle_visible = ", ".join(f"{c}: {visibles[c]}" for c in COLORES)
            detalle_unicas = ", ".join(f"{c}: {unicas[c]}" for c in COLORES)
            return (
                f"Visibles ahora: {datos['pelotas_visibles_total']} ({detalle_visible}). "
                f"Trayectorias únicas: {datos['pelotas_unicas_total']} ({detalle_unicas}). "
                f"Rendimiento: {datos['fps_procesamiento']:.1f} FPS y {datos['latencia_ms']:.1f} ms."
            )

        return None

    def _respuesta_openai(self, pregunta, estado):
        if self.cliente is None:
            return None

        datos = self._resumen(estado)
        instrucciones = (
            "Eres el asistente de un sistema de visión de un submarino. "
            "Responde en español, de forma breve y técnica. Usa únicamente los datos JSON proporcionados. "
            "No inventes conteos ni afirmes que una trayectoria única equivale necesariamente a una pelota física: "
            "ByteTrack puede reasignar IDs si un objeto sale y vuelve a entrar. Distingue siempre entre visibles ahora "
            "y trayectorias únicas acumuladas."
        )

        respuesta = self.cliente.responses.create(
            model=self.modelo,
            instructions=instrucciones,
            input=(
                f"Estado actual del sistema:\n{json.dumps(datos, ensure_ascii=False)}\n\n"
                f"Pregunta del usuario: {pregunta}"
            ),
        )
        return respuesta.output_text.strip()

    def responder(self, pregunta, estado):
        pregunta = pregunta.strip()
        if not pregunta:
            raise ValueError("La pregunta está vacía")

        respuesta = self._respuesta_local(pregunta, estado)
        if respuesta is not None:
            return {"respuesta": respuesta, "origen": "local"}

        try:
            respuesta = self._respuesta_openai(pregunta, estado)
            if respuesta:
                return {"respuesta": respuesta, "origen": "openai"}
        except Exception as error:
            return {
                "respuesta": (
                    "No se pudo consultar el modelo de IA. Puedes preguntar por conteos, "
                    "pelotas visibles, FPS, latencia o estado del sistema."
                ),
                "origen": "local_fallback",
                "error": str(error),
            }

        return {
            "respuesta": (
                "No reconocí esa consulta. Pregunta, por ejemplo: “¿Cuántas pelotas rojas se ven?”, "
                "“¿Cuál color es más frecuente?” o “Dame un resumen”."
            ),
            "origen": "local",
        }
