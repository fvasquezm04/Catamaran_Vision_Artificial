const colores = ["amarillo", "azul", "naranja", "rojo", "verde"];
const tarjetas = document.getElementById("tarjetas");
for (const color of colores) {
  tarjetas.insertAdjacentHTML("beforeend", `<article class="color-card ${color}"><div class="dot"></div><div><h3>${color[0].toUpperCase() + color.slice(1)}</h3><p>Visibles: <strong id="visible-${color}">0</strong></p><p>Únicas: <strong id="unica-${color}">0</strong></p></div></article>`);
}

async function actualizarEstado() {
  try {
    const respuesta = await fetch("/api/estado");
    const estado = await respuesta.json();
    document.getElementById("fps").textContent = estado.fps_procesamiento ?? 0;
    document.getElementById("latencia").textContent = `${estado.latencia_ms ?? 0} ms`;
    document.getElementById("total-visibles").textContent = estado.pelotas_visibles_total ?? 0;
    document.getElementById("total-unicas").textContent = estado.pelotas_unicas_total ?? 0;
    document.getElementById("fuente").textContent = estado.fuente || "Sin fuente";

    const badge = document.getElementById("estado-pipeline");
    badge.textContent = estado.pipeline_ejecutando ? "Procesando" : "Detenido";
    badge.classList.toggle("active", Boolean(estado.pipeline_ejecutando));

    const detalle = document.getElementById("detalle-modelo");
    detalle.textContent = `YOLO en ${estado.dispositivo} · imgsz ${estado.tamano_yolo} · frames procesados ${estado.numero_frame ?? 0}`;

    const camara = estado.camara;
    const estadoCamara = document.getElementById("estado-camara");
    if (camara) {
      estadoCamara.textContent = camara.camara_conectada
        ? `GoPro conectada | ${camara.fps_entrada} FPS de entrada${camara.resolucion ? ` | ${camara.resolucion[0]}x${camara.resolucion[1]}` : ""}`
        : `GoPro sin frames | ${camara.ultimo_error || "revisar conexión"}`;
      estadoCamara.classList.toggle("connected", Boolean(camara.camara_conectada));
    } else {
      estadoCamara.textContent = "GoPro sin iniciar";
      estadoCamara.classList.remove("connected");
    }

    for (const color of colores) {
      document.getElementById(`visible-${color}`).textContent = estado.pelotas_visibles?.[color] ?? 0;
      document.getElementById(`unica-${color}`).textContent = estado.pelotas_unicas?.[color] ?? 0;
    }
    if (estado.pipeline_error) mostrarMensaje(estado.pipeline_error, true);
    else if (document.getElementById("mensaje").classList.contains("error") && estado.pipeline_ejecutando) mostrarMensaje("Procesando");
  } catch (error) { mostrarMensaje(`No se pudo consultar el estado: ${error}`, true); }
}

function mostrarMensaje(texto, error = false) {
  const elemento = document.getElementById("mensaje");
  elemento.textContent = texto;
  elemento.classList.toggle("error", error);
}

async function post(url, cuerpo = null) {
  const opciones = { method: "POST", headers: { "Content-Type": "application/json" } };
  if (cuerpo) opciones.body = JSON.stringify(cuerpo);
  const respuesta = await fetch(url, opciones);
  const datos = await respuesta.json();
  if (!respuesta.ok) throw new Error(datos.detail || "La operación falló");
  return datos;
}

const imagenVideo = document.getElementById("video");

function reconectarVideo() {
  imagenVideo.src = `/video?t=${Date.now()}`;
}
// Si el stream MJPEG se corta (servidor reiniciado, red), reintenta solo.
imagenVideo.addEventListener("error", () => setTimeout(reconectarVideo, 1500));

async function iniciarFuente(boton, url, cuerpo, textoEspera, textoOk) {
  const botones = document.querySelectorAll("#iniciar-gopro, #iniciar-video");
  botones.forEach((b) => (b.disabled = true));
  try {
    mostrarMensaje(textoEspera);
    await post(url, cuerpo);
    reconectarVideo();
    mostrarMensaje(textoOk);
  } catch (error) {
    mostrarMensaje(error.message, true);
  } finally {
    botones.forEach((b) => (b.disabled = false));
  }
}

document.getElementById("iniciar-gopro").addEventListener("click", (e) =>
  iniciarFuente(
    e.target,
    "/api/fuente/gopro",
    null,
    "Conectando con la GoPro (10.5.5.9). Puede tardar hasta 15 segundos...",
    "GoPro conectada: YOLO + clasificador de color procesando"
  )
);

document.getElementById("iniciar-video").addEventListener("click", (e) =>
  iniciarFuente(
    e.target,
    "/api/fuente/video",
    { repetir: true },
    "Abriendo video de prueba...",
    "Video de prueba en procesamiento"
  )
);

document.getElementById("reiniciar").addEventListener("click", async () => {
  try { await post("/api/reiniciar"); mostrarMensaje("Conteos reiniciados"); }
  catch (error) { mostrarMensaje(error.message, true); }
});

document.getElementById("detener").addEventListener("click", async () => {
  try { await post("/api/detener"); mostrarMensaje("Pipeline detenido"); }
  catch (error) { mostrarMensaje(error.message, true); }
});

function agregarBurbuja(texto, clase) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = `bubble ${clase}`;
  div.textContent = texto;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

document.getElementById("chat-form").addEventListener("submit", async (evento) => {
  evento.preventDefault();
  const entrada = document.getElementById("pregunta");
  const pregunta = entrada.value.trim();
  if (!pregunta) return;
  agregarBurbuja(pregunta, "user");
  entrada.value = "";
  try {
    const datos = await post("/api/agente", { pregunta });
    agregarBurbuja(datos.respuesta, "assistant");
  } catch (error) { agregarBurbuja(error.message, "error-bubble"); }
});

actualizarEstado();
setInterval(actualizarEstado, 700);
