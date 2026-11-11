from pathlib import Path
import csv
import json
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_ENTRADA = RAIZ / "pelotas_juntas"
MAPA_CSV = CARPETA_ENTRADA / "mapa_renombrado.csv"

CARPETA_SALIDA = RAIZ / "clasificador_color" / "datos"
CARPETA_RECORTES = CARPETA_SALIDA / "recortes"
CARPETA_RESULTADOS = RAIZ / "clasificador_color" / "resultados"

CSV_DATASET = CARPETA_SALIDA / "dataset_recortes.csv"
CSV_AUDITORIA = CARPETA_RESULTADOS / "auditoria.csv"

COLORES_VALIDOS = {
    "amarillo",
    "azul",
    "naranja",
    "rojo",
    "verde",
}

# Porcentaje de margen adicional alrededor de la caja.
# Un margen pequeño ayuda si la anotación corta el borde de la pelota.
MARGEN_CAJA = 0.03

# Tamaño mínimo aceptado para un recorte.
ANCHO_MINIMO = 15
ALTO_MINIMO = 15

# Tamaño de las miniaturas de los mosaicos.
TAM_MINIATURA = 128
COLUMNAS_MOSAICO = 8


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def leer_csv_mapa(ruta_csv):
    """Lee el mapa conservando los nombres como texto."""
    return pd.read_csv(ruta_csv, dtype=str, encoding="utf-8-sig")


def normalizar_texto(texto):
    return str(texto).strip().lower()


def extraer_numero_original(nombre_original):
    """
    Extrae el último grupo numérico del nombre original.

    Ejemplos:
    frame_amarillo_000030.jpg -> 30
    pelota1_frame_001760.jpg  -> 1760
    """
    numeros = re.findall(r"\d+", Path(nombre_original).stem)

    if not numeros:
        return -1

    return int(numeros[-1])


def extraer_secuencia(nombre_original):
    """
    Obtiene un identificador de secuencia retirando el número final.

    Ejemplos:
    frame_amarillo_000030.jpg -> frame_amarillo
    pelota1_frame_001760.jpg  -> pelota1_frame
    pelota2_frame_000020.jpg  -> pelota2_frame
    """
    stem = Path(nombre_original).stem
    secuencia = re.sub(r"_?\d+$", "", stem)

    if not secuencia:
        secuencia = "secuencia_desconocida"

    return secuencia.lower()


def construir_grupo_temporal(carpeta_origen, secuencia, posicion, tam_bloque=25):
    """
    Crea bloques consecutivos de aproximadamente 25 recortes.

    Este grupo se usará posteriormente para evitar que recortes
    consecutivos queden mezclados aleatoriamente.
    """
    bloque = posicion // tam_bloque

    carpeta = normalizar_texto(carpeta_origen)
    secuencia = normalizar_texto(secuencia)

    return f"{carpeta}__{secuencia}__bloque_{bloque:02d}"


def leer_rectangulo_labelme(ruta_json):
    """
    Lee la primera anotación rectangular válida de un archivo LabelMe.
    """
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    shapes = datos.get("shapes", [])

    if not shapes:
        raise ValueError("El JSON no contiene shapes")

    rectangulos = [
        shape
        for shape in shapes
        if shape.get("shape_type") == "rectangle"
        and len(shape.get("points", [])) >= 2
    ]

    if not rectangulos:
        raise ValueError("El JSON no contiene rectángulos válidos")

    shape = rectangulos[0]
    puntos = shape["points"]

    x1, y1 = puntos[0]
    x2, y2 = puntos[1]

    x_min = min(float(x1), float(x2))
    y_min = min(float(y1), float(y2))
    x_max = max(float(x1), float(x2))
    y_max = max(float(y1), float(y2))

    return x_min, y_min, x_max, y_max, len(shapes)


def ajustar_caja(x1, y1, x2, y2, ancho_imagen, alto_imagen):
    """
    Añade un margen pequeño y limita la caja al tamaño de la imagen.
    """
    ancho_caja = x2 - x1
    alto_caja = y2 - y1

    margen_x = ancho_caja * MARGEN_CAJA
    margen_y = alto_caja * MARGEN_CAJA

    x1 = int(np.floor(x1 - margen_x))
    y1 = int(np.floor(y1 - margen_y))
    x2 = int(np.ceil(x2 + margen_x))
    y2 = int(np.ceil(y2 + margen_y))

    x1 = max(0, min(x1, ancho_imagen - 1))
    y1 = max(0, min(y1, alto_imagen - 1))
    x2 = max(x1 + 1, min(x2, ancho_imagen))
    y2 = max(y1 + 1, min(y2, alto_imagen))

    return x1, y1, x2, y2


def crear_mosaico(rutas_recortes, color, ruta_salida):
    """
    Crea una cuadrícula con todos los recortes de una clase.
    """
    miniaturas = []

    for ruta in rutas_recortes:
        imagen = cv2.imread(str(ruta))

        if imagen is None:
            continue

        alto, ancho = imagen.shape[:2]

        escala = min(
            TAM_MINIATURA / max(ancho, 1),
            TAM_MINIATURA / max(alto, 1),
        )

        nuevo_ancho = max(1, int(ancho * escala))
        nuevo_alto = max(1, int(alto * escala))

        mini = cv2.resize(
            imagen,
            (nuevo_ancho, nuevo_alto),
            interpolation=cv2.INTER_AREA,
        )

        lienzo = np.full(
            (TAM_MINIATURA, TAM_MINIATURA, 3),
            235,
            dtype=np.uint8,
        )

        inicio_x = (TAM_MINIATURA - nuevo_ancho) // 2
        inicio_y = (TAM_MINIATURA - nuevo_alto) // 2

        lienzo[
            inicio_y:inicio_y + nuevo_alto,
            inicio_x:inicio_x + nuevo_ancho,
        ] = mini

        miniaturas.append(lienzo)

    if not miniaturas:
        return

    filas = int(np.ceil(len(miniaturas) / COLUMNAS_MOSAICO))

    mosaico = np.full(
        (
            filas * TAM_MINIATURA,
            COLUMNAS_MOSAICO * TAM_MINIATURA,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for indice, miniatura in enumerate(miniaturas):
        fila = indice // COLUMNAS_MOSAICO
        columna = indice % COLUMNAS_MOSAICO

        y1 = fila * TAM_MINIATURA
        x1 = columna * TAM_MINIATURA

        mosaico[
            y1:y1 + TAM_MINIATURA,
            x1:x1 + TAM_MINIATURA,
        ] = miniatura

    cv2.imwrite(str(ruta_salida), mosaico)

    print(
        f"Mosaico {color}: "
        f"{len(miniaturas)} recortes -> {ruta_salida.name}"
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    CARPETA_RECORTES.mkdir(parents=True, exist_ok=True)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    for color in sorted(COLORES_VALIDOS):
        (CARPETA_RECORTES / color).mkdir(parents=True, exist_ok=True)

    if not MAPA_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{MAPA_CSV}"
        )

    mapa = leer_csv_mapa(MAPA_CSV)

    columnas_requeridas = {
        "carpeta_origen",
        "nombre_original",
        "nombre_nuevo",
        "color",
    }

    columnas_faltantes = columnas_requeridas - set(mapa.columns)

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas en mapa_renombrado.csv: "
            f"{sorted(columnas_faltantes)}"
        )

    mapa["color"] = mapa["color"].apply(normalizar_texto)
    mapa["numero_original"] = mapa["nombre_original"].apply(
        extraer_numero_original
    )
    mapa["secuencia"] = mapa["nombre_original"].apply(
        extraer_secuencia
    )

    mapa = mapa.sort_values(
        by=[
            "carpeta_origen",
            "secuencia",
            "numero_original",
        ]
    ).reset_index(drop=True)

    # Posición consecutiva dentro de carpeta y secuencia.
    mapa["posicion_secuencia"] = (
        mapa.groupby(["carpeta_origen", "secuencia"])
        .cumcount()
    )

    dataset = []
    auditoria = []

    for _, fila in mapa.iterrows():
        nombre_imagen = str(fila["nombre_nuevo"]).strip()
        nombre_original = str(fila["nombre_original"]).strip()
        carpeta_origen = str(fila["carpeta_origen"]).strip()
        color = normalizar_texto(fila["color"])
        numero_original = int(fila["numero_original"])
        secuencia = str(fila["secuencia"])
        posicion_secuencia = int(fila["posicion_secuencia"])

        ruta_imagen = CARPETA_ENTRADA / nombre_imagen
        ruta_json = CARPETA_ENTRADA / (
            Path(nombre_imagen).stem + ".json"
        )

        estado = "correcto"
        detalle = ""

        if color not in COLORES_VALIDOS:
            estado = "error_color"
            detalle = f"Color no válido: {color}"

        elif not ruta_imagen.exists():
            estado = "falta_imagen"
            detalle = str(ruta_imagen)

        elif not ruta_json.exists():
            estado = "falta_json"
            detalle = str(ruta_json)

        if estado != "correcto":
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": estado,
                    "detalle": detalle,
                }
            )
            continue

        imagen = cv2.imread(str(ruta_imagen))

        if imagen is None:
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": "imagen_ilegible",
                    "detalle": str(ruta_imagen),
                }
            )
            continue

        alto_imagen, ancho_imagen = imagen.shape[:2]

        try:
            x1, y1, x2, y2, cantidad_shapes = (
                leer_rectangulo_labelme(ruta_json)
            )
        except Exception as error:
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": "json_invalido",
                    "detalle": str(error),
                }
            )
            continue

        x1, y1, x2, y2 = ajustar_caja(
            x1,
            y1,
            x2,
            y2,
            ancho_imagen,
            alto_imagen,
        )

        ancho_recorte = x2 - x1
        alto_recorte = y2 - y1

        if ancho_recorte < ANCHO_MINIMO or alto_recorte < ALTO_MINIMO:
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": "recorte_muy_pequeno",
                    "detalle": (
                        f"{ancho_recorte}x{alto_recorte}"
                    ),
                }
            )
            continue

        recorte = imagen[y1:y2, x1:x2]

        if recorte.size == 0:
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": "recorte_vacio",
                    "detalle": f"Caja: {x1},{y1},{x2},{y2}",
                }
            )
            continue

        nombre_recorte = Path(nombre_imagen).stem + "_recorte.jpg"
        ruta_recorte = CARPETA_RECORTES / color / nombre_recorte

        guardado_correcto = cv2.imwrite(
            str(ruta_recorte),
            recorte,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not guardado_correcto:
            auditoria.append(
                {
                    "nombre_nuevo": nombre_imagen,
                    "color": color,
                    "estado": "error_guardado",
                    "detalle": str(ruta_recorte),
                }
            )
            continue

        grupo_temporal = construir_grupo_temporal(
            carpeta_origen=carpeta_origen,
            secuencia=secuencia,
            posicion=posicion_secuencia,
            tam_bloque=25,
        )

        dataset.append(
            {
                "nombre_nuevo": nombre_imagen,
                "nombre_original": nombre_original,
                "carpeta_origen": carpeta_origen,
                "color": color,
                "numero_original": numero_original,
                "secuencia": secuencia,
                "posicion_secuencia": posicion_secuencia,
                "grupo_temporal": grupo_temporal,
                "ruta_imagen": str(ruta_imagen.relative_to(RAIZ)),
                "ruta_json": str(ruta_json.relative_to(RAIZ)),
                "ruta_recorte": str(ruta_recorte.relative_to(RAIZ)),
                "ancho_imagen": ancho_imagen,
                "alto_imagen": alto_imagen,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "ancho_recorte": ancho_recorte,
                "alto_recorte": alto_recorte,
                "cantidad_shapes": cantidad_shapes,
            }
        )

        auditoria.append(
            {
                "nombre_nuevo": nombre_imagen,
                "color": color,
                "estado": "correcto",
                "detalle": "",
            }
        )

    dataframe_dataset = pd.DataFrame(dataset)
    dataframe_auditoria = pd.DataFrame(auditoria)

    dataframe_dataset.to_csv(
        CSV_DATASET,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    dataframe_auditoria.to_csv(
        CSV_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    print("\n========================================")
    print("RESUMEN DE EXTRACCIÓN")
    print("========================================")
    print(f"Filas en el mapa: {len(mapa)}")
    print(f"Recortes correctos: {len(dataframe_dataset)}")
    print(f"Auditoría: {CSV_AUDITORIA}")
    print(f"Dataset: {CSV_DATASET}")

    if not dataframe_auditoria.empty:
        print("\nEstados encontrados:")
        print(
            dataframe_auditoria["estado"]
            .value_counts()
            .to_string()
        )

    if not dataframe_dataset.empty:
        print("\nRecortes por color:")
        print(
            dataframe_dataset["color"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nGrupos temporales por color:")
        resumen_grupos = (
            dataframe_dataset
            .groupby("color")["grupo_temporal"]
            .nunique()
            .sort_index()
        )
        print(resumen_grupos.to_string())

        for color in sorted(COLORES_VALIDOS):
            filas_color = dataframe_dataset[
                dataframe_dataset["color"] == color
            ]

            rutas_color = [
                RAIZ / ruta
                for ruta in filas_color["ruta_recorte"].tolist()
            ]

            ruta_mosaico = (
                CARPETA_RESULTADOS /
                f"mosaico_{color}.jpg"
            )

            crear_mosaico(
                rutas_recortes=rutas_color,
                color=color,
                ruta_salida=ruta_mosaico,
            )

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()