from pathlib import Path
import json
import time

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parent.parent

CSV_DATASET = (
    RAIZ
    / "clasificador_color"
    / "datos"
    / "dataset_recortes.csv"
)

CARPETA_MODELOS = (
    RAIZ
    / "clasificador_color"
    / "modelos"
)

CARPETA_RESULTADOS = (
    RAIZ
    / "clasificador_color"
    / "resultados"
)

RUTA_MODELO = CARPETA_MODELOS / "clasificador_color.joblib"
RUTA_FEATURES = CARPETA_RESULTADOS / "features_extraidas.csv"
RUTA_METRICAS = CARPETA_RESULTADOS / "metricas_modelos.csv"
RUTA_REPORTE = CARPETA_RESULTADOS / "reporte_mejor_modelo.txt"
RUTA_CONFUSION = CARPETA_RESULTADOS / "matriz_confusion.png"
RUTA_CONFIGURACION = CARPETA_RESULTADOS / "configuracion_modelo.json"

CLASES = [
    "amarillo",
    "azul",
    "naranja",
    "rojo",
    "verde",
]

SEMILLA = 42

# Número máximo de particiones.
# Se reducirá automáticamente si hay pocas agrupaciones.
NUMERO_FOLDS_DESEADO = 5

# Tamaño normalizado del recorte.
TAMANO_NORMALIZADO = 128

# La elipse utiliza el 86 % del ancho y alto.
FACTOR_ELIPSE = 0.86

# Se descartan píxeles casi sin información cromática.
SATURACION_MINIMA = 30

# Se descartan sombras extremadamente oscuras.
VALOR_MINIMO = 20

# Reflejo especular aproximado.
SATURACION_REFLEJO = 55
VALOR_REFLEJO = 235

# Número de bins para histogramas.
BINS_H = 18
BINS_S = 12
BINS_V = 12
BINS_LAB = 12


# ============================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================

def crear_mascara_eliptica(alto, ancho):
    """
    Crea una máscara elíptica centrada para reducir la influencia
    de las esquinas y del fondo del bounding box.
    """
    mascara = np.zeros((alto, ancho), dtype=np.uint8)

    centro = (ancho // 2, alto // 2)

    ejes = (
        max(1, int(ancho * FACTOR_ELIPSE / 2)),
        max(1, int(alto * FACTOR_ELIPSE / 2)),
    )

    cv2.ellipse(
        mascara,
        centro,
        ejes,
        0,
        0,
        360,
        255,
        -1,
    )

    return mascara


def histograma_normalizado(canal, mascara, bins, rango):
    """
    Calcula un histograma normalizado de un canal.
    """
    histograma = cv2.calcHist(
        [canal],
        [0],
        mascara,
        [bins],
        rango,
    )

    histograma = histograma.flatten().astype(np.float32)

    suma = histograma.sum()

    if suma > 0:
        histograma /= suma

    return histograma


def estadisticas_robustas(valores):
    """
    Calcula estadísticas resistentes a reflejos y valores extremos.
    """
    if len(valores) == 0:
        return np.zeros(7, dtype=np.float32)

    percentiles = np.percentile(
        valores,
        [10, 25, 50, 75, 90],
    )

    media = np.mean(valores)
    desviacion = np.std(valores)

    return np.array(
        [
            media,
            desviacion,
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            percentiles[4],
        ],
        dtype=np.float32,
    )


def extraer_features(imagen):
    """
    Extrae características HSV y CIELAB del recorte.

    No intenta segmentar perfectamente la pelota. Utiliza una región
    elíptica central y elimina píxeles poco informativos.
    """
    if imagen is None or imagen.size == 0:
        raise ValueError("Imagen vacía")

    imagen = cv2.resize(
        imagen,
        (TAMANO_NORMALIZADO, TAMANO_NORMALIZADO),
        interpolation=cv2.INTER_AREA,
    )

    imagen_suave = cv2.GaussianBlur(imagen, (5, 5), 0)

    hsv = cv2.cvtColor(imagen_suave, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagen_suave, cv2.COLOR_BGR2LAB)

    alto, ancho = imagen.shape[:2]

    mascara_elipse = crear_mascara_eliptica(alto, ancho)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    mascara_color = (
        (s >= SATURACION_MINIMA)
        & (v >= VALOR_MINIMO)
    )

    mascara_reflejo = (
        (s <= SATURACION_REFLEJO)
        & (v >= VALOR_REFLEJO)
    )

    mascara_valida = (
        (mascara_elipse > 0)
        & mascara_color
        & (~mascara_reflejo)
    )

    # Si quedaron muy pocos píxeles, se relajan las restricciones.
    cantidad_elipse = np.count_nonzero(mascara_elipse)
    cantidad_valida = np.count_nonzero(mascara_valida)

    if cantidad_valida < max(100, int(cantidad_elipse * 0.10)):
        mascara_valida = (
            (mascara_elipse > 0)
            & (v >= 10)
            & (~mascara_reflejo)
        )

    mascara_uint8 = (
        mascara_valida.astype(np.uint8) * 255
    )

    features = []

    # Estadísticas por canal.
    for canal in [h, s, v, l, a, b]:
        valores = canal[mascara_valida]
        features.extend(estadisticas_robustas(valores))

    # Histogramas HSV.
    features.extend(
        histograma_normalizado(
            h,
            mascara_uint8,
            BINS_H,
            [0, 180],
        )
    )

    features.extend(
        histograma_normalizado(
            s,
            mascara_uint8,
            BINS_S,
            [0, 256],
        )
    )

    features.extend(
        histograma_normalizado(
            v,
            mascara_uint8,
            BINS_V,
            [0, 256],
        )
    )

    # Histogramas CIELAB.
    for canal in [l, a, b]:
        features.extend(
            histograma_normalizado(
                canal,
                mascara_uint8,
                BINS_LAB,
                [0, 256],
            )
        )

    # Proporciones informativas.
    total_elipse = max(
        1,
        np.count_nonzero(mascara_elipse),
    )

    proporcion_valida = (
        np.count_nonzero(mascara_valida) / total_elipse
    )

    proporcion_reflejo = (
        np.count_nonzero(
            (mascara_elipse > 0) & mascara_reflejo
        )
        / total_elipse
    )

    proporcion_baja_saturacion = (
        np.count_nonzero(
            (mascara_elipse > 0) & (s < SATURACION_MINIMA)
        )
        / total_elipse
    )

    proporcion_oscura = (
        np.count_nonzero(
            (mascara_elipse > 0) & (v < 40)
        )
        / total_elipse
    )

    features.extend(
        [
            proporcion_valida,
            proporcion_reflejo,
            proporcion_baja_saturacion,
            proporcion_oscura,
        ]
    )

    return np.asarray(features, dtype=np.float32)


# ============================================================
# CARGA DEL DATASET
# ============================================================

def cargar_dataset():
    if not CSV_DATASET.exists():
        raise FileNotFoundError(
            "No se encontró dataset_recortes.csv.\n"
            "Ejecuta primero 01_extraer_recortes.py.\n"
            f"Ruta esperada: {CSV_DATASET}"
        )

    dataframe = pd.read_csv(
        CSV_DATASET,
        encoding="utf-8-sig",
    )

    columnas = {
        "color",
        "grupo_temporal",
        "ruta_recorte",
        "nombre_nuevo",
    }

    faltantes = columnas - set(dataframe.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas en dataset_recortes.csv: "
            f"{sorted(faltantes)}"
        )

    dataframe["color"] = (
        dataframe["color"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe["color"].isin(CLASES)
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No hay muestras con colores válidos."
        )

    return dataframe.reset_index(drop=True)


def construir_matriz_features(dataframe):
    features = []
    etiquetas = []
    grupos = []
    filas_validas = []

    total = len(dataframe)

    print("\nExtrayendo características...")

    for indice, fila in dataframe.iterrows():
        ruta_recorte = RAIZ / str(fila["ruta_recorte"])

        imagen = cv2.imread(str(ruta_recorte))

        if imagen is None:
            print(
                f"Advertencia: no se pudo leer "
                f"{ruta_recorte}"
            )
            continue

        try:
            vector = extraer_features(imagen)
        except Exception as error:
            print(
                f"Advertencia en {ruta_recorte.name}: "
                f"{error}"
            )
            continue

        features.append(vector)
        etiquetas.append(fila["color"])
        grupos.append(str(fila["grupo_temporal"]))
        filas_validas.append(fila.to_dict())

        if (
            (indice + 1) % 50 == 0
            or indice + 1 == total
        ):
            print(
                f"Procesadas {indice + 1}/{total}"
            )

    if not features:
        raise ValueError(
            "No fue posible extraer características."
        )

    X = np.vstack(features)
    y = np.asarray(etiquetas)
    groups = np.asarray(grupos)

    dataframe_valido = pd.DataFrame(filas_validas)

    nombres_features = [
        f"feature_{indice:03d}"
        for indice in range(X.shape[1])
    ]

    dataframe_features = pd.DataFrame(
        X,
        columns=nombres_features,
    )

    dataframe_features.insert(
        0,
        "grupo_temporal",
        groups,
    )

    dataframe_features.insert(
        0,
        "color",
        y,
    )

    dataframe_features.insert(
        0,
        "nombre_nuevo",
        dataframe_valido["nombre_nuevo"].values,
    )

    dataframe_features.to_csv(
        RUTA_FEATURES,
        index=False,
        encoding="utf-8-sig",
    )

    return X, y, groups, dataframe_valido


# ============================================================
# MODELOS
# ============================================================

def crear_modelos():
    modelos = {
        "SVM_RBF": Pipeline(
            [
                ("escalador", StandardScaler()),
                (
                    "clasificador",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=SEMILLA,
                    ),
                ),
            ]
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEMILLA,
        ),

        "KNN": Pipeline(
            [
                ("escalador", StandardScaler()),
                (
                    "clasificador",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        ),

        "Regresion_Logistica": Pipeline(
            [
                ("escalador", StandardScaler()),
                (
                    "clasificador",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=SEMILLA,
                    ),
                ),
            ]
        ),
    }

    return modelos


# ============================================================
# VALIDACIÓN POR GRUPOS
# ============================================================

def determinar_numero_folds(y, groups):
    """
    Determina una cantidad segura de folds.

    Cada clase debe tener varios grupos temporales para aparecer
    tanto en entrenamiento como en validación.
    """
    tabla = pd.DataFrame(
        {
            "color": y,
            "grupo": groups,
        }
    )

    grupos_por_clase = (
        tabla.groupby("color")["grupo"]
        .nunique()
    )

    minimo = int(grupos_por_clase.min())

    folds = min(
        NUMERO_FOLDS_DESEADO,
        minimo,
    )

    if folds < 2:
        raise ValueError(
            "No hay suficientes grupos temporales por clase "
            "para realizar validación agrupada."
        )

    print("\nGrupos temporales por clase:")
    print(grupos_por_clase.to_string())
    print(f"\nNúmero de folds utilizado: {folds}")

    return folds


def evaluar_modelo(
    nombre,
    modelo,
    X,
    y,
    groups,
    validacion,
):
    predicciones = np.empty(
        len(y),
        dtype=object,
    )

    tiempos = []

    print(f"\nEvaluando {nombre}...")

    for numero_fold, (indices_train, indices_test) in enumerate(
        validacion.split(X, y, groups),
        start=1,
    ):
        modelo_fold = clone(modelo)

        X_train = X[indices_train]
        y_train = y[indices_train]

        X_test = X[indices_test]

        inicio = time.perf_counter()

        modelo_fold.fit(X_train, y_train)
        prediccion_fold = modelo_fold.predict(X_test)

        fin = time.perf_counter()

        predicciones[indices_test] = prediccion_fold

        tiempo_por_muestra = (
            (fin - inicio)
            / max(1, len(indices_test))
        )

        tiempos.append(tiempo_por_muestra)

        clases_test = sorted(set(y[indices_test]))

        print(
            f"  Fold {numero_fold}: "
            f"train={len(indices_train)}, "
            f"test={len(indices_test)}, "
            f"clases_test={clases_test}"
        )

    accuracy = accuracy_score(y, predicciones)

    balanced_accuracy = balanced_accuracy_score(
        y,
        predicciones,
    )

    macro_f1 = f1_score(
        y,
        predicciones,
        average="macro",
    )

    weighted_f1 = f1_score(
        y,
        predicciones,
        average="weighted",
    )

    tiempo_medio_ms = np.mean(tiempos) * 1000

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "tiempo_medio_ms": tiempo_medio_ms,
    }

    print(
        f"  Accuracy:          {accuracy:.4f}\n"
        f"  Balanced accuracy: {balanced_accuracy:.4f}\n"
        f"  Macro F1:          {macro_f1:.4f}\n"
        f"  Weighted F1:       {weighted_f1:.4f}"
    )

    return metricas, predicciones


# ============================================================
# RESULTADOS
# ============================================================

def guardar_matriz_confusion(y_real, y_predicha, nombre_modelo):
    matriz = confusion_matrix(
        y_real,
        y_predicha,
        labels=CLASES,
    )

    matriz_normalizada = confusion_matrix(
        y_real,
        y_predicha,
        labels=CLASES,
        normalize="true",
    )

    figura, ejes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASES,
        yticklabels=CLASES,
        ax=ejes[0],
    )

    ejes[0].set_title(
        f"Matriz de confusión\n{nombre_modelo}"
    )
    ejes[0].set_xlabel("Color predicho")
    ejes[0].set_ylabel("Color real")

    sns.heatmap(
        matriz_normalizada,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASES,
        yticklabels=CLASES,
        ax=ejes[1],
    )

    ejes[1].set_title(
        "Matriz normalizada por clase"
    )
    ejes[1].set_xlabel("Color predicho")
    ejes[1].set_ylabel("Color real")

    plt.tight_layout()
    plt.savefig(
        RUTA_CONFUSION,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figura)


def guardar_reporte(y_real, y_predicha, nombre_modelo):
    reporte = classification_report(
        y_real,
        y_predicha,
        labels=CLASES,
        target_names=CLASES,
        digits=4,
        zero_division=0,
    )

    contenido = (
        f"MEJOR MODELO: {nombre_modelo}\n\n"
        "Las métricas se obtuvieron mediante validación "
        "cruzada estratificada por grupos temporales.\n\n"
        f"{reporte}"
    )

    RUTA_REPORTE.write_text(
        contenido,
        encoding="utf-8",
    )

    print("\nReporte del mejor modelo:")
    print(reporte)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    CARPETA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    CARPETA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = cargar_dataset()

    print("========================================")
    print("ENTRENAMIENTO DEL CLASIFICADOR DE COLOR")
    print("========================================")
    print(f"Muestras registradas: {len(dataframe)}")

    print("\nDistribución por color:")
    print(
        dataframe["color"]
        .value_counts()
        .reindex(CLASES)
        .to_string()
    )

    X, y, groups, dataframe_valido = (
        construir_matriz_features(dataframe)
    )

    print("\nMatriz de características:")
    print(f"Muestras: {X.shape[0]}")
    print(f"Features por muestra: {X.shape[1]}")
    print(f"Grupos temporales: {len(np.unique(groups))}")

    numero_folds = determinar_numero_folds(
        y,
        groups,
    )

    validacion = StratifiedGroupKFold(
        n_splits=numero_folds,
        shuffle=True,
        random_state=SEMILLA,
    )

    modelos = crear_modelos()

    resultados = []
    predicciones_modelos = {}

    for nombre, modelo in modelos.items():
        metricas, predicciones = evaluar_modelo(
            nombre=nombre,
            modelo=modelo,
            X=X,
            y=y,
            groups=groups,
            validacion=validacion,
        )

        resultados.append(metricas)
        predicciones_modelos[nombre] = predicciones

    dataframe_resultados = pd.DataFrame(
        resultados
    ).sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        ],
        ascending=False,
    )

    dataframe_resultados.to_csv(
        RUTA_METRICAS,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========================================")
    print("COMPARACIÓN DE MODELOS")
    print("========================================")

    print(
        dataframe_resultados.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.4f}",
        )
    )

    mejor_nombre = dataframe_resultados.iloc[0][
        "modelo"
    ]

    mejor_modelo = clone(modelos[mejor_nombre])

    predicciones_mejor = predicciones_modelos[
        mejor_nombre
    ]

    guardar_reporte(
        y_real=y,
        y_predicha=predicciones_mejor,
        nombre_modelo=mejor_nombre,
    )

    guardar_matriz_confusion(
        y_real=y,
        y_predicha=predicciones_mejor,
        nombre_modelo=mejor_nombre,
    )

    # Entrenamiento final usando todos los datos.
    print(
        f"\nEntrenando {mejor_nombre} "
        "con todas las muestras..."
    )

    mejor_modelo.fit(X, y)

    paquete_modelo = {
        "modelo": mejor_modelo,
        "clases": CLASES,
        "cantidad_features": int(X.shape[1]),
        "tamano_normalizado": TAMANO_NORMALIZADO,
        "factor_elipse": FACTOR_ELIPSE,
        "saturacion_minima": SATURACION_MINIMA,
        "valor_minimo": VALOR_MINIMO,
        "saturacion_reflejo": SATURACION_REFLEJO,
        "valor_reflejo": VALOR_REFLEJO,
        "bins_h": BINS_H,
        "bins_s": BINS_S,
        "bins_v": BINS_V,
        "bins_lab": BINS_LAB,
        "mejor_modelo": mejor_nombre,
        "macro_f1_validacion": float(
            dataframe_resultados.iloc[0]["macro_f1"]
        ),
        "balanced_accuracy_validacion": float(
            dataframe_resultados.iloc[0][
                "balanced_accuracy"
            ]
        ),
        "version_paquete": 1,
    }

    joblib.dump(
        paquete_modelo,
        RUTA_MODELO,
    )

    configuracion = {
        clave: valor
        for clave, valor in paquete_modelo.items()
        if clave != "modelo"
    }

    with open(
        RUTA_CONFIGURACION,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            configuracion,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print("\n========================================")
    print("PROCESO TERMINADO")
    print("========================================")
    print(f"Mejor modelo: {mejor_nombre}")
    print(f"Modelo guardado en:\n{RUTA_MODELO}")
    print(f"Métricas:\n{RUTA_METRICAS}")
    print(f"Matriz de confusión:\n{RUTA_CONFUSION}")
    print(f"Reporte:\n{RUTA_REPORTE}")
    print(f"Features:\n{RUTA_FEATURES}")


if __name__ == "__main__":
    main()