"""
Aprendizaje supervisado: comparacion de modelos sobre dos datasets.

Fase 1 - Preparacion (reproduce Machine learning-codigo.md):
    incompletitud -> ruido/excepciones -> conversion -> division train/test.

Fase 2 - Modelado: dos variantes de cada familia vista en clase.
    Regresion (1.6), Arboles (1.7), Vectores de soporte/SVM (1.7), Redes neuronales (1.8).

    Customer.csv         -> clasificacion de 'Segment'
    covid_19_data.csv    -> regresion de 'Deaths' (en escala log1p)

Uso:  .venv/bin/python src/aprendizaje_supervisado.py
"""

import glob
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge,
                                  RidgeClassifier)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RAIZ = Path(__file__).resolve().parent.parent  # el script corre desde cualquier ruta
DATOS = RAIZ / 'datos'
RESULTADOS = RAIZ / 'resultados'

SEMILLA = 42
# ponytail: SVM y MLP escalan cuadraticamente; 20000 filas dan ~3 min de corrida.
# Subir si se quiere mas precision y se tiene tiempo de computo.
COVID_MUESTRA = 20000


# ==========================================================================
# FASE 1. PREPARACION DE DATOS (los tres procesos de limpieza)
# ==========================================================================

def extraer_zip(ruta_zip=DATOS / 'covid.zip', destino=DATOS / 'covid'):
    """Descomprime el zip de covid si aun no fue extraido."""
    if not os.path.isdir(destino):
        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(destino)
    return glob.glob(f"{destino}/**/*.csv", recursive=True)


def huella(ruta):
    """Identidad verificable del archivo de origen: tamano, fecha y SHA-256.

    Permite comprobar que el dashboard se genero a partir de estos archivos y no
    de numeros escritos a mano: el mismo comando sha256sum sobre el CSV debe dar
    el mismo valor.
    """
    ruta = Path(ruta)
    digest = hashlib.sha256(ruta.read_bytes()).hexdigest()
    return {'archivo': ruta.name, 'ruta': str(ruta.relative_to(RAIZ)),
            'bytes': ruta.stat().st_size,
            'modificado': time.strftime('%Y-%m-%d %H:%M', time.localtime(ruta.stat().st_mtime)),
            'sha256': digest}


def muestra_filas(df, n=3):
    """Filas reales del dataset, como texto, para dejar constancia de lo detectado."""
    return [{k: str(v) for k, v in fila.items()}
            for fila in df.head(n).to_dict(orient='records')]


def limpiar(df, nombre, objetivo, columnas_log=(), traza=None):
    """Aplica los tres procesos de limpieza e imprime las tablas de fallos.

    columnas_log: columnas de conteos muy sesgados. Se transforman con log1p
    antes del filtro IQR; sin esto el rango intercuartil elimina casi todo el
    dataset de covid, cuya distribucion tiene cola larga.

    traza: dict opcional donde se registra cada etapa con sus filas reales, para
    que el dashboard pueda mostrar de donde salio cada numero.
    """
    registro = {'dataset': nombre, 'filas_iniciales': int(df.shape[0]),
                'columnas_iniciales': int(df.shape[1]), 'etapas': []}

    def anotar(etapa, detectadas, ejemplos, antes):
        registro['etapas'].append({
            'etapa': etapa, 'detectadas': int(detectadas), 'ejemplos': ejemplos,
            'filas_antes': int(antes), 'filas_despues': int(df.shape[0])})
    print(f"\n{'=' * 70}\nPREPARACION: {nombre}   (dimensiones iniciales: {df.shape})\n{'=' * 70}")

    # --- Proceso 1: incompletitud --------------------------------------
    print("--- 1. TABLA DE FALLOS: INCOMPLETITUD (VALORES NULOS) ---")
    fallos_nulos = df[df.isnull().any(axis=1)]
    print(f"Detectadas {fallos_nulos.shape[0]} filas con datos faltantes.")
    if not fallos_nulos.empty:
        print(fallos_nulos.head())
    antes = df.shape[0]
    df = df.dropna()
    anotar('Incompletitud (nulos)', fallos_nulos.shape[0], muestra_filas(fallos_nulos), antes)
    print(f"-> Dimension tras limpiar nulos: {df.shape}\n")

    # --- Proceso 2A: ruido (duplicados) --------------------------------
    print("--- 2A. TABLA DE FALLOS: RUIDO (DUPLICADOS) ---")
    fallos_duplicados = df[df.duplicated(keep=False)]
    print(f"Detectadas {fallos_duplicados.shape[0]} filas duplicadas.")
    if not fallos_duplicados.empty:
        print(fallos_duplicados.head())
    antes = df.shape[0]
    df = df.drop_duplicates()
    anotar('Ruido (duplicados)', fallos_duplicados.shape[0],
           muestra_filas(fallos_duplicados), antes)

    # --- Proceso 2B: ruido (excepciones / outliers) --------------------
    print("\n--- 2B. TABLA DE FALLOS: RUIDO (EXCEPCIONES/OUTLIERS) ---")
    for col in columnas_log:
        negativos = (df[col] < 0).sum()
        if negativos:
            print(f"'{col}': {negativos} valores negativos imposibles, se descartan.")
            df = df[df[col] >= 0]
        df[col] = np.log1p(df[col])

    columnas_numericas = df.select_dtypes(include=['number']).columns
    mascara_outliers = pd.Series(False, index=df.index)
    for col in columnas_numericas:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mascara_outliers |= (df[col] < limite_inferior) | (df[col] > limite_superior)

    fallos_outliers = df[mascara_outliers]
    print(f"Detectadas {fallos_outliers.shape[0]} filas atipicas.")
    if not fallos_outliers.empty:
        print(fallos_outliers.head())
    antes = df.shape[0]
    df = df[~mascara_outliers]
    anotar('Excepciones (IQR)', fallos_outliers.shape[0], muestra_filas(fallos_outliers), antes)
    print(f"-> Dimension tras limpiar ruido: {df.shape}\n")

    # --- Proceso 3: conversion de datos --------------------------------
    print("--- 3. TABLA DE FALLOS: CONVERSION (VARIABLES DE TEXTO) ---")
    y = df[objetivo]
    X = df.drop(objetivo, axis=1)
    fallos_texto = X.select_dtypes(include=['object', 'string'])
    print(f"Columnas que requieren conversion numerica: {list(fallos_texto.columns)}")
    if not fallos_texto.empty:
        print(fallos_texto.head())
    ejemplo_antes = muestra_filas(fallos_texto, 1)
    columnas_previas = list(X.columns)
    X = pd.get_dummies(X, drop_first=True)
    print(f"-> Dimension tras la conversion: {X.shape}\n")

    if traza is not None:
        nuevas = [c for c in X.columns if c not in columnas_previas]
        activas = [c for c in nuevas if bool(X.iloc[0][c])]
        registro['conversion'] = {
            'columnas_texto': list(fallos_texto.columns),
            'ejemplo_antes': ejemplo_antes[0] if ejemplo_antes else {},
            'columnas_generadas': int(len(nuevas)),
            'activas_primera_fila': activas[:6],
            'columnas_finales': int(X.shape[1]),
        }
        registro['objetivo'] = objetivo
        registro['clases'] = (y.value_counts().to_dict() if y.dtype == object
                              else {'tipo': 'numerico continuo'})
        traza.setdefault('datasets', []).append(registro)

    return X, y


def dividir(X, y, estratificar=False):
    """80% entrenamiento / 20% prueba, con mezcla aleatoria."""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA,
        stratify=y if estratificar else None)
    print("--- DIVISION DE DATOS ---")
    print(f"Set de Entrenamiento (X_train): {X_train.shape[0]} muestras (80%).")
    print(f"Set de Prueba (X_test): {X_test.shape[0]} muestras (20%).")
    return X_train, X_test, y_train, y_test


# ==========================================================================
# CARGA ESPECIFICA DE CADA DATASET
# ==========================================================================

def cargar_customer(traza=None):
    """Clientes: predecir el Segment a partir de edad y ubicacion.

    Se descartan ID y nombre (un valor distinto por fila: get_dummies generaria
    793 columnas inutiles) y Country (constante, sin informacion).
    """
    df = pd.read_csv(DATOS / 'Customer.csv')
    df = df.drop(columns=['Customer ID', 'Customer Name', 'Country'])
    df['Postal Code'] = df['Postal Code'].astype(str)  # es un codigo, no una cantidad
    return limpiar(df, 'Customer.csv -> clasificacion de Segment', objetivo='Segment',
                   traza=traza)


def cargar_covid(rutas_csv, traza=None):
    """Covid: predecir Deaths a partir de fecha, pais, confirmados y recuperados."""
    ruta = next(r for r in rutas_csv if os.path.basename(r) == 'covid_19_data.csv')
    df = pd.read_csv(ruta)
    # Province/State es nulo en 78k filas: se elimina la columna, no las filas
    # (dropna sobre ella borraria una cuarta parte del dataset).
    # SNo es un contador y Last Update duplica ObservationDate.
    df = df.drop(columns=['SNo', 'Province/State', 'Last Update'])
    fechas = pd.to_datetime(df['ObservationDate'], format='%m/%d/%Y')
    df['Dia'] = (fechas - fechas.min()).dt.days
    df = df.drop(columns=['ObservationDate'])
    df = df.sample(n=COVID_MUESTRA, random_state=SEMILLA)
    return limpiar(df, 'covid_19_data.csv -> regresion de Deaths', objetivo='Deaths',
                   columnas_log=['Confirmed', 'Deaths', 'Recovered'], traza=traza)


# ==========================================================================
# FASE 2. MODELOS: DOS VARIANTES DE CADA FAMILIA
# ==========================================================================

def modelos_clasificacion():
    return [
        ('Regresion', 'Regresion logistica (softmax)',
         LogisticRegression(max_iter=2000, random_state=SEMILLA)),
        ('Regresion', 'Clasificador Ridge (L2)',
         RidgeClassifier(alpha=1.0, random_state=SEMILLA)),
        ('Arboles', 'Arbol de decision (prof. 5)',
         DecisionTreeClassifier(max_depth=5, random_state=SEMILLA)),
        ('Arboles', 'Random Forest (200 arboles)',
         RandomForestClassifier(n_estimators=200, random_state=SEMILLA, n_jobs=-1)),
        ('Vectores', 'SVM kernel lineal',
         SVC(kernel='linear', random_state=SEMILLA)),
        ('Vectores', 'SVM kernel RBF',
         SVC(kernel='rbf', random_state=SEMILLA)),
        ('Red neuronal', 'MLP 1 capa (32)',
         MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=SEMILLA)),
        ('Red neuronal', 'MLP 2 capas (64,32)',
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=SEMILLA)),
    ]


def modelos_regresion():
    return [
        ('Regresion', 'Regresion lineal', LinearRegression()),
        ('Regresion', 'Regresion Ridge (L2)', Ridge(alpha=1.0, random_state=SEMILLA)),
        ('Arboles', 'Arbol de decision (prof. 8)',
         DecisionTreeRegressor(max_depth=8, random_state=SEMILLA)),
        ('Arboles', 'Random Forest (200 arboles)',
         RandomForestRegressor(n_estimators=200, random_state=SEMILLA, n_jobs=-1)),
        ('Vectores', 'SVR kernel lineal', SVR(kernel='linear')),
        ('Vectores', 'SVR kernel RBF', SVR(kernel='rbf')),
        ('Red neuronal', 'MLP 1 capa (32)',
         MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=SEMILLA)),
        ('Red neuronal', 'MLP 2 capas (64,32)',
         MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=SEMILLA)),
    ]


def evaluar(modelos, datos, clasificacion):
    """Entrena cada modelo y devuelve una tabla de metricas ordenada."""
    X_train, X_test, y_train, y_test = datos
    filas = []
    for familia, nombre, estimador in modelos:
        # El escalado es obligatorio para SVM, MLP y modelos lineales
        # regularizados; a los arboles no les afecta, asi que se aplica a todos.
        modelo = make_pipeline(StandardScaler(), estimador)
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        segundos = time.perf_counter() - inicio

        if clasificacion:
            metricas = {'Exactitud': accuracy_score(y_test, pred),
                        'F1 macro': f1_score(y_test, pred, average='macro')}
        else:
            metricas = {'R2': r2_score(y_test, pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                        'MAE': mean_absolute_error(y_test, pred)}
        filas.append({'Familia': familia, 'Modelo': nombre, **metricas,
                      'Segundos': round(segundos, 2)})

    orden, ascendente = ('Exactitud', False) if clasificacion else ('R2', False)
    return pd.DataFrame(filas).sort_values(orden, ascending=ascendente).reset_index(drop=True)


def informe(titulo, tabla, clasificacion, y_train, y_test):
    print(f"\n{'=' * 70}\nRESULTADOS: {titulo}\n{'=' * 70}")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    mejor = tabla.iloc[0]
    if clasificacion:
        # Baseline: acertar siempre la clase mayoritaria. Un modelo que no lo
        # supera no aprendio nada del dataset.
        mayoritaria = y_train.value_counts().idxmax()
        base = (y_test == mayoritaria).mean()
        print(f"\nBaseline (predecir siempre '{mayoritaria}'): exactitud {base:.4f}")
        veredicto = ('SUPERA' if mejor['Exactitud'] > base else 'NO SUPERA')
        print(f"Mejor modelo: {mejor['Modelo']} ({mejor['Familia']}), "
              f"exactitud {mejor['Exactitud']:.4f} -> {veredicto} al baseline.")
        detalle = {'baseline': round(float(base), 4), 'clase_mayoritaria': str(mayoritaria),
                   'supera_baseline': bool(mejor['Exactitud'] > base)}
    else:
        base = float(np.sqrt(((y_test - y_train.mean()) ** 2).mean()))
        print(f"\nBaseline (predecir siempre la media): RMSE {base:.4f}")
        print(f"Mejor modelo: {mejor['Modelo']} ({mejor['Familia']}), "
              f"R2 {mejor['R2']:.4f} / RMSE {mejor['RMSE']:.4f}.")
        detalle = {'baseline_rmse': round(base, 4)}

    return {'titulo': titulo, 'mejor_modelo': mejor['Modelo'], 'mejor_familia': mejor['Familia'],
            'filas_entrenamiento': int(len(y_train)), 'filas_prueba': int(len(y_test)),
            **detalle}


def main():
    rutas = extraer_zip()
    print(f"Archivos CSV disponibles: {len(rutas)}")
    traza = {'corrida': time.strftime('%Y-%m-%d %H:%M'), 'origen': []}

    X, y = cargar_customer(traza)
    datos_customer = dividir(X, y, estratificar=True)
    traza['datasets'][-1]['particion'] = {'entrenamiento': int(datos_customer[0].shape[0]),
                                          'prueba': int(datos_customer[1].shape[0])}
    tabla_customer = evaluar(modelos_clasificacion(), datos_customer, clasificacion=True)

    X, y = cargar_covid(rutas, traza)
    datos_covid = dividir(X, y)
    traza['datasets'][-1]['particion'] = {'entrenamiento': int(datos_covid[0].shape[0]),
                                          'prueba': int(datos_covid[1].shape[0])}
    tabla_covid = evaluar(modelos_regresion(), datos_covid, clasificacion=False)

    mejor_c = informe('Customer.csv (clasificacion de Segment)', tabla_customer,
                      True, datos_customer[2], datos_customer[3])
    mejor_r = informe('covid_19_data.csv (regresion de Deaths, escala log1p)', tabla_covid,
                      False, datos_covid[2], datos_covid[3])

    print(f"\n{'=' * 70}\nCONCLUSION\n{'=' * 70}")
    print(f"Customer.csv -> {mejor_c['mejor_modelo']} ({mejor_c['mejor_familia']})")
    print(f"covid_19_data.csv -> {mejor_r['mejor_modelo']} ({mejor_r['mejor_familia']})")

    tabla_customer.to_csv(RESULTADOS / 'resultados_customer.csv', index=False)
    tabla_covid.to_csv(RESULTADOS / 'resultados_covid.csv', index=False)
    ruta_covid = next(r for r in rutas if os.path.basename(r) == 'covid_19_data.csv')
    traza['origen'] = [huella(DATOS / 'Customer.csv'), huella(ruta_covid)]
    with open(RESULTADOS / 'limpieza.json', 'w', encoding='utf-8') as f:
        json.dump(traza, f, ensure_ascii=False, indent=2)

    with open(RESULTADOS / 'resumen.json', 'w', encoding='utf-8') as f:
        json.dump({'customer': mejor_c, 'covid': mejor_r, 'covid_muestra': COVID_MUESTRA},
                  f, ensure_ascii=False, indent=2)
    print("\nTablas guardadas en resultados/ (csv + resumen.json + limpieza.json)")


if __name__ == '__main__':
    main()
