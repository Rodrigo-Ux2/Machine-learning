"""
Ajusta sobre los datos reales de covid las funciones que describen los apuntes
(1.6 regresion, 1.7 arboles y SVM) y exporta los puntos a resultados/funciones.json
para que el dashboard las dibuje.

  - La recta de la regresion lineal y la curva de la regresion polinomica.
  - Los escalones de un arbol de decision: la misma tarea sin funcion continua.
  - La sigmoide de la regresion logistica y su umbral en 0.5.
  - El hiperplano del SVM lineal con sus margenes y vectores de soporte.

Uso:  .venv/bin/python src/funciones.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from aprendizaje_supervisado import RESULTADOS, SEMILLA, extraer_zip

MUESTRA = 3000          # filas para ajustar las funciones
PUNTOS_DIBUJO = 260     # puntos de dispersion que se exportan al dashboard
RESOLUCION = 120        # puntos con que se traza cada curva


def muestra_covid():
    """Misma limpieza que el pipeline principal, sin codificar variables."""
    ruta = next(r for r in extraer_zip() if os.path.basename(r) == 'covid_19_data.csv')
    df = pd.read_csv(ruta, usecols=['Confirmed', 'Deaths', 'Recovered'])
    df = df[(df['Confirmed'] >= 0) & (df['Deaths'] >= 0) & (df['Recovered'] >= 0)]
    df = df.sample(n=MUESTRA, random_state=SEMILLA)
    return np.log1p(df).reset_index(drop=True)


def redondear(pares):
    return [[round(float(x), 3), round(float(y), 3)] for x, y in pares]


def recta_y_curva(df):
    """Regresion lineal contra regresion polinomica de grado 3 (apunte 1.6)."""
    x = df[['Confirmed']].values
    y = df['Deaths'].values
    rejilla = np.linspace(x.min(), x.max(), RESOLUCION)

    lineal = LinearRegression().fit(x, y)
    coeficientes = np.polyfit(x.ravel(), y, deg=3)
    polinomica = np.poly1d(coeficientes)

    return {
        'puntos': redondear(zip(df['Confirmed'][:PUNTOS_DIBUJO], y[:PUNTOS_DIBUJO])),
        'recta': redondear(zip(rejilla, lineal.predict(rejilla.reshape(-1, 1)))),
        'curva': redondear(zip(rejilla, polinomica(rejilla))),
        'r2_recta': round(float(r2_score(y, lineal.predict(x))), 4),
        'r2_curva': round(float(r2_score(y, polinomica(x.ravel()))), 4),
        'pendiente': round(float(lineal.coef_[0]), 3),
        'sesgo': round(float(lineal.intercept_), 3),
        'x_min': round(float(x.min()), 2), 'x_max': round(float(x.max()), 2),
        'y_min': round(float(y.min()), 2), 'y_max': round(float(y.max()), 2),
    }


def escalones(df):
    """El arbol de decision no ajusta una funcion continua: parte el eje en tramos."""
    x = df[['Confirmed']].values
    y = df['Deaths'].values
    rejilla = np.linspace(x.min(), x.max(), RESOLUCION * 4)

    arbol = DecisionTreeRegressor(max_depth=3, random_state=SEMILLA).fit(x, y)
    cortes = sorted(float(u) for u in arbol.tree_.threshold[arbol.tree_.threshold > -2])

    return {
        'puntos': redondear(zip(df['Confirmed'][:PUNTOS_DIBUJO], y[:PUNTOS_DIBUJO])),
        'escalera': redondear(zip(rejilla, arbol.predict(rejilla.reshape(-1, 1)))),
        'cortes': [round(c, 2) for c in cortes],
        'r2': round(float(r2_score(y, arbol.predict(x))), 4),
        'hojas': int(arbol.get_n_leaves()),
    }


def sigmoide(df):
    """Regresion logistica: probabilidad de que el registro tenga muchas muertes."""
    corte = df['Deaths'].median()
    y = (df['Deaths'] > corte).astype(int).values
    x = df[['Confirmed']].values
    rejilla = np.linspace(x.min(), x.max(), RESOLUCION)

    modelo = LogisticRegression().fit(x, y)
    probabilidades = modelo.predict_proba(rejilla.reshape(-1, 1))[:, 1]

    # x donde la sigmoide cruza 0.5: es el umbral de decision del modelo.
    umbral = float(-modelo.intercept_[0] / modelo.coef_[0][0])

    return {
        'curva': redondear(zip(rejilla, probabilidades)),
        'positivos': redondear(zip(df['Confirmed'][y == 1][:130], np.ones((y == 1).sum())[:130])),
        'negativos': redondear(zip(df['Confirmed'][y == 0][:130], np.zeros((y == 0).sum())[:130])),
        'umbral': round(umbral, 2),
        'peso': round(float(modelo.coef_[0][0]), 3),
        'sesgo': round(float(modelo.intercept_[0]), 3),
        'exactitud': round(float(accuracy_score(y, modelo.predict(x))), 4),
        'corte_muertes': round(float(corte), 2),
        'x_min': round(float(x.min()), 2), 'x_max': round(float(x.max()), 2),
    }


def hiperplano(df):
    """SVM lineal: la recta que separa las clases y sus dos margenes (apunte 1.7)."""
    corte = df['Deaths'].median()
    y = (df['Deaths'] > corte).astype(int).values
    X = df[['Confirmed', 'Recovered']].values

    # Muestra reducida: el SVM lineal sobre 3000 filas no aporta un dibujo distinto
    # y tarda mucho mas.
    X_ajuste, y_ajuste = X[:800], y[:800]
    modelo = SVC(kernel='linear', C=1.0).fit(X_ajuste, y_ajuste)

    w, b = modelo.coef_[0], float(modelo.intercept_[0])
    x_min, x_max = float(X[:, 0].min()), float(X[:, 0].max())
    extremos = np.array([x_min, x_max])

    def linea(desplazamiento):
        # w0*x + w1*y + b = desplazamiento  ->  y = (desplazamiento - b - w0*x) / w1
        return redondear(zip(extremos, (desplazamiento - b - w[0] * extremos) / w[1]))

    soportes = modelo.support_vectors_[:60]

    return {
        'clase_1': redondear(X_ajuste[y_ajuste == 1][:130]),
        'clase_0': redondear(X_ajuste[y_ajuste == 0][:130]),
        'frontera': linea(0),
        'margen_superior': linea(1),
        'margen_inferior': linea(-1),
        'soportes': redondear(soportes),
        'total_soportes': int(len(modelo.support_vectors_)),
        'muestras': int(len(X_ajuste)),
        'exactitud': round(float(modelo.score(X_ajuste, y_ajuste)), 4),
        'ancho_margen': round(float(2 / np.linalg.norm(w)), 3),
        'corte_muertes': round(float(corte), 2),
        'x_min': round(x_min, 2), 'x_max': round(x_max, 2),
        'y_min': round(float(X[:, 1].min()), 2), 'y_max': round(float(X[:, 1].max()), 2),
    }


def main():
    df = muestra_covid()
    salida = {
        'muestra': int(len(df)),
        'recta_curva': recta_y_curva(df),
        'escalones': escalones(df),
        'sigmoide': sigmoide(df),
        'hiperplano': hiperplano(df),
    }
    destino = RESULTADOS / 'funciones.json'
    destino.write_text(json.dumps(salida, ensure_ascii=False), encoding='utf-8')

    print(f"Funciones ajustadas sobre {len(df)} registros de covid.")
    print(f"  Recta:      y = {salida['recta_curva']['pendiente']}x "
          f"+ ({salida['recta_curva']['sesgo']}), R2 {salida['recta_curva']['r2_recta']}")
    print(f"  Polinomica: grado 3, R2 {salida['recta_curva']['r2_curva']}")
    print(f"  Arbol:      {salida['escalones']['hojas']} escalones, "
          f"R2 {salida['escalones']['r2']}")
    print(f"  Sigmoide:   umbral en x = {salida['sigmoide']['umbral']}, "
          f"exactitud {salida['sigmoide']['exactitud']}")
    print(f"  SVM:        {salida['hiperplano']['total_soportes']} vectores de soporte de "
          f"{salida['hiperplano']['muestras']} muestras, "
          f"margen {salida['hiperplano']['ancho_margen']}, "
          f"exactitud {salida['hiperplano']['exactitud']}")
    print(f"\nEscrito en {destino}")


if __name__ == '__main__':
    main()
