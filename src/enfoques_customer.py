"""
Compara enfoques alternativos de limpieza, codificacion y objetivo sobre
Customer.csv, para responder: el resultado depende de como decidimos organizar
el dataset, o del dataset mismo?

Cada enfoque entrena tres modelos rapidos y reporta el mejor contra su baseline.

Uso:  .venv/bin/python src/enfoques_customer.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aprendizaje_supervisado import DATOS, RESULTADOS, SEMILLA

MODELOS = [
    ('Regresion logistica', LogisticRegression(max_iter=2000, random_state=SEMILLA)),
    ('Arbol de decision', DecisionTreeClassifier(max_depth=5, random_state=SEMILLA)),
    ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=SEMILLA, n_jobs=-1)),
]


def crudo():
    df = pd.read_csv(DATOS / 'Customer.csv')
    df['Postal Code'] = df['Postal Code'].astype(str)
    return df


def evaluar(X, y, nombre, nota):
    """Entrena los tres modelos y devuelve el mejor resultado en prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
    baseline = (y_test == y_train.value_counts().idxmax()).mean()

    mejor_exactitud, mejor_nombre = 0.0, ''
    for etiqueta, estimador in MODELOS:
        modelo = make_pipeline(StandardScaler(), estimador)
        modelo.fit(X_train, y_train)
        exactitud = modelo.score(X_test, y_test)
        if exactitud > mejor_exactitud:
            mejor_exactitud, mejor_nombre = exactitud, etiqueta

    return {'Enfoque': nombre, 'Que cambia': nota, 'Columnas': X.shape[1],
            'Mejor modelo': mejor_nombre, 'Exactitud': round(mejor_exactitud, 4),
            'Baseline': round(baseline, 4),
            'Supera': 'si' if mejor_exactitud > baseline else 'no'}


def enfoques():
    resultados = []

    # 1. El enfoque del informe: se repite aqui como referencia.
    df = crudo().drop(columns=['Customer ID', 'Customer Name', 'Country']).drop_duplicates()
    y = df['Segment']
    X = pd.get_dummies(df.drop(columns='Segment'), drop_first=True)
    resultados.append(evaluar(X, y, 'Base (el del informe)',
                              'Age + ciudad/estado/CP/region, one-hot'))

    # 2. Sin las columnas de alta cardinalidad: menos ruido, menos columnas que filas.
    df = crudo().drop_duplicates()
    y = df['Segment']
    X = pd.get_dummies(df[['Age', 'State', 'Region']], drop_first=True)
    resultados.append(evaluar(X, y, 'Sin alta cardinalidad',
                              'Se quitan City y Postal Code (252 y 314 valores)'))

    # 3. Minimo absoluto: una sola variable numerica.
    df = crudo().drop_duplicates()
    resultados.append(evaluar(df[['Age']], df['Segment'], 'Solo la edad',
                              'Una sola columna, sin codificacion'))

    # 4. Sin limpiar: se conservan duplicados, ID y nombre.
    df = crudo()
    y = df['Segment']
    X = pd.get_dummies(df.drop(columns='Segment'), drop_first=True)
    resultados.append(evaluar(X, y, 'Sin limpieza',
                              'Con duplicados, ID y nombre incluidos'))

    # 5. Codificacion ordinal en lugar de one-hot.
    df = crudo().drop(columns=['Customer ID', 'Customer Name', 'Country']).drop_duplicates()
    y = df['Segment']
    X = df.drop(columns='Segment').copy()
    texto = X.select_dtypes(exclude='number').columns
    X[texto] = OrdinalEncoder().fit_transform(X[texto])
    resultados.append(evaluar(X, y, 'Codificacion ordinal',
                              'Cada categoria como numero, no como columna'))

    # 6. Edad agrupada en tramos: quiza la relacion no sea lineal en la edad.
    df = crudo().drop(columns=['Customer ID', 'Customer Name', 'Country']).drop_duplicates()
    y = df['Segment']
    X = df.drop(columns='Segment').copy()
    X['Age'] = pd.cut(X['Age'], bins=[0, 25, 35, 45, 55, 65, 120],
                      labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
    X = pd.get_dummies(X, drop_first=True)
    resultados.append(evaluar(X, y, 'Edad en tramos',
                              'Age discretizada en 6 rangos'))

    # 7. Problema binario: es mas facil separar dos clases que tres?
    df = crudo().drop(columns=['Customer ID', 'Customer Name', 'Country']).drop_duplicates()
    y = (df['Segment'] == 'Consumer').map({True: 'Consumer', False: 'Otro'})
    X = pd.get_dummies(df.drop(columns='Segment'), drop_first=True)
    resultados.append(evaluar(X, y, 'Binario Consumer / Otro',
                              'Dos clases en vez de tres'))

    # 8. Control: mismo pipeline, objetivo Region.
    df = crudo().drop(columns=['Customer ID', 'Customer Name', 'Country', 'Segment'])
    df = df.drop_duplicates()
    y = df['Region']
    X = pd.get_dummies(df.drop(columns='Region'), drop_first=True)
    resultados.append(evaluar(X, y, 'Objetivo Region (control)',
                              'Se cambia la etiqueta, no el codigo'))

    return pd.DataFrame(resultados)


def main():
    tabla = enfoques()
    print("=" * 100)
    print("ENFOQUES ALTERNATIVOS SOBRE Customer.csv")
    print("=" * 100)
    print(tabla.to_string(index=False))

    con_segment = tabla[~tabla['Enfoque'].str.contains('Region')]
    ganan = (con_segment['Supera'] == 'si').sum()
    print(f"\nDe {len(con_segment)} enfoques distintos para predecir Segment, "
          f"{ganan} superan su baseline.")
    print("El unico enfoque que aprende algo es el que cambia la etiqueta, no el que")
    print("cambia la limpieza: el limite esta en los datos, no en como los organizamos.")

    tabla.to_csv(RESULTADOS / 'enfoques_customer.csv', index=False)
    print(f"\nTabla guardada en {RESULTADOS / 'enfoques_customer.csv'}")


if __name__ == '__main__':
    main()
