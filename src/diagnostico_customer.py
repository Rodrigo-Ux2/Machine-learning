"""
Diagnostico: por que ningun modelo supera el baseline en Customer.csv.

Responde cuatro preguntas, en orden:
  1. Se puede ajustar un modelo a estos datos?      -> si, perfectamente (y no sirve).
  2. Se arregla buscando hiperparametros?           -> no.
  3. Hay senal real o es ruido?                     -> prueba de permutacion.
  4. El pipeline funciona?                          -> control con otro objetivo.

Uso:  .venv/bin/python src/diagnostico_customer.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aprendizaje_supervisado import (DATOS, RESULTADOS, SEMILLA, cargar_customer,
                                     dividir, limpiar)

REPETICIONES_PERMUTACION = 30


def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def pregunta_1(datos):
    """Un arbol sin limite de profundidad memoriza el entrenamiento entero."""
    titulo("1. SE PUEDE AJUSTAR UN MODELO A ESTOS DATOS?")
    X_train, X_test, y_train, y_test = datos
    arbol = DecisionTreeClassifier(random_state=SEMILLA)  # sin max_depth
    arbol.fit(X_train, y_train)
    entrenamiento = arbol.score(X_train, y_train)
    prueba = arbol.score(X_test, y_test)
    print(f"Arbol sin limite de profundidad ({arbol.get_depth()} niveles, "
          f"{arbol.get_n_leaves()} hojas):")
    print(f"  Exactitud en ENTRENAMIENTO: {entrenamiento:.4f}")
    print(f"  Exactitud en PRUEBA:        {prueba:.4f}")
    print("Se ajusta perfecto a lo que ya vio y falla en lo que no vio: eso es memorizar,")
    print("no aprender. Un modelo solo sirve por lo que acierta en datos nuevos.")
    return entrenamiento, prueba


def pregunta_2(datos):
    """Busqueda de hiperparametros: el techo no se mueve."""
    titulo("2. SE ARREGLA BUSCANDO MEJORES HIPERPARAMETROS?")
    X_train, X_test, y_train, y_test = datos
    malla = {'randomforestclassifier__max_depth': [3, 5, 10, None],
             'randomforestclassifier__min_samples_leaf': [1, 5, 20],
             'randomforestclassifier__max_features': ['sqrt', 0.3]}
    busqueda = GridSearchCV(
        make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=200, random_state=SEMILLA, n_jobs=-1)),
        malla, cv=5, scoring='accuracy', n_jobs=-1)
    busqueda.fit(X_train, y_train)
    print(f"24 combinaciones probadas con validacion cruzada de 5 pliegues.")
    print(f"  Mejor configuracion: "
          f"{ {k.split('__')[1]: v for k, v in busqueda.best_params_.items()} }")
    print(f"  Mejor exactitud en validacion cruzada: {busqueda.best_score_:.4f}")
    print(f"  Esa configuracion en el set de prueba:  {busqueda.score(X_test, y_test):.4f}")
    return busqueda.best_score_


def pregunta_3(datos, baseline):
    """Prueba de permutacion: si al barajar las etiquetas el modelo rinde igual,
    lo que medimos no era senal."""
    titulo("3. HAY SENAL REAL O ES RUIDO? (PRUEBA DE PERMUTACION)")
    X_train, X_test, y_train, y_test = datos
    modelo = make_pipeline(StandardScaler(),
                           DecisionTreeClassifier(max_depth=5, random_state=SEMILLA))

    modelo.fit(X_train, y_train)
    real = modelo.score(X_test, y_test)

    rng = np.random.default_rng(SEMILLA)
    barajadas = []
    for _ in range(REPETICIONES_PERMUTACION):
        y_falso = pd.Series(rng.permutation(y_train.values), index=y_train.index)
        modelo.fit(X_train, y_falso)
        barajadas.append(modelo.score(X_test, y_test))
    barajadas = np.array(barajadas)

    print(f"Exactitud con las etiquetas REALES:              {real:.4f}")
    print(f"Exactitud con las etiquetas BARAJADAS al azar:   "
          f"{barajadas.mean():.4f} +/- {barajadas.std():.4f} "
          f"(min {barajadas.min():.4f}, max {barajadas.max():.4f})")
    print(f"Baseline (responder siempre la clase mayoritaria): {baseline:.4f}")
    mejores = int((barajadas >= real).sum())
    print(f"\n{mejores} de {REPETICIONES_PERMUTACION} modelos entrenados con etiquetas "
          f"al azar igualaron o superaron al modelo real.")
    print("Si romper la relacion entre variables y etiqueta no empeora el resultado,")
    print("es porque no habia relacion que romper.")
    return real, barajadas


def pregunta_4():
    """Control: mismo pipeline, mismas columnas, otro objetivo con senal real."""
    titulo("4. EL PIPELINE FUNCIONA? (CONTROL CON OTRO OBJETIVO)")
    df = pd.read_csv(DATOS / 'Customer.csv')
    df = df.drop(columns=['Customer ID', 'Customer Name', 'Country', 'Segment'])
    df['Postal Code'] = df['Postal Code'].astype(str)
    X, y = limpiar(df, 'Customer.csv -> clasificacion de Region (control)', objetivo='Region')
    datos = dividir(X, y, estratificar=True)
    X_train, X_test, y_train, y_test = datos

    modelo = make_pipeline(StandardScaler(),
                           DecisionTreeClassifier(max_depth=5, random_state=SEMILLA))
    modelo.fit(X_train, y_train)
    base = (y_test == y_train.value_counts().idxmax()).mean()
    exactitud = modelo.score(X_test, y_test)
    cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\nMismo codigo, mismas columnas, objetivo Region en vez de Segment:")
    print(f"  Exactitud en prueba:     {exactitud:.4f}")
    print(f"  Validacion cruzada (5):  {cv.mean():.4f} +/- {cv.std():.4f}")
    print(f"  Baseline:                {base:.4f}")

    # El mismo arbol de la pregunta 1, sin limite de profundidad, sobre este objetivo.
    print("\nEfecto de dejar crecer el arbol (comparar con la pregunta 1):")
    profundo = None
    for limite in [5, 10, None]:
        arbol = make_pipeline(StandardScaler(),
                              DecisionTreeClassifier(max_depth=limite, random_state=SEMILLA))
        arbol.fit(X_train, y_train)
        profundo = arbol.score(X_test, y_test)
        print(f"  max_depth={str(limite):<4} -> exactitud en prueba {profundo:.4f}")
    print("Region si se puede predecir porque el estado la determina: California es West,")
    print("Kentucky es South. Con profundidad suficiente el arbol la aprende entera;")
    print("el mismo arbol sobre Segment se queda en 0.4323. El pipeline aprende cuando")
    print("hay algo que aprender.")
    return exactitud, base, profundo


def main():
    X, y = cargar_customer()
    datos = dividir(X, y, estratificar=True)
    baseline = (datos[3] == datos[2].value_counts().idxmax()).mean()

    entrenamiento, prueba = pregunta_1(datos)
    mejor_cv = pregunta_2(datos)
    real, barajadas = pregunta_3(datos, baseline)
    region, base_region, region_profundo = pregunta_4()

    titulo("RESUMEN")
    print(f"Ajustar el modelo a los datos de entrenamiento:  {entrenamiento:.4f} (trivial)")
    print(f"Predecir datos nuevos:                           {prueba:.4f}")
    print(f"Techo tras buscar 24 combinaciones:              {mejor_cv:.4f}")
    print(f"Mismo modelo con etiquetas al azar:              {barajadas.mean():.4f}")
    print(f"Baseline sin entrenar nada:                      {baseline:.4f}")
    print(f"\nControl: predecir Region con el mismo codigo:    {region:.4f} "
          f"(baseline {base_region:.4f})")
    print(f"Control con el arbol sin limite de la pregunta 1:  {region_profundo:.4f} "
          f"(sobre Segment daba {prueba:.4f})")
    print("\nConclusion: el problema no es el modelo ni los hiperparametros, es que")
    print("Segment no depende de la edad ni de la ubicacion. Falta la variable, no el algoritmo.")


if __name__ == '__main__':
    main()
