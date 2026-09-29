"""
Red neuronal artificial implementada desde cero con NumPy.

Perceptron multicapa (MLP) feedforward entrenado con backpropagation y
descenso de gradiente por mini-lotes con momento. Se entrena sobre los dos
datasets, reutilizando la misma limpieza de aprendizaje_supervisado.py:

    covid_19_data.csv  -> regresion de Deaths (escala log1p), salida lineal
    Customer.csv       -> clasificacion de Segment, salida softmax

Para evitar el sobreajuste, el 80% de entrenamiento se divide otra vez en
entrenamiento (85%) y validacion (15%): el entrenamiento se detiene cuando la
perdida de validacion deja de bajar (early stopping) y se conservan los pesos
de la mejor epoca. El 20% de prueba no se toca hasta medir las metricas.

Al terminar, pide valores al usuario y predice con la red entrenada.

Uso:  .venv/bin/python src/red_neuronal.py
      .venv/bin/python src/red_neuronal.py --sin-preguntas   (solo entrenar y medir)
"""

import contextlib
import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from aprendizaje_supervisado import (DATOS, RESULTADOS, SEMILLA, cargar_covid,
                                     cargar_customer, dividir, extraer_zip)

ANCHO = 78


# ==========================================================================
# LA RED: PERCEPTRON MULTICAPA
# ==========================================================================

class PerceptronMulticapa:
    """MLP con capas ocultas ReLU y salida lineal (regresion) o softmax (clasificacion).

    capas: neuronas por capa, incluidas entrada y salida, p. ej. [10, 32, 16, 1].
    alfa: regularizacion L2 sobre los pesos (penaliza pesos grandes).
    """

    def __init__(self, capas, tarea, tasa=0.01, momento=0.9, alfa=1e-4, semilla=SEMILLA):
        if tarea not in ('regresion', 'clasificacion'):
            raise ValueError(f"tarea desconocida: {tarea}")
        self.capas, self.tarea = list(capas), tarea
        self.tasa, self.momento, self.alfa = tasa, momento, alfa
        rng = np.random.default_rng(semilla)
        # Inicializacion de He: varianza 2/n_entrada, adecuada para ReLU.
        self.W = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
                  for n_in, n_out in zip(capas[:-1], capas[1:])]
        self.b = [np.zeros(n_out) for n_out in capas[1:]]

    def _salida(self, z):
        if self.tarea == 'regresion':
            return z
        z = z - z.max(axis=1, keepdims=True)  # estabilidad numerica de exp()
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def propagar(self, X):
        """Feedforward: devuelve la activacion de cada capa, empezando por la entrada."""
        activaciones = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = activaciones[-1] @ W + b
            es_ultima = i == len(self.W) - 1
            activaciones.append(self._salida(z) if es_ultima else np.maximum(z, 0))
        return activaciones

    def perdida(self, salida, Y):
        """MSE/2 (regresion) o entropia cruzada (clasificacion), mas la penalizacion L2."""
        if self.tarea == 'regresion':
            datos = 0.5 * np.mean(np.sum((salida - Y) ** 2, axis=1))
        else:
            datos = -np.mean(np.sum(Y * np.log(salida + 1e-12), axis=1))
        return datos + 0.5 * self.alfa * sum(np.sum(W ** 2) for W in self.W)

    def retropropagar(self, activaciones, Y):
        """Backpropagation: gradiente de la perdida respecto de cada peso y sesgo.

        Con salida lineal + MSE/2 y con softmax + entropia cruzada, el error de
        la capa de salida es el mismo: prediccion - objetivo. Desde ahi el
        error viaja hacia atras multiplicado por los pesos y por la derivada
        de ReLU (1 si la neurona estaba activa, 0 si no).
        """
        delta = (activaciones[-1] - Y) / len(Y)
        grad_W, grad_b = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            grad_W[i] = activaciones[i].T @ delta + self.alfa * self.W[i]
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (activaciones[i] > 0)
        return grad_W, grad_b

    def entrenar(self, X, Y, X_val, Y_val, epocas=500, lote=64, paciencia=25):
        """Descenso de gradiente por mini-lotes con momento y early stopping."""
        rng = np.random.default_rng(SEMILLA)
        vel_W = [np.zeros_like(W) for W in self.W]
        vel_b = [np.zeros_like(b) for b in self.b]
        historial = {'entrenamiento': [], 'validacion': []}
        mejor, mejor_epoca, mejores_pesos = np.inf, 0, None

        for epoca in range(1, epocas + 1):
            orden = rng.permutation(len(X))
            for inicio in range(0, len(X), lote):
                idx = orden[inicio:inicio + lote]
                grad_W, grad_b = self.retropropagar(self.propagar(X[idx]), Y[idx])
                for i in range(len(self.W)):
                    vel_W[i] = self.momento * vel_W[i] - self.tasa * grad_W[i]
                    vel_b[i] = self.momento * vel_b[i] - self.tasa * grad_b[i]
                    self.W[i] += vel_W[i]
                    self.b[i] += vel_b[i]

            p_train = self.perdida(self.propagar(X)[-1], Y)
            p_val = self.perdida(self.propagar(X_val)[-1], Y_val)
            if not np.isfinite(p_train):
                raise FloatingPointError(f"La perdida diverge en la epoca {epoca}: "
                                         f"bajar la tasa de aprendizaje ({self.tasa}).")
            historial['entrenamiento'].append(round(float(p_train), 6))
            historial['validacion'].append(round(float(p_val), 6))

            if p_val < mejor:
                mejor, mejor_epoca = p_val, epoca
                mejores_pesos = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            elif epoca - mejor_epoca >= paciencia:
                break

        self.W, self.b = mejores_pesos
        historial['mejor_epoca'] = mejor_epoca
        historial['epocas_corridas'] = epoca
        return historial

    def predecir(self, X):
        """Regresion: valor estimado. Clasificacion: probabilidad de cada clase."""
        salida = self.propagar(X)[-1]
        return salida[:, 0] if self.tarea == 'regresion' else salida


# ==========================================================================
# PREPARACION PARA LA RED
# ==========================================================================

class Estandarizador:
    """z = (x - media) / desviacion, con media y desviacion del set de entrenamiento."""

    def __init__(self, X):
        self.media = X.mean(axis=0)
        self.desv = X.std(axis=0)
        self.desv[self.desv == 0] = 1.0  # columnas constantes: evita dividir entre 0

    def __call__(self, X):
        return (X - self.media) / self.desv


@contextlib.contextmanager
def silencio():
    """Oculta las tablas de limpieza: ya las muestran aprendizaje_supervisado.py y demo.py."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def particionar(X, y, estratificar):
    """Prueba 20% (igual que el resto del proyecto) y validacion 15% del entrenamiento."""
    with silencio():
        X_train, X_test, y_train, y_test = dividir(X, y, estratificar)
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEMILLA,
        stratify=y_train if estratificar else None)
    return X_sub, X_val, X_test, y_sub, y_val, y_test


def titulo(texto):
    print(f"\n{'=' * ANCHO}\n{texto}\n{'=' * ANCHO}")


def describir_entrenamiento(red, historial, n_sub, n_val, n_test):
    print(f"Arquitectura: {' -> '.join(map(str, red.capas))} neuronas "
          f"(ReLU en ocultas, {'lineal' if red.tarea == 'regresion' else 'softmax'} en salida)")
    print(f"Division: entrenamiento {n_sub} / validacion {n_val} / prueba {n_test}")
    print(f"Tasa {red.tasa}, momento {red.momento}, L2 {red.alfa}")
    print(f"Early stopping: mejor epoca {historial['mejor_epoca']} de "
          f"{historial['epocas_corridas']} corridas "
          f"(perdida validacion {historial['validacion'][historial['mejor_epoca'] - 1]:.4f})")


# ==========================================================================
# COVID: REGRESION DE DEATHS
# ==========================================================================

def entrenar_covid():
    titulo("RNA 1. covid_19_data.csv -> regresion de Deaths (escala log1p)")
    rutas = extraer_zip()
    with silencio():
        X, y = cargar_covid(rutas)
    ruta = next(r for r in rutas if os.path.basename(r) == 'covid_19_data.csv')
    crudo = pd.read_csv(ruta, usecols=['ObservationDate', 'Country/Region'])
    fechas = pd.to_datetime(crudo['ObservationDate'], format='%m/%d/%Y')

    X_sub, X_val, X_test, y_sub, y_val, y_test = particionar(X, y, estratificar=False)
    escalar = Estandarizador(X_sub.to_numpy(float))
    a_matriz = lambda d: escalar(d.to_numpy(float))
    a_columna = lambda s: s.to_numpy(float).reshape(-1, 1)

    red = PerceptronMulticapa([X.shape[1], 32, 16, 1], 'regresion', tasa=0.005)
    historial = red.entrenar(a_matriz(X_sub), a_columna(y_sub),
                             a_matriz(X_val), a_columna(y_val))
    describir_entrenamiento(red, historial, len(X_sub), len(X_val), len(X_test))

    def metricas(X_, y_):
        pred = red.predecir(a_matriz(X_))
        return {'R2': r2_score(y_, pred), 'MSE': mean_squared_error(y_, pred),
                'RMSE': float(np.sqrt(mean_squared_error(y_, pred))),
                'MAE': mean_absolute_error(y_, pred),
                'MedAE': median_absolute_error(y_, pred)}

    tabla = pd.DataFrame({'Entrenamiento': metricas(X_sub, y_sub),
                          'Validacion': metricas(X_val, y_val),
                          'Prueba': metricas(X_test, y_test)})
    print("\nMetricas (sobre log1p(Deaths)):")
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
    base = float(np.sqrt(((y_test - y_sub.mean()) ** 2).mean()))
    print(f"Baseline (predecir siempre la media): RMSE {base:.4f}")
    brecha = tabla.loc['R2', 'Entrenamiento'] - tabla.loc['R2', 'Prueba']
    print(f"Brecha de R2 entrenamiento - prueba: {brecha:.4f} "
          f"({'sin sobreajuste apreciable' if brecha < 0.02 else 'posible sobreajuste'})")

    modelo = {'red': red, 'escalar': escalar, 'columnas': list(X.columns),
              'rango': X_sub.agg(['min', 'max']),
              'fecha_inicial': fechas.min(),
              'paises': sorted(crudo.loc[X.index, 'Country/Region'].unique())}
    resumen = {'arquitectura': red.capas, 'metricas': tabla.round(4).to_dict(),
               'baseline_rmse': round(base, 4), 'historial': historial}
    return modelo, resumen


def preguntar_covid(m):
    titulo("PREDICCION DE MUERTES POR COVID CON DATOS DEL USUARIO")
    pais = pedir_categoria("Pais o region (ej. Peru, Mexico, Brazil)", m['paises'])
    fecha = pedir_fecha("Fecha de observacion (AAAA-MM-DD)", m['fecha_inicial'])
    confirmados = pedir_numero("Casos confirmados acumulados")
    recuperados = pedir_numero("Recuperados acumulados")

    fila = {'Confirmed': np.log1p(confirmados), 'Recovered': np.log1p(recuperados),
            'Dia': (fecha - m['fecha_inicial']).days, f'Country/Region_{pais}': 1}
    avisar_extrapolacion(fila, m['rango'], {'Confirmed': confirmados,
                                            'Recovered': recuperados,
                                            'Dia': fecha.date()})
    X = pd.DataFrame([fila]).reindex(columns=m['columnas'], fill_value=0)
    log_muertes = float(m['red'].predecir(m['escalar'](X.to_numpy(float)))[0])
    muertes = max(np.expm1(log_muertes), 0.0)

    print(f"\nSalida de la red: log1p(Deaths) = {log_muertes:.4f}")
    print(f"Muertes acumuladas estimadas: {muertes:,.0f}")
    if confirmados > 0:
        print(f"Letalidad implicita: {100 * muertes / confirmados:.2f}% de los confirmados.")
    print("Interpretacion: la red estima las muertes acumuladas que corresponden a ese "
          "volumen de casos\ny recuperados, en ese pais y en esa etapa de la pandemia, "
          "segun el patron aprendido.")


# ==========================================================================
# CUSTOMER: CLASIFICACION DE SEGMENT
# ==========================================================================

CAMPOS_CUSTOMER = ['City', 'State', 'Postal Code', 'Region']


def entrenar_customer():
    titulo("RNA 2. Customer.csv -> clasificacion de Segment")
    with silencio():
        X, y = cargar_customer()
    crudo = pd.read_csv(DATOS / 'Customer.csv')
    crudo['Postal Code'] = crudo['Postal Code'].astype(str)  # igual que cargar_customer
    clases = np.array(sorted(y.unique()))
    a_one_hot = lambda s: (s.to_numpy()[:, None] == clases).astype(float)

    X_sub, X_val, X_test, y_sub, y_val, y_test = particionar(X, y, estratificar=True)
    escalar = Estandarizador(X_sub.to_numpy(float))
    a_matriz = lambda d: escalar(d.to_numpy(float))

    # Red chica y L2 fuerte: 527 filas contra cientos de columnas one-hot. Con
    # L2 0.01 o 0.1 la perdida de validacion sube desde las primeras epocas
    # (memoriza); L2 1.0 dio la menor perdida de validacion de la busqueda.
    red = PerceptronMulticapa([X.shape[1], 16, len(clases)], 'clasificacion',
                              tasa=0.01, alfa=1.0)
    historial = red.entrenar(a_matriz(X_sub), a_one_hot(y_sub),
                             a_matriz(X_val), a_one_hot(y_val), lote=32)
    describir_entrenamiento(red, historial, len(X_sub), len(X_val), len(X_test))

    def metricas(X_, y_):
        pred = clases[red.predecir(a_matriz(X_)).argmax(axis=1)]
        promedio = {'average': 'macro', 'zero_division': 0}
        return {'Exactitud': accuracy_score(y_, pred),
                'Precision macro': precision_score(y_, pred, **promedio),
                'Recall macro': recall_score(y_, pred, **promedio),
                'F1 macro': f1_score(y_, pred, **promedio),
                'F1 ponderado': f1_score(y_, pred, average='weighted', zero_division=0)}

    tabla = pd.DataFrame({'Entrenamiento': metricas(X_sub, y_sub),
                          'Validacion': metricas(X_val, y_val),
                          'Prueba': metricas(X_test, y_test)})
    print("\nMetricas:")
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))

    pred_test = clases[red.predecir(a_matriz(X_test)).argmax(axis=1)]
    matriz = pd.DataFrame(confusion_matrix(y_test, pred_test, labels=clases),
                          index=[f"real {c}" for c in clases],
                          columns=[f"pred {c}" for c in clases])
    print("\nMatriz de confusion (prueba):")
    print(matriz.to_string())
    mayoritaria = y_sub.value_counts().idxmax()
    base = float((y_test == mayoritaria).mean())
    exactitud = tabla.loc['Exactitud', 'Prueba']
    print(f"Baseline (predecir siempre '{mayoritaria}'): exactitud {base:.4f} -> "
          f"la red {'SUPERA' if exactitud > base else 'NO SUPERA'} al baseline.")

    modelo = {'red': red, 'escalar': escalar, 'columnas': list(X.columns), 'clases': clases,
              'rango': X_sub[['Age']].agg(['min', 'max']), 'supera_baseline': exactitud > base,
              'valores': {c: sorted(crudo.loc[X.index, c].unique()) for c in CAMPOS_CUSTOMER}}
    resumen = {'arquitectura': red.capas, 'metricas': tabla.round(4).to_dict(),
               'matriz_confusion': matriz.to_dict(), 'baseline': round(base, 4),
               'clase_mayoritaria': str(mayoritaria), 'historial': historial}
    return modelo, resumen


def preguntar_customer(m):
    titulo("PREDICCION DEL SEGMENTO DE UN CLIENTE CON DATOS DEL USUARIO")
    edad = pedir_numero("Edad")
    fila = {'Age': edad}
    for campo in CAMPOS_CUSTOMER:
        valor = pedir_categoria(campo, m['valores'][campo])
        fila[f'{campo}_{valor}'] = 1
    avisar_extrapolacion(fila, m['rango'], {'Age': edad})

    X = pd.DataFrame([fila]).reindex(columns=m['columnas'], fill_value=0)
    probabilidades = m['red'].predecir(m['escalar'](X.to_numpy(float)))[0]
    print("\nProbabilidad de cada segmento (salida softmax):")
    for clase, p in sorted(zip(m['clases'], probabilidades), key=lambda t: -t[1]):
        print(f"  {clase:<12} {p:6.1%}  {'#' * round(p * 40)}")
    print(f"Segmento predicho: {m['clases'][probabilidades.argmax()]}")
    if not m['supera_baseline']:
        print("Interpretacion: en prueba la red no supera a predecir siempre la clase "
              "mayoritaria;\nlas probabilidades reflejan sobre todo la proporcion de cada "
              "segmento, no una relacion\nreal entre edad/ubicacion y segmento (ver "
              "INFORME.md, seccion 6).")


# ==========================================================================
# ENTRADA DEL USUARIO (validada)
# ==========================================================================

def pedir(texto):
    try:
        return input(f"{texto}: ").strip()
    except EOFError:
        print()
        sys.exit(0)


def pedir_numero(texto):
    while True:
        crudo = pedir(texto).replace(',', '')
        try:
            valor = float(crudo)
        except ValueError:
            print("  Debe ser un numero.")
            continue
        if not np.isfinite(valor) or valor < 0:
            print("  Debe ser un numero finito mayor o igual a 0.")
            continue
        return valor


def pedir_fecha(texto, minima):
    while True:
        try:
            fecha = datetime.strptime(pedir(texto), '%Y-%m-%d')
        except ValueError:
            print("  Formato invalido; ejemplo: 2020-06-15.")
            continue
        if fecha < minima:
            print(f"  El dataset empieza el {minima.date()}; ingrese una fecha posterior.")
            continue
        return fecha


def pedir_categoria(texto, validos):
    por_minuscula = {str(v).lower(): v for v in validos}
    while True:
        valor = pedir(texto)
        if valor.lower() in por_minuscula:
            return por_minuscula[valor.lower()]
        parecidos = [v for v in validos if valor.lower() in str(v).lower()][:8]
        print(f"  '{valor}' no aparece en los datos de entrenamiento."
              + (f" Parecidos: {', '.join(map(str, parecidos))}" if parecidos else ""))


def avisar_extrapolacion(fila, rango, originales):
    """La red solo es confiable dentro del rango que vio al entrenar."""
    for col, original in originales.items():
        if not rango.loc['min', col] <= fila[col] <= rango.loc['max', col]:
            print(f"  Aviso: {col} = {original} esta fuera del rango de entrenamiento; "
                  "la prediccion es una extrapolacion.")


# ==========================================================================

def main():
    modelo_covid, resumen_covid = entrenar_covid()
    modelo_customer, resumen_customer = entrenar_customer()

    with open(RESULTADOS / 'red_neuronal.json', 'w', encoding='utf-8') as f:
        json.dump({'covid': resumen_covid, 'customer': resumen_customer},
                  f, ensure_ascii=False, indent=2)
    print("\nMetricas e historial de perdida guardados en resultados/red_neuronal.json")

    if '--sin-preguntas' in sys.argv:
        return
    acciones = {'1': (preguntar_covid, modelo_covid),
                '2': (preguntar_customer, modelo_customer)}
    while True:
        opcion = pedir("\n1) Predecir muertes por covid   2) Predecir segmento de cliente   "
                       "0) Salir\nOpcion")
        if opcion == '0':
            break
        if opcion not in acciones:
            print("  Opcion invalida.")
            continue
        funcion, modelo = acciones[opcion]
        funcion(modelo)


if __name__ == '__main__':
    main()
