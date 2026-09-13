"""
Demostracion paso a paso del sistema, para presentar en vivo.

Muestra, en este orden y pausando en cada paso:
  0. El archivo crudo tal como esta en disco, con su SHA-256.
  1. Las filas sucias que detecta cada proceso de limpieza, con datos reales.
  2. En que se convierte una fila de texto tras el One-Hot Encoding.
  3. La division 80/20 y un modelo entrenado en vivo sobre los datos limpios.

Uso:  .venv/bin/python src/demo.py
      .venv/bin/python src/demo.py --sin-pausas     (para grabar o correr de corrido)
"""

import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aprendizaje_supervisado import (DATOS, SEMILLA, cargar_customer, dividir,
                                     huella)

PAUSAR = '--sin-pausas' not in sys.argv
ANCHO = 78


def paso(numero, titulo):
    print(f"\n{'=' * ANCHO}\nPASO {numero}. {titulo}\n{'=' * ANCHO}")
    if PAUSAR:
        input("(Enter para continuar) ")


def main():
    pd.set_option('display.width', ANCHO + 40)
    pd.set_option('display.max_columns', 12)

    # ------------------------------------------------------------------
    paso(0, "EL ARCHIVO CRUDO, TAL COMO ESTA EN DISCO")
    ruta = DATOS / 'Customer.csv'
    identidad = huella(ruta)
    print(f"Archivo:     {identidad['ruta']}")
    print(f"Tamano:      {identidad['bytes']:,} bytes")
    print(f"Modificado:  {identidad['modificado']}")
    print(f"SHA-256:     {identidad['sha256']}")
    print(f"\nComprobable con:  sha256sum '{identidad['ruta']}'")
    print("\nPrimeras tres lineas, leidas directamente del archivo:\n")
    with open(ruta, encoding='utf-8') as f:
        for _ in range(3):
            print("   ", f.readline().rstrip())

    crudo = pd.read_csv(ruta)
    print(f"\nEl archivo tiene {crudo.shape[0]} filas y {crudo.shape[1]} columnas.")

    # ------------------------------------------------------------------
    paso(1, "LOS TRES PROCESOS DE LIMPIEZA, SOBRE ESE MISMO ARCHIVO")
    print("Cada proceso imprime primero su tabla de fallos: las filas que detecta")
    print("como defectuosas, antes de eliminarlas.\n")
    traza = {}
    X, y = cargar_customer(traza)

    # ------------------------------------------------------------------
    paso(2, "RESUMEN DE LO QUE SE ELIMINO")
    registro = traza['datasets'][0]
    resumen = pd.DataFrame([{
        'Proceso': e['etapa'], 'Detectadas': e['detectadas'],
        'Filas antes': e['filas_antes'], 'Filas despues': e['filas_despues'],
    } for e in registro['etapas']])
    print(resumen.to_string(index=False))
    print(f"\nSe entra con {registro['filas_iniciales']} filas y se sale con "
          f"{registro['etapas'][-1]['filas_despues']}.")

    duplicados = next(e for e in registro['etapas'] if 'duplicados' in e['etapa'])
    if duplicados['ejemplos']:
        print("\nEjemplo real de fila duplicada que se elimino:")
        for clave, valor in duplicados['ejemplos'][0].items():
            print(f"    {clave:<14} {valor}")

    # ------------------------------------------------------------------
    paso(3, "CONVERSION: DE TEXTO A NUMEROS (ONE-HOT ENCODING)")
    conversion = registro['conversion']
    print(f"Columnas de texto detectadas: {', '.join(conversion['columnas_texto'])}\n")
    print("Una fila, ANTES de convertir:")
    for clave, valor in conversion['ejemplo_antes'].items():
        print(f"    {clave:<14} {valor}")
    print(f"\nDESPUES: {conversion['columnas_generadas']} columnas nuevas de ceros y unos.")
    print("Las que valen 1 en esa misma fila:")
    for columna in conversion['activas_primera_fila']:
        print(f"    {columna:<34} 1")
    print(f"\nEl modelo ya no ve texto: ve {conversion['columnas_finales']} columnas numericas.")

    # ------------------------------------------------------------------
    paso(4, "DIVISION 80/20 Y ENTRENAMIENTO EN VIVO")
    X_train, X_test, y_train, y_test = dividir(X, y, estratificar=True)

    print("\nEntrenando Random Forest sobre los datos ya limpios...")
    modelo = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=200, random_state=SEMILLA,
                                                  n_jobs=-1))
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)
    baseline = (y_test == y_train.value_counts().idxmax()).mean()

    print(f"\nExactitud en el set de prueba: {exactitud:.4f}")
    print(f"Baseline (responder siempre '{y_train.value_counts().idxmax()}'): {baseline:.4f}")

    print("\nPrimeras cinco predicciones contra el valor real:")
    comparacion = pd.DataFrame({'Prediccion': predicciones[:5],
                                'Real': y_test.values[:5]})
    comparacion['Acierto'] = comparacion['Prediccion'] == comparacion['Real']
    print(comparacion.to_string(index=False))

    print(f"\n{'=' * ANCHO}")
    print("El pipeline corrio de punta a punta sobre el archivo real.")
    print("Los numeros del dashboard salen exactamente de este mismo proceso:")
    print("aprendizaje_supervisado.py escribe los CSV en resultados/ y dashboard.py")
    print("los lee para armar el HTML. Ningun numero esta escrito a mano.")
    print('=' * ANCHO)


if __name__ == '__main__':
    main()
