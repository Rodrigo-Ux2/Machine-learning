# Machine Learning — Aprendizaje supervisado

Comparación de ocho modelos supervisados (dos variantes de cada familia vista en clase) sobre los
dos datasets de la materia.

## Estructura

```
datos/        Customer.csv, covid.zip y los CSV extraídos en covid/
src/          aprendizaje_supervisado.py (pipeline + los 8 modelos)
              diagnostico_customer.py (por qué Customer.csv no tiene señal)
              enfoques_customer.py (8 formas de preparar el mismo dataset)
              funciones.py (ajusta las funciones de los apuntes)
              demo.py (recorrido paso a paso para presentar en vivo)
              dashboard.py (genera el dashboard con todo lo anterior)
resultados/   salidas de cada script (ver tabla abajo) + dashboard.html
docs/         INFORME.md y  código original
.venv/        entorno con pandas y scikit-learn
```

## Correr

```bash
.venv/bin/python src/aprendizaje_supervisado.py    # ~3 min 20 s
.venv/bin/python src/diagnostico_customer.py       # ~1 min
.venv/bin/python src/enfoques_customer.py          # ~20 s
.venv/bin/python src/funciones.py                  # ~10 s
.venv/bin/python src/dashboard.py                  # regenera resultados/dashboard.html
```

Los cuatro primeros escriben en `resultados/`; el último los lee para armar el dashboard. Si se
corre `dashboard.py` sin haber corrido los otros, falla: no tiene de dónde sacar los números.

| Archivo en `resultados/` | Lo escribe | Qué contiene |
|---|---|---|
| `resultados_covid.csv` | `aprendizaje_supervisado.py` | Métricas de los 8 modelos de regresión |
| `resultados_customer.csv` | `aprendizaje_supervisado.py` | Métricas de los 8 modelos de clasificación |
| `resumen.json` | `aprendizaje_supervisado.py` | Mejor modelo y baseline de cada dataset |
| `limpieza.json` | `aprendizaje_supervisado.py` | Rastro de la limpieza: etapas, filas reales descartadas y SHA-256 de los CSV |
| `salida_completa.txt` | `aprendizaje_supervisado.py` | Todo lo que imprimió la corrida, tablas de fallos incluidas |
| `diagnostico_customer.txt` | `diagnostico_customer.py` | Los 4 experimentos sobre Customer.csv |
| `enfoques_customer.csv/.txt` | `enfoques_customer.py` | Los 8 enfoques de preparación comparados |
| `funciones.json` | `funciones.py` | Puntos y coeficientes de las funciones ajustadas |
| `dashboard.html` | `dashboard.py` | La página con todo lo anterior |

## Demostrar el sistema en vivo

```bash
.venv/bin/python src/demo.py              # pausa en cada paso (Enter para avanzar)
.venv/bin/python src/demo.py --sin-pausas # de corrido
```

Cinco pasos, pausando en cada uno:

| Paso | Qué muestra |
|---|---|
| 0 | El archivo crudo: ruta, tamaño, fecha, SHA-256 y sus primeras líneas leídas del disco |
| 1 | Los tres procesos de limpieza ejecutándose, con la tabla de fallos de cada uno |
| 2 | Resumen de filas eliminadas por proceso, con una fila duplicada real de ejemplo |
| 3 | Una fila de texto antes y después del One-Hot Encoding |
| 4 | La división 80/20 y un Random Forest entrenado en vivo, con sus primeras predicciones |

### Verificar que los resultados salen de estos archivos

El dashboard muestra el SHA-256 de cada CSV que leyó, en la sección «De dónde sale cada número de
esta página». Se comprueba con:

```bash
sha256sum datos/Customer.csv
sha256sum datos/covid/covid_19_data.csv
```

Si el valor coincide con el de la página, los resultados salieron de ese archivo exacto y no de
números escritos a mano.

Si el entorno no existe:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scikit-learn
```

Python 3.14 del sistema todavía no tiene ruedas estables de scikit-learn; por eso el entorno
está fijado en 3.12.

## Resultado

| Dataset | Tarea | Mejor modelo |
|---|---|---|
| covid_19_data.csv | Regresión de `Deaths` | Random Forest — R² 0.9686 |
| Customer.csv | Clasificación de `Segment` | Ninguno supera el baseline de 0.5097 |

El análisis completo está en [docs/INFORME.md](docs/INFORME.md); las gráficas, en
`resultados/dashboard.html`.
