# Machine Learning — Aprendizaje supervisado

Proyecto de Ciencia de Datos y Machine Learning (UMSS, Grupo 5, II/2026) sobre los dos datasets de
la materia:

- **covid_19_data.csv** — regresión de los fallecidos acumulados (`Deaths`).
- **Customer.csv** — clasificación del segmento del cliente (`Segment`).

Incluye el pre-procesamiento de ambos datasets, ocho modelos de scikit-learn (dos por familia:
regresión, árboles, vectores de soporte y redes neuronales), una **red neuronal escrita desde cero**
con NumPy (perceptrón multicapa, feedforward y backpropagation) y una aplicación web para entrenar,
ver los resultados y hacer predicciones con valores propios.

## Resultado

| Dataset | Tarea | Mejor modelo | Red neuronal desde cero |
|---|---|---|---|
| covid_19_data.csv | Regresión de `Deaths` (log1p) | Random Forest — R² 0.9686 | R² 0.9632, brecha entrenamiento–prueba 0.0051 |
| Customer.csv | Clasificación de `Segment` | Ninguno supera el baseline de 0.5097 | Exactitud 0.5097: responde la proporción de clases |

El informe final está en `docs/INFORME FINAL - CIENCIA DE DATOS Y MACHINE LEARNING.docx`; el
análisis técnico detallado, en [docs/INFORME.md](docs/INFORME.md).

## Requisitos

- **Python 3.12** en un entorno virtual `.venv` con pandas y scikit-learn. Python 3.14 todavía no
  tiene paquetes estables de scikit-learn; por eso el entorno está fijado en 3.12.
- **[uv](https://docs.astral.sh/uv/)** para crear el entorno (`pip install uv` o
  `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Node.js / npm** es opcional: solo sirve para el atajo `npm run dev`. Sin Node, todo funciona con
  el comando de Python equivalente.

## Cómo correr el programa

### 1. Crear el entorno (una sola vez)

```bash
npm run setup
```

Sin Node, es lo mismo que:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scikit-learn
```

### 2. Abrir la aplicación

```bash
npm run dev
```

Sin Node:

```bash
.venv/bin/python src/servidor.py             # Linux / macOS
.venv\Scripts\python src\servidor.py         # Windows
```

Se abre el navegador en `http://127.0.0.1:8000/` (si el puerto está ocupado usa el siguiente libre,
hasta el 8010; la terminal muestra la dirección). Para detenerla: `Ctrl+C` en la terminal. Con
`--sin-navegador` no abre el navegador automáticamente.

La aplicación tiene tres vistas en el menú lateral:

| Vista | Qué hace |
|---|---|
| **Entrenar** | Corre los seis scripts en orden con **Entrenar todo** (~5 min) o uno solo con **Correr**. Muestra el estado de cada paso, su tiempo y la salida de la consola en vivo. Se puede cancelar; si se recarga la página, el entrenamiento sigue y el registro se vuelve a mostrar. |
| **Ocho modelos** | El dashboard: comparación de los modelos, costo, funciones ajustadas, diagnóstico de Customer.csv y trazabilidad de la limpieza. |
| **Red neuronal** | Predicciones con valores propios. Cada tarea parte de casos reales del set de prueba (un clic) y valida lo que se escribe: país con sugerencias y nombres en español, fechas dentro del período, conteos no negativos, avisos de extrapolación. Incluye las curvas de pérdida, las métricas y la matriz de confusión. |

Al terminar un entrenamiento, las vistas **Ocho modelos** y **Red neuronal** se recargan solas con
los resultados nuevos. El servidor solo escucha en `127.0.0.1` y solo ejecuta los scripts del
proyecto.

> El repositorio ya trae los resultados de una corrida de referencia en `resultados/`, así que las
> vistas de resultados funcionan sin entrenar. Los scripts son deterministas (semilla 42): volver a
> entrenar da las mismas métricas; solo cambian los tiempos, que dependen de la máquina.

### 3. Predecir desde la terminal (opcional)

```bash
.venv/bin/python src/red_neuronal.py
```

Entrena las dos redes (~15 s), muestra sus métricas y abre un menú para ingresar valores:
`1` predice fallecidos por covid, `2` el segmento de un cliente, `0` sale. Con `--sin-preguntas`
solo entrena, mide y regenera `resultados/red_neuronal.html`.

## Correr cada script a mano

La aplicación hace esto por ti; estos comandos sirven para correr un paso suelto:

```bash
.venv/bin/python src/aprendizaje_supervisado.py    # ~3 min 20 s: limpieza + los 8 modelos
.venv/bin/python src/diagnostico_customer.py       # ~20 s: por qué Customer.csv no tiene señal
.venv/bin/python src/enfoques_customer.py          # ~10 s: 8 formas de preparar Customer.csv
.venv/bin/python src/funciones.py                  # ~3 s: funciones de los apuntes sobre datos reales
.venv/bin/python src/red_neuronal.py --sin-preguntas   # ~15 s: la red neuronal desde cero
.venv/bin/python src/dashboard.py                  # regenera resultados/dashboard.html
```

`dashboard.py` lee lo que escriben los otros scripts; si se corre sin haberlos corrido, falla porque
no tiene de dónde sacar los números.

| Archivo en `resultados/` | Lo escribe | Qué contiene |
|---|---|---|
| `resultados_covid.csv` | `aprendizaje_supervisado.py` | Métricas de los 8 modelos de regresión |
| `resultados_customer.csv` | `aprendizaje_supervisado.py` | Métricas de los 8 modelos de clasificación |
| `resumen.json` | `aprendizaje_supervisado.py` | Mejor modelo y baseline de cada dataset |
| `limpieza.json` | `aprendizaje_supervisado.py` | Rastro de la limpieza: etapas, filas reales descartadas y SHA-256 de los CSV |
| `enfoques_customer.csv` | `enfoques_customer.py` | Los 8 enfoques de preparación comparados |
| `funciones.json` | `funciones.py` | Puntos y coeficientes de las funciones ajustadas |
| `red_neuronal.json` | `red_neuronal.py` | Métricas e historial de pérdida de las dos redes |
| `red_neuronal.html` | `red_neuronal.py` | Página de predicción con los pesos de la red entrenada |
| `dashboard.html` | `dashboard.py` | El dashboard de los ocho modelos |
| `salida_completa.txt` | la aplicación | Salida de `aprendizaje_supervisado.py`, tablas de fallos incluidas |
| `diagnostico_customer.txt` | la aplicación | Salida de `diagnostico_customer.py` |
| `enfoques_customer.txt` | la aplicación | Salida de `enfoques_customer.py` |

Los tres `.txt` los guarda la aplicación al terminar cada paso. A mano se obtienen con
`.venv/bin/python src/diagnostico_customer.py | tee resultados/diagnostico_customer.txt`.

## Verificar

```bash
.venv/bin/python src/test_red_neuronal.py
```

Compara el gradiente de la backpropagation con el gradiente numérico (diferencias finitas) en
regresión y clasificación, y comprueba que la red aprende XOR. Imprime `OK` si todo está bien.

Para comprobar que los resultados salen de estos datos y no de números escritos a mano, el dashboard
muestra el SHA-256 de cada CSV que leyó (sección «De dónde sale cada número de esta página»):

```bash
sha256sum datos/Customer.csv
sha256sum datos/covid/covid_19_data.csv
```

## Demostrar la limpieza en vivo

```bash
.venv/bin/python src/demo.py              # pausa en cada paso (Enter para avanzar)
.venv/bin/python src/demo.py --sin-pausas # de corrido
```

| Paso | Qué muestra |
|---|---|
| 0 | El archivo crudo: ruta, tamaño, fecha, SHA-256 y sus primeras líneas leídas del disco |
| 1 | Los tres procesos de limpieza ejecutándose, con la tabla de fallos de cada uno |
| 2 | Resumen de filas eliminadas por proceso, con una fila duplicada real de ejemplo |
| 3 | Una fila de texto antes y después del One-Hot Encoding |
| 4 | La división 80/20 y un Random Forest entrenado en vivo, con sus primeras predicciones |

## Estructura

```
datos/          Customer.csv y covid.zip (se descomprime solo en datos/covid/)
src/
  aprendizaje_supervisado.py   pre-procesamiento + los 8 modelos
  diagnostico_customer.py      4 experimentos: por qué Customer.csv no tiene señal
  enfoques_customer.py         8 formas de preparar el mismo dataset
  funciones.py                 ajusta las funciones de los apuntes sobre datos reales
  red_neuronal.py              perceptrón multicapa desde cero + predicción en terminal
  test_red_neuronal.py         verificación del gradiente y de XOR
  plantilla_red_neuronal.html  plantilla de la página de predicción
  dashboard.py                 genera el dashboard
  demo.py                      recorrido paso a paso para presentar en vivo
  servidor.py + app.html       la aplicación de npm run dev
resultados/     salidas de cada script (tabla de arriba)
docs/           informe final (.docx), INFORME.md, avances y enunciado de la práctica
package.json    atajos npm run dev y npm run setup
```

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `.venv/bin/python: No such file or directory` | Falta el entorno: `npm run setup` (o los comandos de uv del paso 1). |
| `npm: command not found` | Usar el comando de Python equivalente; Node no es necesario. |
| `No module named 'sklearn'` | Se está usando el Python del sistema; correr con `.venv/bin/python`. |
| La vista de resultados dice «Todavía no hay resultados» | Correr **Entrenar todo** en la vista Entrenar. |
| `Los puertos 8000-8010 estan ocupados` | Cerrar otras instancias de la aplicación (`Ctrl+C`) y volver a correrla. |
