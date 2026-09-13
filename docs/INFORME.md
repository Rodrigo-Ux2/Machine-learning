# Informe — Aprendizaje supervisado sobre Customer.csv y covid_19_data.csv

**Tarea:** reproducir el script de preparación de datos y extenderlo para probar dos variantes de
cada familia de aprendizaje supervisado, y decidir cuál es la mejor para cada dataset.

**Entregable ejecutable:** `src/aprendizaje_supervisado.py`
**Evidencia de corrida:** `resultados/salida_completa.txt`
**Tablas:** `resultados/resultados_covid.csv`, `resultados/resultados_customer.csv`
**Diagnóstico de Customer.csv:** `src/diagnostico_customer.py` → `resultados/diagnostico_customer.txt`
**Enfoques alternativos:** `src/enfoques_customer.py` → `resultados/enfoques_customer.csv`
**Funciones ajustadas:** `src/funciones.py` → `resultados/funciones.json`
**Dashboard:** `resultados/dashboard.html`

---

## 1. Conclusión

| Dataset | Tarea | Mejor modelo | Métrica |
|---|---|---|---|
| `covid_19_data.csv` | Regresión de `Deaths` (log1p) | **Random Forest (200 árboles)** | R² 0.9686 · RMSE 0.4973 |
| `Customer.csv` | Clasificación de `Segment` | **Ninguno** | Mejor: 0.4903 vs baseline 0.5097 |

En covid gana la familia de árboles con holgura: Random Forest explica el 96.9 % de la variación del
objetivo, contra 92.5 % del mejor modelo lineal. La ventaja no es marginal ni cara — el bosque
entrena en 5 segundos, mientras que el SVM lineal, que queda séptimo, necesita 75.

En Customer.csv la conclusión es negativa y esa es la respuesta correcta: ninguno de los ocho
modelos supera la exactitud de responder siempre `Consumer`. El dataset no contiene relación entre
el segmento del cliente y su edad o ubicación, y ningún algoritmo puede aprender una relación que
no existe.

---

## 2. Datasets y objetivos

### Customer.csv — 793 filas, 9 columnas

Solo dos columnas son numéricas (`Age` y `Postal Code`, y esta última es un código, no una
cantidad). Las candidatas a etiqueta son `Segment` (3 clases) y `Region` (4 clases). Se eligió
`Segment` como objetivo de **clasificación**, con `Age`, `City`, `State`, `Postal Code` y `Region`
como variables predictoras.

Reparto de clases: Consumer 409, Corporate 236, Home Office 148. La clase mayoritaria representa
el 51.6 % del dataset, y ese porcentaje es el baseline contra el que hay que medir cualquier modelo.

### covid_19_data.csv — 306 429 filas, 8 columnas

Registros diarios acumulados por país/provincia con `Confirmed`, `Deaths` y `Recovered`. Se eligió
`Deaths` como objetivo de **regresión**, a partir de `Confirmed`, `Recovered`, el día de
observación y el país.

Se trabaja con una muestra aleatoria de 20 000 filas (`COVID_MUESTRA`, semilla 42). La razón es de
costo: SVM y redes neuronales escalan de forma cuadrática con el número de muestras, y sobre los
306 429 registros completos el SVR no termina en tiempo razonable. Con 20 000 filas la corrida
completa toma 3 minutos y 20 segundos.

---

## 3. Preparación de datos

Se conservan los tres procesos del código original —incompletitud, ruido/excepciones y
conversión— aplicados a ambos datasets por la misma función `limpiar()`, que imprime las tablas de
fallos antes de cada corrección. Sobre el código original se corrigieron cuatro puntos:

**Selección del objetivo.** El script original tomaba `df_limpio.columns[-1]` como columna objetivo
*después* de aplicar `get_dummies()`. Esa última columna es una variable dummy arbitraria — en la
práctica, `Region_West` —, así que el modelo quedaba entrenado para predecir un artefacto de la
codificación. Ahora el objetivo se nombra explícitamente y se separa **antes** de codificar.

**Columnas descartadas antes de codificar.** `Customer ID` y `Customer Name` tienen 793 valores
distintos en 793 filas: codificarlas genera 793 columnas dummy que no aportan información y sí
ruido. `Country` es constante. En covid se eliminan `SNo` (un contador) y `Last Update` (duplica
`ObservationDate`).

**Nulos: columna en vez de filas.** `Province/State` está vacía en 78 103 registros de covid.
Aplicar `dropna()` con esa columna presente habría borrado un cuarto del dataset; se elimina la
columna y se conservan las filas.

**Outliers sobre datos sesgados.** Los conteos de covid tienen cola larga (media de `Confirmed`
85 671, máximo 5 863 138). El filtro de rango intercuartílico aplicado directamente marca como
atípico a casi todo el dataset. Se aplica `log1p` a `Confirmed`, `Deaths` y `Recovered` antes del
filtro, que además es la transformación adecuada para regresión sobre conteos. También se
descartan los valores negativos, imposibles en un conteo acumulado: el mínimo de `Confirmed` en el
archivo completo es −302 844, un error de captura de la fuente. En la muestra de 20 000
filas usada para esta corrida no cayó ninguno, pero el filtro queda en el código.

Resultado de la limpieza:

| Dataset | Filas iniciales | Duplicados | Atípicos | Filas finales | Columnas tras codificar |
|---|---|---|---|---|---|
| Customer.csv | 793 | 33 | 0 | 775 | 608 |
| covid (muestra) | 20 000 | 36 | 476 | 19 503 | 197 |

División 80/20 con `random_state=42`, estratificada por clase en Customer.csv para que la
proporción de segmentos sea la misma en entrenamiento y prueba.

---

## 4. Modelos evaluados

Dos variantes de cada familia, cada una dentro de un pipeline con `StandardScaler`. El escalado es
obligatorio para SVM, redes neuronales y modelos lineales regularizados; a los árboles no les
afecta, así que se aplica a todos por uniformidad.

| Familia | Variante 1 | Variante 2 | Documento |
|---|---|---|---|
| Regresión | Logística softmax / Lineal | Ridge (L2) | 1.6 |
| Árboles | Árbol de decisión | Random Forest (200) | 1.7 |
| Vectores de soporte | Kernel lineal | Kernel RBF | 1.7 |
| Red neuronal | MLP 1 capa (32) | MLP 2 capas (64, 32) | 1.8 |

La variante Ridge corresponde al término de regularización L2 descrito en el documento 1.6: penaliza
los pesos grandes para reducir el sobreajuste.

---

## 5. Resultados

### 5.1 covid_19_data.csv — regresión de `Deaths` (log1p)

| Familia | Modelo | R² | RMSE | MAE | Segundos |
|---|---|---|---|---|---|
| Árboles | Random Forest (200 árboles) | **0.9686** | **0.4973** | **0.3075** | 5.02 |
| Red neuronal | MLP 2 capas (64,32) | 0.9626 | 0.5422 | 0.3631 | 31.21 |
| Red neuronal | MLP 1 capa (32) | 0.9586 | 0.5709 | 0.3772 | 11.54 |
| Árboles | Árbol de decisión (prof. 8) | 0.9251 | 0.7675 | 0.5290 | 0.26 |
| Regresión | Regresión lineal | 0.9246 | 0.7702 | 0.5156 | 0.48 |
| Regresión | Regresión Ridge (L2) | 0.9246 | 0.7702 | 0.5156 | 0.21 |
| Vectores | SVR kernel lineal | 0.9212 | 0.7874 | 0.4977 | 74.91 |
| Vectores | SVR kernel RBF | 0.8642 | 1.0335 | 0.6112 | 59.01 |

Baseline (predecir siempre la media): RMSE 2.8052. Los ocho modelos lo superan con claridad.

Lectura de los resultados:

- **Los árboles ganan porque la relación no es lineal.** Lineal y Ridge dan resultados idénticos
  hasta el cuarto decimal (0.9246), lo que indica que la regularización no cambia nada: el modelo
  lineal no está sobreajustando, simplemente le falta capacidad. Random Forest gana 4.4 puntos de
  R² sobre ellos capturando esa no linealidad.
- **Random Forest mejora mucho sobre un solo árbol** (0.9686 contra 0.9251): el promedio de 200
  árboles sobre muestras distintas reduce la varianza que un único árbol arrastra.
- **Las redes neuronales quedan cerca pero cuestan más**: el MLP de dos capas alcanza 0.9626 en 31
  segundos, seis veces el tiempo del bosque para 0.6 puntos menos de R².
- **El SVM es el peor negocio de la tabla**: el kernel RBF queda último (0.8642) y tarda 59
  segundos. El kernel lineal tarda 75 segundos para igualar lo que la regresión lineal consigue en
  medio segundo. Sobre 19 503 muestras y 197 variables, el SVM no compite.

### 5.2 Customer.csv — clasificación de `Segment`

| Familia | Modelo | Exactitud | F1 macro | Segundos |
|---|---|---|---|---|
| Árboles | Árbol de decisión (prof. 5) | 0.4903 | 0.2213 | 0.02 |
| Vectores | SVM kernel lineal | 0.4710 | 0.3158 | 0.14 |
| Vectores | SVM kernel RBF | 0.4581 | 0.2594 | 0.10 |
| Regresión | Regresión logística (softmax) | 0.4323 | 0.3051 | 0.08 |
| Regresión | Clasificador Ridge (L2) | 0.4323 | 0.3051 | 0.13 |
| Red neuronal | MLP 1 capa (32) | 0.4194 | 0.3163 | 0.92 |
| Árboles | Random Forest (200 árboles) | 0.4129 | 0.2963 | 0.66 |
| Red neuronal | MLP 2 capas (64,32) | 0.3484 | 0.2789 | 2.67 |

**Baseline: 0.5097** — la exactitud de responder siempre `Consumer` sin entrenar nada.

Ningún modelo lo supera. Esto no es un fallo del código ni de los hiperparámetros: es una propiedad
del dataset. `Segment` no guarda relación con la edad ni con la ubicación del cliente, que son las
únicas variables disponibles. El F1 macro bajo (0.22–0.32 contra 0.33 de un clasificador aleatorio
balanceado) confirma que los modelos tampoco aciertan en las clases minoritarias.

El detalle revelador es que el árbol de decisión de profundidad 5, el modelo más simple, es el que
más se acerca al baseline (0.4903): lo hace prediciendo casi siempre la clase mayoritaria, que es
exactamente lo que hace el baseline. Los modelos con más capacidad —Random Forest, MLP de dos
capas— quedan más abajo porque sobreajustan ruido en 608 variables generadas por la codificación
one-hot, con solo 620 filas de entrenamiento. Hay más columnas que filas.


### 5.3 Las funciones que ajusta cada familia

Ajustadas sobre 3 000 registros reales de covid (`src/funciones.py`), tomando
`log(1+Confirmed)` como variable y `log(1+Deaths)` como objetivo. Las gráficas están en el
dashboard; estas son las cifras.

| Función | Forma | R² |
|---|---|---|
| Regresión lineal | Recta: `y = 0.884x - 2.791` | 0.8229 |
| Regresión polinómica grado 3 | Curva | 0.8551 |
| Árbol de decisión (prof. 3) | 8 escalones, cortes en x = 4.52, 6.10, 7.24, 8.07 | 0.8485 |

La curva de grado 3 le gana 3.2 puntos de R² a la recta sobre exactamente los mismos datos: la
relación entre casos confirmados y muertes no es una línea. El árbol, que no ajusta ninguna función
continua sino un valor constante por tramo, llega casi al mismo sitio que la curva. Esto explica en
una sola figura por qué los modelos de árbol ganan la tabla de covid.

La regresión polinómica sigue siendo regresión **lineal** en el sentido del apunte 1.6: los pesos
`w` entran de forma lineal en el modelo, y la no linealidad está en las potencias de la variable.

Para las dos familias de clasificación, ajustadas sobre la misma muestra para separar los registros
con muchas muertes (por encima de la mediana) de los demás:

| Modelo | Parámetros | Exactitud |
|---|---|---|
| Regresión logística | Sigmoide con `w = 1.911`, `b = -17.657`; cruza 0.5 en x = 9.24 | 0.8980 |
| SVM lineal | 192 vectores de soporte de 800 muestras, margen de ancho 1.544 | 0.9062 |

La sigmoide no devuelve una clase sino una probabilidad; el umbral de 0.5 es lo que la convierte en
decisión. El SVM, en cambio, traza directamente la frontera con el margen más ancho posible, y solo
los 192 vectores de soporte —el 24 % de las muestras— determinan dónde queda: el resto de los datos
podría moverse sin cambiar la frontera.

---

## 6. ¿No se puede entrenar un modelo que sí se ajuste a Customer.csv?

Sí se puede ajustar, y ese es justamente el punto: **ajustarse a los datos de entrenamiento y
predecir datos nuevos son dos cosas distintas.** El script `src/diagnostico_customer.py` responde la
pregunta con cuatro experimentos; la salida está en `resultados/diagnostico_customer.txt`.

### 6.1 Sí se puede ajustar — y no sirve de nada

Un árbol de decisión sin límite de profundidad crece hasta 83 niveles y 310 hojas sobre las 620
filas de entrenamiento:

| | Exactitud |
|---|---|
| Entrenamiento | 0.9790 |
| Prueba | 0.4323 |

El modelo memorizó casi fila por fila lo que ya había visto y falla en lo que no vio. Eso es
sobreajuste, no aprendizaje. Un modelo solo vale por lo que acierta en datos nuevos, y el valor que
cuenta es el de la columna de prueba.

(No llega a 1.0000 en entrenamiento por un detalle revelador: hay filas con exactamente las mismas
variables y distinto `Segment`. Dos clientes idénticos en edad y ubicación pertenecen a segmentos
distintos, así que ni memorizando se puede acertar a todo.)

### 6.2 Ajustar hiperparámetros tampoco lo arregla

Búsqueda en malla sobre Random Forest: 24 combinaciones de profundidad, tamaño mínimo de hoja y
número de variables por corte, con validación cruzada de 5 pliegues.

| | Exactitud |
|---|---|
| Mejor configuración en validación cruzada | 0.5145 |
| Esa misma configuración en el set de prueba | 0.5032 |
| Baseline | 0.5097 |

La mejor configuración encontrada es `max_depth=3`: un bosque tan podado que, en la práctica,
responde casi siempre la clase mayoritaria. La búsqueda de hiperparámetros llega *al* baseline
reproduciéndolo, no lo supera.

### 6.3 Prueba de permutación: no hay señal que aprender

Se entrenó el mismo modelo 30 veces con las etiquetas **barajadas al azar**, destruyendo a propósito
cualquier relación entre las variables y el segmento.

| | Exactitud en prueba |
|---|---|
| Etiquetas reales | 0.4903 |
| Etiquetas barajadas (30 corridas) | 0.4983 ± 0.0193 (mín 0.4323, máx 0.5161) |

**25 de las 30 corridas con etiquetas al azar igualaron o superaron al modelo entrenado con las
etiquetas reales.** Si romper la relación entre variables y etiqueta no empeora el resultado, es
porque no había relación que romper. Esta es la evidencia más fuerte del informe.

### 6.4 Control: el mismo código sí aprende cuando hay algo que aprender

Mismo pipeline, mismas columnas, mismo árbol — solo se cambia el objetivo de `Segment` a `Region`:

| Objetivo | max_depth=5 | max_depth=10 | Sin límite | Baseline |
|---|---|---|---|---|
| `Region` | 0.6711 | 0.8684 | **1.0000** | 0.3224 |
| `Segment` | 0.4903 | — | 0.4323 | 0.5097 |

El árbol sin límite que sobre `Segment` se queda en 0.4323 alcanza **exactitud perfecta** sobre
`Region`, porque la relación existe y es determinista: el estado determina la región (California es
West, Kentucky es South). El código, la limpieza y la partición funcionan.

### 6.5 Entonces, ¿qué haría falta?

No un modelo distinto: **variables distintas**. `Segment` describe el tipo de cliente (consumidor,
corporativo, oficina en casa), y eso se refleja en su comportamiento de compra —volumen, frecuencia,
categorías de producto, descuentos, tipo de envío—, no en su edad ni en su código postal. Ese
comportamiento vive en la tabla de órdenes del dataset original de Superstore, que no forma parte de
los archivos entregados. Con `Customer.csv` solo, la tarea no tiene solución; con las órdenes
unidas por `Customer ID`, probablemente sí.


### 6.6 ¿Y si limpiáramos u organizáramos el dataset de otra forma?

La pregunta correcta: ¿el resultado es consecuencia de cómo decidimos limpiar y codificar, o del
dataset mismo? `src/enfoques_customer.py` prueba ocho preparaciones distintas del mismo archivo,
entrenando tres modelos sobre cada una y quedándose con el mejor.

| Enfoque | Qué cambia | Columnas | Exactitud | Baseline | ¿Supera? |
|---|---|---|---|---|---|
| Base (el del informe) | Age + ciudad/estado/CP/región, one-hot | 608 | 0.4903 | 0.5097 | no |
| Sin alta cardinalidad | Se quitan City y Postal Code | 44 | 0.5157 | 0.5157 | no |
| Solo la edad | Una sola columna, sin codificación | 1 | 0.5157 | 0.5157 | no |
| Sin limpieza | Con duplicados, ID y nombre incluidos | 2192 | 0.5157 | 0.5157 | no |
| Codificación ordinal | Cada categoría como número, no como columna | 5 | 0.5097 | 0.5097 | no |
| Edad en tramos | Age discretizada en 6 rangos | 612 | 0.4968 | 0.5097 | no |
| Binario Consumer / Otro | Dos clases en vez de tres | 608 | 0.4968 | 0.5097 | no |
| **Objetivo Region (control)** | **Se cambia la etiqueta, no el código** | 605 | **1.0000** | 0.3224 | **sí** |

**Siete formas distintas de organizar los datos, cero que superen su baseline.** Quitar las columnas
de alta cardinalidad, no limpiar nada, cambiar one-hot por codificación ordinal, agrupar la edad en
tramos, simplificar a un problema binario: ninguna mueve el techo. Varias lo empatan exacto
(0.5157 contra 0.5157), que es la firma de un modelo que terminó respondiendo siempre la clase
mayoritaria.

El único cambio que produce un modelo que aprende es el que **cambia la etiqueta**, no el que cambia
la preparación. Con el mismo código y las mismas columnas, predecir `Region` da exactitud perfecta.

La respuesta, entonces: el resultado **no** depende de cómo decidimos limpiar y organizar el
dataset. Depende de qué se puede predecir con las columnas que hay.

---

## 7. Respuesta a la pregunta de la tarea

**¿Cuál es el mejor modelo para los datasets que tenemos?**

Para `covid_19_data.csv`: **Random Forest**. Mejor R², mejor RMSE, mejor MAE, y sexto lugar en
tiempo de entrenamiento — gana en calidad sin pagar el costo de los modelos que le siguen.

Para `Customer.csv`: **ninguno es utilizable**. La recomendación técnica no es elegir el menos malo,
sino reconocer que el dataset no soporta la tarea; la sección 6 lo demuestra con una prueba de
permutación y un control. Para que sirviera harían falta variables con relación real con el
segmento del cliente (historial de compras, volumen, frecuencia, tipo de producto), que este
archivo no contiene.

---

## 8. Limitaciones

- La muestra de 20 000 filas de covid es una restricción de tiempo de cómputo, no metodológica. La
  constante `COVID_MUESTRA` permite subirla.
- `Confirmed` y `Recovered` están fuertemente correlacionados con `Deaths` por construcción: son
  conteos acumulados del mismo brote. El R² alto refleja esa correlación y no debe leerse como
  capacidad de predicción a futuro; el modelo estima muertes del mismo día, no las pronostica.
- No se hizo búsqueda de hiperparámetros. Los valores usados son los de referencia de cada
  documento de clase. Una búsqueda en malla sobre Random Forest y los MLP podría mover los
  resultados de covid, pero no cambiaría la conclusión de Customer.csv, donde el problema es la
  ausencia de señal.
- La evaluación usa una sola partición 80/20. Una validación cruzada de 5 pliegues daría intervalos
  de confianza en vez de un número puntual.

---

## 9. Cómo reproducir

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scikit-learn
.venv/bin/python src/aprendizaje_supervisado.py    # 3 min 20 s
.venv/bin/python src/diagnostico_customer.py       # 1 min, diagnóstico de Customer.csv
.venv/bin/python src/enfoques_customer.py          # enfoques alternativos de preparación
.venv/bin/python src/funciones.py                  # ajusta las funciones de los apuntes
.venv/bin/python src/dashboard.py                  # genera el dashboard
```

El script es determinista: `random_state=42` en la muestra, la partición y cada modelo. Dos
corridas sobre la misma máquina dan las mismas cifras.

---

## 10. Cómo demostrar que el sistema realmente limpia los datos

El dashboard es una **foto de una corrida**, no una aplicación que lea los CSV al abrirse. Los
números no están escritos a mano: `aprendizaje_supervisado.py` procesa los archivos y escribe
`resultados/*.csv` y `resultados/*.json`; `dashboard.py` lee esos archivos y genera el HTML. Pero
para demostrarlo hay tres niveles de evidencia, de menor a mayor:

### 10.1 Demostración en vivo — `src/demo.py`

El recorrido completo en la terminal, pausando en cada paso (Enter para avanzar):

```
.venv/bin/python src/demo.py
```

| Paso | Qué muestra |
|---|---|
| 0 | El archivo crudo: ruta, tamaño, fecha y SHA-256, y sus primeras líneas leídas directamente del disco |
| 1 | Los tres procesos de limpieza ejecutándose, con la tabla de fallos de cada uno |
| 2 | Resumen de filas eliminadas por proceso, con una fila duplicada real de ejemplo |
| 3 | Una fila de texto antes y después del One-Hot Encoding |
| 4 | La división 80/20 y un Random Forest entrenado en vivo, con sus primeras predicciones |

Para correrlo sin pausas (útil si se graba en video): `.venv/bin/python src/demo.py --sin-pausas`.

### 10.2 El rastro dentro del dashboard

La sección **«De dónde sale cada número de esta página»** muestra, generado desde
`resultados/limpieza.json`:

- Los archivos leídos con su tamaño, fecha de modificación y **SHA-256**. Cualquiera puede correr
  `sha256sum datos/Customer.csv` y comprobar que coincide: si coincide, los resultados salieron de
  ese archivo exacto.
- El embudo de filas por cada proceso de limpieza: cuántas entran, cuántas se detectan, cuántas
  quedan.
- Una **fila real descartada**, con sus valores, y la conversión de una fila de texto a sus columnas
  numéricas.
- La fecha y hora de la corrida que generó la página.

### 10.3 La salida completa de la corrida

`resultados/salida_completa.txt` guarda todo lo que imprimió el pipeline, incluidas las tablas de
fallos con las filas detectadas. Es el registro que respalda cada cifra del informe.

**Resumen de la respuesta**: los datos sí se limpian, y hay tres formas de verificarlo —
ejecutándolo en vivo, comprobando la huella SHA-256 de los archivos de origen contra la que muestra
la página, y leyendo el registro de la corrida.
