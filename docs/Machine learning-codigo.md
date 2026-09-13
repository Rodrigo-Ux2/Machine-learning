Código:

```python
import pandas as pd
import zipfile
import glob
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. EXTRACCIÓN Y CARGA DE DATOS
# ==========================================
carpeta_destino = 'datos_covid'
with zipfile.ZipFile('covid.zip', 'r') as zip_ref:
    zip_ref.extractall(carpeta_destino)
print("Archivos CSV extraídos:", glob.glob(f"{carpeta_destino}/*.csv"))

df = pd.read_csv('Customer.csv')
print(f"\nDataset cargado. Dimensiones iniciales: {df.shape}\n")

# ==========================================
# 2. PROCESO 1: INCOMPLETITUD
# ==========================================
print("--- 1. TABLA DE FALLOS: INCOMPLETITUD (VALORES NULOS) ---")
fallos_nulos = df[df.isnull().any(axis=1)]
print(f"Detectadas {fallos_nulos.shape[0]} filas con datos faltantes.")
if not fallos_nulos.empty: print(fallos_nulos.head())

df_limpio = df.dropna()
print(f"-> Dimensión tras limpiar nulos: {df_limpio.shape}\n")

# ==========================================
# 3. PROCESO 2: RUIDO Y EXCEPCIONES
# ==========================================
print("--- 2A. TABLA DE FALLOS: RUIDO (DUPLICADOS) ---")
fallos_duplicados = df_limpio[df_limpio.duplicated(keep=False)]
print(f"Detectadas {fallos_duplicados.shape[0]} filas duplicadas.")
if not fallos_duplicados.empty: print(fallos_duplicados.head())

df_limpio = df_limpio.drop_duplicates()
```

```python
print("\n--- 2B. TABLA DE FALLOS: RUIDO (EXCEPCIONES/OUTLIERS) ---")
columnas_numericas = df_limpio.select_dtypes(include=['number']).columns
mascara_outliers = pd.Series(False, index=df_limpio.index)

for col in columnas_numericas:
    Q1 = df_limpio[col].quantile(0.25)
    Q3 = df_limpio[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    mascara_outliers = mascara_outliers | (df_limpio[col] < limite_inferior) | (df_limpio[col] > limite_superior)
    df_limpio = df_limpio[(df_limpio[col] >= limite_inferior) & (df_limpio[col] <= limite_superior)]

fallos_outliers = df[df.index.isin(mascara_outliers[mascara_outliers].index)]
print(f"Detectadas {fallos_outliers.shape[0]} filas atípicas.")
if not fallos_outliers.empty: print(fallos_outliers.head())
print(f"-> Dimensión tras limpiar ruido: {df_limpio.shape}\n")

# ==========================================
# 4. PROCESO 3: CONVERSIÓN DE DATOS
# ==========================================
print("--- 3. TABLA DE FALLOS: CONVERSIÓN (VARIABLES DE TEXTO) ---")
# Corrección aplicada: Se añadió 'string' para silenciar la advertencia de Pandas
fallos_texto = df_limpio.select_dtypes(include=['object', 'string'])
print(f"Columnas que requieren conversión numérica: {list(fallos_texto.columns)}")
print(fallos_texto.head())

df_limpio = pd.get_dummies(df_limpio, drop_first=True)
print(f"-> Dimensión tras la conversión: {df_limpio.shape}\n")

# ==========================================
# 5. DETERMINACIÓN DE ENTRENAMIENTO Y PRUEBA
# ==========================================
print("--- DIVISIÓN DE DATOS ---")
columna_objetivo = df_limpio.columns[-1]
X = df_limpio.drop(columna_objetivo, axis=1)
y = df_limpio[columna_objetivo]
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

print(f"Set de Entrenamiento (X_train): {X_train.shape[0]} muestras (80%).")
print(f"Set de Prueba (X_test): {X_test.shape[0]} muestras (20%).")
```

## Comando para instalar librerias:

pip install pandas scikit-learn

## Cómo funciona:

Debes crear una carpeta donde debe estar el codigo y los 2 data sets proporcionados por la ing.

## Funcionamiento de código:

El script automatiza la preparación inicial de los datasets utilizando la librería pandas para estructurar datos en forma de tablas y scikit-learn para aplicar las particiones estadísticas.

**Carga y Consolidación**

*   pd.read_csv(): Lee el archivo suelto y lo transforma en un *DataFrame*, una estructura bidimensional similar a una hoja de cálculo.

*   zipfile.ZipFile(...): Abre el archivo comprimido en modo lectura y extrae su contenido en la carpeta datos_descomprimidos.

*   glob.glob(...): Escanea la ruta indicada y recopila dinámicamente todos los archivos que terminen con la extensión .csv.

*   pd.concat(...): Toma la lista de archivos extraídos y los apila verticalmente, logrando la consolidación de datos provenientes de múltiples fuentes en una sola matriz masiva.

**Los Tres Procesos de Limpieza**

*   **Incompletitud**: La instrucción <u>dropna()</u> escanea la tabla y elimina permanentemente cualquier fila que carezca de atributos (valores <mark>NaN</mark>), garantizando que el modelo reciba registros íntegros.

*   <u>**Ruido y Excepciones:** Se</u> utiliza un filtro condicional de comparación (como `df1_limpio['columna'] >= 0`) que evalúa cada fila; solo las que cumplen la condición matemática se mantienen, eliminando datos ilógicos o registros incorrectos.

*   **Conversión de Datos:** La función `pd.get_dummies()` aplica una técnica llamada *One-Hot Encoding*. Detecta automáticamente las columnas de texto y las transforma en nuevas columnas numéricas (con valores 0 y 1), un requisito indispensable para que las variables de entrada puedan ser procesadas por el modelo matemático.

### Determinación de Entrenamiento y Prueba

*   **Separación de Variables:** El código utiliza `drop(columna_objetivo, axis=1)` para aislar las características predictivas en la variable `X`, y extrae la columna objetivo en la variable `y` (la etiqueta que el algoritmo deberá aprender a predecir).

*   `train_test_split(...)`: Es el motor de la división de datos. Mezcla aleatoriamente todos los registros para evitar sesgos de orden y fragmenta el dataset según el parámetro `test_size=0.2`. Envía el 80% a los conjuntos de entrenamiento (`X_train`, `y_train`) para formar el modelo, y reserva el 20% en los conjuntos de validación (`X_test`, `y_test`) para probar posteriormente su capacidad de generalización en un entorno simulado.