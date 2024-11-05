# Proyecto de Manejo de Datos

Este proyecto proporciona un conjunto de funciones en Python para cargar, guardar, verificar y limpiar datos utilizando la biblioteca `pandas` y la base de datos `SQLite`.

## Funciones Principales

- **Cargar Datos**
  - `cargar_datos(ruta_archivo, tipo="csv")`: Carga datos desde archivos CSV o Excel.

- **Guardar Datos**
  - `guardar_datos(df, ruta_archivo, tipo="csv")`: Guarda un DataFrame en un archivo CSV o Excel.

- **Verificación de Nulos**
  - `verificar_nulos(df)`: Verifica si hay valores nulos en el DataFrame.
  - `eliminar_nulos(df)`: Elimina filas con valores nulos.

- **Relleno de Nulos**
  - `rellenar_nulos(df, metodo="media", cero=None)`: Rellena valores nulos utilizando distintos métodos (media, mediana, moda, valor específico).

- **Verificación y Eliminación de Duplicados**
  - `verificar_duplicados(df)`: Verifica si hay filas duplicadas en el DataFrame.
  - `eliminar_duplicados(df)`: Elimina filas duplicadas.

- **Normalización de Datos**
  - `normalizar_datos(df, metodo)`: Normaliza datos utilizando diferentes métodos.

- **Visualización de Datos**
  - `visualizar_histograma(df, columna, hits)`: Visualiza un histograma de una columna específica.
  - `visualizar_datos(df, x, y)`: Crea un gráfico de dispersión entre dos columnas.
  - `visualizar_todo(df)`: Muestra gráficos de dispersión para todas las combinaciones de columnas numéricas.

- **Manejo de Columnas**
  - `eliminar_columnas(df, columnas)`: Elimina columnas no utilizadas.
  - `ordenar_columnas(df, orden)`: Ordena las columnas del DataFrame.
  - `renombrar_columnas(df, diccionario)`: Renombra columnas según un diccionario de traducción.

- **Base de Datos**
  - `guardar_en_sql(archivo_csv, nombre_tabla, db_name)`: Guarda datos de un archivo CSV en una tabla de una base de datos SQLite.

## Requisitos

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- sqlite3

## Uso

Asegúrate de tener instaladas las bibliotecas requeridas. Puedes instalarlas utilizando pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
