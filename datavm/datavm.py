import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import sqlite3

# cargar datos csv
def cargar_datos(ruta_archivo, tipo="csv"):
    if tipo == "csv":
        return pd.read_csv(ruta_archivo)
    elif tipo == "excel":
        return pd.read_excel(ruta_archivo)
    else:
        return "Tipo de archivo no soportado"

# guardar datos
def guardar_datos(df, ruta_archivo, tipo="csv"):
    try:
        if tipo == "csv":
            df.to_csv(ruta_archivo, index=False)
        elif tipo == "excel":
            df.to_excel(ruta_archivo, index=False)
        else:
            return "Tipo de archivo no soportado"
        print(f"Datos guardados en {ruta_archivo} correctamente.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")


# Verificar nulos
def verificar_nulos(df):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    nulos = df.isnull().sum()
    
    if nulos.sum() > 0:
        nulos_detalle = nulos[nulos > 0]
        total_nulos = nulos_detalle.sum()
        print(f"Se encontraron {total_nulos} datos nulos en el DataFrame:")
        print(nulos_detalle)
        return nulos_detalle
    else:
        return "No hay valores nulos en el DataFrame."


def eliminar_nulos(df):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Verificar si hay nulos
    if df.isnull().values.any():
        df_limpio = df.dropna()  # Eliminar filas con nulos
        filas_eliminadas = df.shape[0] - df_limpio.shape[0]
        print(f"Se eliminaron {filas_eliminadas} filas con datos nulos.")
        return df_limpio
    else:
        print('No hay datos nulos en el DataFrame.')
        return df  # Devuelve el DataFrame original si no hay nulos

# rellenar datos nulos
def rellenar_nulos(df, metodo="media", cero=None):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    for col in df.columns:
        if metodo == "media":
            df[col] = df[col].fillna(df[col].mean())
        elif metodo == "mediana":
            df[col] = df[col].fillna(df[col].median())
        elif metodo == "moda":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif metodo == "valor" and cero is not None:
            df[col] = df[col].fillna(cero)
        else:
            print(f"Método '{metodo}' no soportado.")
    return df

# Datos duplicados
def verificar_duplicados(df):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    num_duplicados = df.duplicated().sum()
    
    if num_duplicados > 0:
        # Retornar las filas duplicadas
        filas_duplicadas = df[df.duplicated(keep=False)]  # Mantener todas las instancias de duplicados
        print(f"Se encontraron {num_duplicados} filas duplicadas.")
        return filas_duplicadas
    else:
        return "No se encontraron duplicados en el DataFrame."


# Eliminar datos duplicados
def eliminar_duplicados(df):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Contar el número de duplicados antes de eliminarlos
    num_duplicados = df.duplicated().sum()
    
    if num_duplicados > 0:
        df_sin_duplicados = df.drop_duplicates()
        print(f"Se eliminaron {num_duplicados} filas duplicadas.")
        return df_sin_duplicados
    else:
        return "No hay duplicados en el DataFrame."


# normalicacion de datos 
def normalizar_datos(df, metodo):
    if metodo == "media":
        return df.fillna(df.mean())
    elif metodo == "mediana":
        return df.fillna(df.median())
    elif metodo == "moda":
        # Verificar si el DataFrame tiene datos
        if not df.empty:
            return df.fillna(df.mode().iloc[0])
        else:
            print("El DataFrame está vacío.")
            return df
    else:
        return "Método no soportado"

# Visualisar con grafica
def visualizar_histograma(df, columna, hits):
    df[columna].hist(bins=hits)
    plt.title(f"Histograma de {columna}")
    plt.xlabel(columna)
    plt.ylabel("Frecuencia")
    plt.show()


# Calcular correlacion
def calcular_correlacion(df):
    return df.corr()

# visualizar datos
def visualizar_datos(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'Gráfico de {y} vs {x}')
    plt.show()

#vizualizar todos los datos
def visualizar_todo(df):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Filtrar columnas numéricas
    columnas_numericas = df.select_dtypes(include='number')
    
    if len(columnas_numericas.columns) < 2:
        return "Se requieren al menos dos columnas numéricas para visualizar."

    # Crear pairplot
    sns.pairplot(columnas_numericas)
    plt.suptitle('Gráficos de Dispersión para Todas las Combinaciones', y=1.02)
    plt.show()

# Eliminar columnas no utilizadas
def eliminar_columnas(df, columnas):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Verificar si las columnas existen
    columnas_existentes = [col for col in columnas if col in df.columns]
    
    if len(columnas_existentes) == 0:
        return "Ninguna de las columnas especificadas existe en el DataFrame."
    
    df_limpio = df.drop(columns=columnas_existentes)
    print(f"Se eliminaron las siguientes columnas: {columnas_existentes}")
    
    return df_limpio

#orednar columnas
def ordenar_columnas(df, orden):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Asegurarse de que 'orden' es una lista
    if not isinstance(orden, list):
        return "El argumento 'orden' debe ser una lista de nombres de columnas."
    
    # Filtrar columnas existentes
    columnas_existentes = [col for col in orden if col in df.columns]
    
    if not columnas_existentes:
        return "No se encontraron columnas para ordenar."
    
    # Reordenar DataFrame
    df_resultado = df[columnas_existentes]
    return df_resultado


def renombrar_columnas(df, diccionario):
    if not isinstance(df, pd.DataFrame):
        return "El argumento no es un DataFrame válido."
    
    # Asegurarse de que es un diccionario
    if not isinstance(diccionario, dict):
        return "El argumento 'diccionario' debe ser del tipo correcto."

    # Traducir las columnas
    nuevas_columnas = {col: diccionario[col] for col in df.columns if col in diccionario}
    
    # Renombrar las columnas
    df_resultado = df.rename(columns=nuevas_columnas)
    return df_resultado

#Conectar sqlite
def guardar_en_sql(archivo_csv, nombre_tabla, db_name):
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(archivo_csv)  # Cargar el CSV automáticamente dentro de la función
    
    # Conectarse o crear la base de datos
    conn = sqlite3.connect(db_name)
    
    # Guardar el DataFrame como una tabla en la base de datos
    df.to_sql(nombre_tabla, conn, if_exists='replace', index=False)
    
    # Cerrar la conexión
    conn.close()
    
    return f"Datos guardados en la tabla '{nombre_tabla}' de la base de datos '{db_name}'"
