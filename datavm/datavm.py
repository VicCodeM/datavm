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

def visualizar_boxplot(df, columna):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[columna])
    plt.title(f"Boxplot de {columna}")
    plt.show()

def visualizar_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Mapa de Calor de Correlación")
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



#Manejo de Outliers
def detectar_outliers(df, columna, metodo="zscore", umbral=3):
    if metodo == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[columna]))
        return df[z_scores > umbral]
    elif metodo == "iqr":
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[columna] < (Q1 - 1.5 * IQR)) | (df[columna] > (Q3 + 1.5 * IQR))]
    else:
        return "Método no soportado"

def eliminar_outliers(df, columna, metodo="zscore", umbral=3):
    if metodo == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[columna]))
        return df[z_scores <= umbral]
    elif metodo == "iqr":
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[columna] >= (Q1 - 1.5 * IQR)) & (df[columna] <= (Q3 + 1.5 * IQR))]
    else:
        return "Método no soportado"
    
#Codificación de Variables Categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def codificar_categoricas(df, columnas, metodo="onehot"):
    if metodo == "onehot":
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = encoder.fit_transform(df[columnas])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columnas))
        return pd.concat([df.drop(columnas, axis=1), encoded_df], axis=1)
    elif metodo == "label":
        encoder = LabelEncoder()
        for col in columnas:
            df[col] = encoder.fit_transform(df[col])
        return df
    else:
        return "Método no soportado"
    
#escaldo de datos 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def escalar_datos(df, columnas, metodo="standard"):
    if metodo == "standard":
        scaler = StandardScaler()
    elif metodo == "minmax":
        scaler = MinMaxScaler()
    else:
        return "Método no soportado"
    
    df[columnas] = scaler.fit_transform(df[columnas])
    return df    

# Selección de Características
# La selección de características puede mejorar el rendimiento de los modelos y reducir el sobreajuste.
from sklearn.feature_selection import SelectKBest, f_classif

def seleccionar_caracteristicas(df, target, k=10):
    X = df.drop(target, axis=1)
    y = df[target]
    selector = SelectKBest(score_func=f_classif, k=k)
    fit = selector.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Caracteristica', 'Puntuacion']
    return feature_scores.nlargest(k, 'Puntuacion')

# Transformación de Datos
# Transformaciones como la logarítmica, exponencial, o Box-Cox pueden ser útiles para normalizar distribuciones sesgadas.

from scipy.stats import boxcox

def transformar_datos(df, columna, metodo="log"):
    if metodo == "log":
        df[columna] = np.log1p(df[columna])
    elif metodo == "boxcox":
        df[columna], _ = boxcox(df[columna])
    else:
        return "Método no soportado"
    return df

# Validación Cruzada
# La validación cruzada es una técnica importante para evaluar el rendimiento de los modelos.
from sklearn.model_selection import cross_val_score

def validar_modelo(modelo, X, y, cv=5):
    scores = cross_val_score(modelo, X, y, cv=cv)
    print(f"Puntuaciones de Validación Cruzada: {scores}")
    print(f"Precisión media: {scores.mean()}")
    return scores

# Manejo de Datos Desbalanceados
# En problemas de clasificación, los datos desbalanceados pueden afectar el rendimiento del modelo.
from imblearn.over_sampling import SMOTE

def balancear_datos(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

#Guardados de modelos
import joblib

def guardar_modelo(modelo, nombre_archivo):
    joblib.dump(modelo, nombre_archivo)
    print(f"Modelo guardado como {nombre_archivo}")

def cargar_modelo(nombre_archivo):
    modelo = joblib.load(nombre_archivo)
    print(f"Modelo cargado desde {nombre_archivo}")
    return modelo


import mysql.connector

# Conectar a MySQL
def conectar_mysql(host, user, password, database):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return conn

# Cargar datos desde una tabla MySQL a un DataFrame
def cargar_desde_mysql(conn, nombre_tabla):
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    return df

# Insertar datos en una tabla MySQL
def insertar_en_mysql(conn, df, nombre_tabla):
    cursor = conn.cursor()
    cols = ", ".join([str(i) for i in df.columns.tolist()])
    for i, row in df.iterrows():
        sql = f"INSERT INTO {nombre_tabla} ({cols}) VALUES ({', '.join(['%s']*len(row))})"
        cursor.execute(sql, tuple(row))
    conn.commit()
    print(f"Datos insertados en la tabla '{nombre_tabla}'")

# Actualizar registros en una tabla MySQL
def actualizar_en_mysql(conn, nombre_tabla, set_clause, where_clause):
    cursor = conn.cursor()
    query = f"UPDATE {nombre_tabla} SET {set_clause} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros actualizados en la tabla '{nombre_tabla}'")

# Eliminar registros de una tabla MySQL
def eliminar_en_mysql(conn, nombre_tabla, where_clause):
    cursor = conn.cursor()
    query = f"DELETE FROM {nombre_tabla} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros eliminados de la tabla '{nombre_tabla}'")

# Ejecutar consultas SQL personalizadas en MySQL
def ejecutar_consulta_mysql(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    resultados = cursor.fetchall()
    return resultados

# Cerrar conexión MySQL
def cerrar_conexion_mysql(conn):
    conn.close()
    print("Conexión MySQL cerrada")

#Conectar sqlite
def guardar_en_sqlite(archivo_csv, nombre_tabla, db_name):
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(archivo_csv)  # Cargar el CSV automáticamente dentro de la función
    
    # Conectarse o crear la base de datos
    conn = sqlite3.connect(db_name)
    
    # Guardar el DataFrame como una tabla en la base de datos
    df.to_sql(nombre_tabla, conn, if_exists='replace', index=False)
    
    # Cerrar la conexión
    conn.close()
    
    return f"Datos guardados en la tabla '{nombre_tabla}' de la base de datos '{db_name}'"


# Cargar datos desde una tabla SQLite a un DataFrame
def cargar_desde_sqlite(db_name, nombre_tabla):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Insertar datos en una tabla SQLite
def insertar_en_sqlite(df, db_name, nombre_tabla):
    conn = sqlite3.connect(db_name)
    df.to_sql(nombre_tabla, conn, if_exists='append', index=False)
    conn.close()
    print(f"Datos insertados en la tabla '{nombre_tabla}' de la base de datos '{db_name}'")

# Actualizar registros en una tabla SQLite
def actualizar_en_sqlite(db_name, nombre_tabla, set_clause, where_clause):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    query = f"UPDATE {nombre_tabla} SET {set_clause} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    conn.close()
    print(f"Registros actualizados en la tabla '{nombre_tabla}' de la base de datos '{db_name}'")

# Eliminar registros de una tabla SQLite
def eliminar_en_sqlite(db_name, nombre_tabla, where_clause):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    query = f"DELETE FROM {nombre_tabla} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    conn.close()
    print(f"Registros eliminados de la tabla '{nombre_tabla}' de la base de datos '{db_name}'")

# Ejecutar consultas SQL personalizadas
def ejecutar_consulta_sqlite(db_name, query):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(query)
    resultados = cursor.fetchall()
    conn.close()
    return resultados

import pyodbc

# Conectar a SQL Server
def conectar_sql_server(server, database, username, password):
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(conn_str)
    return conn

# Cargar datos desde una tabla SQL Server a un DataFrame
def cargar_desde_sql_server(conn, nombre_tabla):
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    return df

# Insertar datos en una tabla SQL Server
def insertar_en_sql_server(conn, df, nombre_tabla):
    cursor = conn.cursor()
    cols = ", ".join([str(i) for i in df.columns.tolist()])
    for i, row in df.iterrows():
        sql = f"INSERT INTO {nombre_tabla} ({cols}) VALUES ({', '.join(['?']*len(row))})"
        cursor.execute(sql, tuple(row))
    conn.commit()
    print(f"Datos insertados en la tabla '{nombre_tabla}'")

# Actualizar registros en una tabla SQL Server
def actualizar_en_sql_server(conn, nombre_tabla, set_clause, where_clause):
    cursor = conn.cursor()
    query = f"UPDATE {nombre_tabla} SET {set_clause} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros actualizados en la tabla '{nombre_tabla}'")

# Eliminar registros de una tabla SQL Server
def eliminar_en_sql_server(conn, nombre_tabla, where_clause):
    cursor = conn.cursor()
    query = f"DELETE FROM {nombre_tabla} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros eliminados de la tabla '{nombre_tabla}'")

# Ejecutar consultas SQL personalizadas en SQL Server
def ejecutar_consulta_sql_server(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    resultados = cursor.fetchall()
    return resultados

# Cerrar conexión SQL Server
def cerrar_conexion_sql_server(conn):
    conn.close()
    print("Conexión SQL Server cerrada")


import psycopg2


# Conectar a PostgreSQL
def conectar_postgresql(host, user, password, database):
    conn = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return conn

# Cargar datos desde una tabla PostgreSQL a un DataFrame
def cargar_desde_postgresql(conn, nombre_tabla):
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    return df

# Insertar datos en una tabla PostgreSQL
def insertar_en_postgresql(conn, df, nombre_tabla):
    cursor = conn.cursor()
    cols = ", ".join([str(i) for i in df.columns.tolist()])
    for i, row in df.iterrows():
        sql = f"INSERT INTO {nombre_tabla} ({cols}) VALUES ({', '.join(['%s']*len(row))})"
        cursor.execute(sql, tuple(row))
    conn.commit()
    print(f"Datos insertados en la tabla '{nombre_tabla}'")

# Actualizar registros en una tabla PostgreSQL
def actualizar_en_postgresql(conn, nombre_tabla, set_clause, where_clause):
    cursor = conn.cursor()
    query = f"UPDATE {nombre_tabla} SET {set_clause} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros actualizados en la tabla '{nombre_tabla}'")

# Eliminar registros de una tabla PostgreSQL
def eliminar_en_postgresql(conn, nombre_tabla, where_clause):
    cursor = conn.cursor()
    query = f"DELETE FROM {nombre_tabla} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros eliminados de la tabla '{nombre_tabla}'")

# Ejecutar consultas SQL personalizadas en PostgreSQL
def ejecutar_consulta_postgresql(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    resultados = cursor.fetchall()
    return resultados

# Cerrar conexión PostgreSQL
def cerrar_conexion_postgresql(conn):
    conn.close()
    print("Conexión PostgreSQL cerrada")    


import cx_Oracle


# Conectar a Oracle
def conectar_oracle(username, password, dsn):
    conn = cx_Oracle.connect(username, password, dsn)
    return conn

# Cargar datos desde una tabla Oracle a un DataFrame
def cargar_desde_oracle(conn, nombre_tabla):
    query = f"SELECT * FROM {nombre_tabla}"
    df = pd.read_sql_query(query, conn)
    return df

# Insertar datos en una tabla Oracle
def insertar_en_oracle(conn, df, nombre_tabla):
    cursor = conn.cursor()
    cols = ", ".join([str(i) for i in df.columns.tolist()])
    for i, row in df.iterrows():
        sql = f"INSERT INTO {nombre_tabla} ({cols}) VALUES ({', '.join([':1']*len(row))})"
        cursor.execute(sql, tuple(row))
    conn.commit()
    print(f"Datos insertados en la tabla '{nombre_tabla}'")

# Actualizar registros en una tabla Oracle
def actualizar_en_oracle(conn, nombre_tabla, set_clause, where_clause):
    cursor = conn.cursor()
    query = f"UPDATE {nombre_tabla} SET {set_clause} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros actualizados en la tabla '{nombre_tabla}'")

# Eliminar registros de una tabla Oracle
def eliminar_en_oracle(conn, nombre_tabla, where_clause):
    cursor = conn.cursor()
    query = f"DELETE FROM {nombre_tabla} WHERE {where_clause}"
    cursor.execute(query)
    conn.commit()
    print(f"Registros eliminados de la tabla '{nombre_tabla}'")

# Ejecutar consultas SQL personalizadas en Oracle
def ejecutar_consulta_oracle(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    resultados = cursor.fetchall()
    return resultados

# Cerrar conexión Oracle
def cerrar_conexion_oracle(conn):
    conn.close()
    print("Conexión Oracle cerrada")