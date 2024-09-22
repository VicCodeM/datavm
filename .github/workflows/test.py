import datavm as vm
import pandas as pd

# Cargar datos
datos = vm.cargar_datos("movies.csv", tipo='csv')
df = pd.DataFrame(datos)

# Verificar nulos
nulos = vm.verificar_nulos(df)
print("Nulos en el DataFrame:")
print(nulos)

# Eliminar nulos
df_limpio = vm.eliminar_nulos(df)
print("\nDataFrame después de eliminar nulos:")
print(df_limpio)

# Eliminar columnas específicas
df_limpio1 = vm.eliminar_columnas(df_limpio, ['poster_path', 'backdrop_path', 'recommendations'])
print("\nDataFrame después de eliminar columnas:")
print(df_limpio)

# Guardar nuevo DataFrame
vm.guardar_datos(df_limpio1, ruta_archivo='movies5.csv')
print("\nDatos guardados en 'movies5.csv'")

# Ejemplo de uso
resultado = vm.guardar_en_sql('movies5.csv', 'movies', 'mi_base_datos.sqlite')
print(resultado)
