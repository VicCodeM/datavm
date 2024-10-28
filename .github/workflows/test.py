import unittest
import pandas as pd
import numpy as np
from mi_modulo import (
    cargar_datos,
    guardar_datos,
    verificar_nulos,
    eliminar_nulos,
    rellenar_nulos,
    verificar_duplicados,
    eliminar_duplicados,
    normalizar_datos,
    visualizar_histograma,
    calcular_correlacion,
    eliminar_columnas,
    ordenar_columnas,
    renombrar_columnas,
    guardar_en_sql
)

class TestMiModulo(unittest.TestCase):

    def setUp(self):
        # Configurar datos de prueba
        self.df = pd.DataFrame({
            'Nombre': ['Ana', 'Luis', 'Carlos', 'Ana'],
            'Edad': [28, None, 22, None],
            'Salario': [50000, 60000, 70000, None]
        })

    def test_cargar_datos(self):
        # Puedes agregar pruebas para cargar datos si tienes archivos de prueba
        pass

    def test_guardar_datos(self):
        # Prueba guardar datos en CSV
        guardar_datos(self.df, 'test_guardar.csv', tipo='csv')
        loaded_df = pd.read_csv('test_guardar.csv')
        pd.testing.assert_frame_equal(loaded_df, self.df)

    def test_verificar_nulos(self):
        self.assertEqual(verificar_nulos(self.df).isnull().sum().sum(), 2)

    def test_eliminar_nulos(self):
        df_limpio = eliminar_nulos(self.df)
        self.assertEqual(df_limpio.isnull().sum().sum(), 0)

    def test_rellenar_nulos(self):
        df_rellenado = rellenar_nulos(self.df)
        self.assertNotIn(None, df_rellenado['Edad'].values)

    def test_verificar_duplicados(self):
        duplicados = verificar_duplicados(self.df)
        self.assertEqual(len(duplicados), 2)  # Debe encontrar 2 filas duplicadas

    def test_eliminar_duplicados(self):
        df_sin_duplicados = eliminar_duplicados(self.df)
        self.assertEqual(df_sin_duplicados.shape[0], 3)  # Debería quedar solo 3 filas

    def test_normalizar_datos(self):
        df_normalizado = normalizar_datos(self.df, 'media')
        self.assertFalse(df_normalizado.isnull().values.any())

    def test_eliminar_columnas(self):
        df_sin_salario = eliminar_columnas(self.df, ['Salario'])
        self.assertNotIn('Salario', df_sin_salario.columns)

    def test_ordenar_columnas(self):
        df_ordenado = ordenar_columnas(self.df, ['Nombre', 'Edad'])
        self.assertListEqual(list(df_ordenado.columns), ['Nombre', 'Edad'])

    def test_renombrar_columnas(self):
        diccionario = {
            'Nombre': 'Name',
            'Edad': 'Age',
            'Salario': 'Salary'
        }
        df_renombrado = renombrar_columnas(self.df, diccionario)
        self.assertIn('Name', df_renombrado.columns)

    def test_guardar_en_sql(self):
        resultado = guardar_en_sql('test_guardar.csv', 'test_table', 'test_db.sqlite')
        self.assertIn("Datos guardados en la tabla", resultado)

if __name__ == '__main__':
    unittest.main()
