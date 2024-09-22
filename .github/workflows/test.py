# test.py
import unittest
from mi_modulo import sumar, restar, multiplicar, dividir

class TestMiModulo(unittest.TestCase):

    def test_sumar(self):
        self.assertEqual(sumar(1, 2), 3)
        self.assertEqual(sumar(-1, 1), 0)
        self.assertEqual(sumar(-1, -1), -2)

    def test_restar(self):
        self.assertEqual(restar(5, 3), 2)
        self.assertEqual(restar(5, 5), 0)
        self.assertEqual(restar(3, 5), -2)

    def test_multiplicar(self):
        self.assertEqual(multiplicar(2, 3), 6)
        self.assertEqual(multiplicar(-1, 5), -5)
        self.assertEqual(multiplicar(0, 10), 0)

    def test_dividir(self):
        self.assertEqual(dividir(6, 3), 2)
        self.assertEqual(dividir(5, 2), 2.5)
        with self.assertRaises(ValueError):
            dividir(5, 0)

if __name__ == '__main__':
    unittest.main()
