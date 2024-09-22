from setuptools import setup, find_packages

setup(
    name='datavm',
    version='0.1',
    description='Una librería para análisis y manipulación de datos',
    author='Víctor Maldoando',
    author_email='vico.luis.ads@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'sqlite3'
    ],
)
