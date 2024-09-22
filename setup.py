from setuptools import setup, find_packages

setup(
    name='datavm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    'pandas>=1.0',
    'numpy<2.0',  # Restringe a versiones 1.x
    'matplotlib>=3.0',
    'seaborn>=0.11',
    'scikit-learn>=0.24'
     ],

    python_requires='>=3.12',
    author='Víctor M.',
    author_email='vico.luis.ads@gmail.com',
    description='Una librería para análisis y manipulación de datos',
    long_description=open('README.md').read(),  # Asegúrate de tener este archivo
    long_description_content_type='text/markdown',
    url='https://github.com/VicCodeM/datavm',  # URL del repositorio
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
