import pandas as pd
#
#
#importante poner la direccion donde se encuentra el archivo este archivo esta en la carpeta resources
dataframe=pd.read_csv('original.csv')
dataframe
dataframe.describe()
dataframe['SALARIO']>30000
