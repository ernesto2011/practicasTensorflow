import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

datos=np.random.randint(0,100,(100,4))
datos

dataframe= pd.DataFrame(data=datos, columns=['c1','c2','c3','etiqueta'])
dataframe
x=dataframe[['c1','c2','c3']]
x

y = dataframe['etiqueta']
y

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
