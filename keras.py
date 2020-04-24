import tensorflow as tf
from sklearn.datasets import load_wine

vino = load_wine()
caracteristicas = vino['data']
objetivo = vino['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(caracteristicas, objetivo, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler()
x_train_normalizado = normalizador.fit_transform(x_train)
x_test_normalizado = normalizador.transform(x_test)

from tensorflow.contrib.keras import models, layers, losses, optimizers, metrics, activations
modelo = models.Sequential()
modelo.add(layers.Dense(units=13, input_dim=13, activation='relu'))
#capas intermedias
modelo.add(layers.Dense(units=13, activation='relu'))
modelo.add(layers.Dense(units=13, activation='relu'))
#capa de salidas
modelo.add(layers.Dense(units=3, activation='softmax'))
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo.fit(x_train_normalizado, y_train, epochs=60)

predicciones = modelo.predict_classes(x_test_normalizado)
from sklearn.metrics import classification_report
print(classification_report(y_test,predicciones))
