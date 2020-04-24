import tensorflow as tf
from tensorflow import estimator
from sklearn.datasets import load_wine

vino = load_wine()
caracteristicas = vino['data']
objetivo = vino['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(caracteristicas, objetivo, test_size=0.3)
from sklearn.preprocessing import  MinMaxScaler
normalizador = MinMaxScaler()

x_train_normalizado = normalizador.fit_transform(x_train)
x_test_normalizado = normalizador.transform(x_test)

columnas_caracteristicas = [tf.feature_column.numeric_column('x',shape=[13])]
modelo = estimator.DNNClassifier(hidden_units=[20,20,20], feature_columns = columnas_caracteristicas, n_classes=3, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01))

funcion_entrada = estimator.inputs.numpy_input_fn(x={'x':x_train_normalizado}, y=y_train, shuffle=True, batch_size=10, num_epochs=10)

modelo.train(input_fn=funcion_entrada, steps=600)
funcion_evaluacion = estimator.inputs.numpy_input_fn(x={'x':x_test_normalizado}, shuffle=False)
predicciones = list(modelo.predict(input_fn=funcion_evaluacion))
predicciones_finales = [ p['class_ids'][0] for p in predicciones]

predicciones_finales

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predicciones_finales))
