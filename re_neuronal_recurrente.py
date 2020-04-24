import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#deben agregar la ruta del archivo que se encuentra en la carpeta resources
leche  = pd.read_csv('produccion-leche.csv',index_col='Month')
leche.index = pd.to_datetime(leche.index)
leche.plot()

conjunto_entrenamiento = leche.head(150)
conjunto_pruebas = leche.tail(18)

from sklearn.preprocessing import MinMaxScaler
normalizacion = MinMaxScaler()
entrenamiento_normalizado = normalizacion.fit_transform(conjunto_entrenamiento)
pruebas_normalizado = normalizacion.transform(conjunto_pruebas)

def lotes(datos_entrenamiento, tamaño_lote, pasos):
    comienzo = np.random.randint(0, len(datos_entrenamiento)-pasos)
    lote_y = np.array(datos_entrenamiento[comienzo:comienzo+pasos+1]).reshape(1,pasos+1)
    return lote_y[:,:-1].reshape(-1,pasos,1), lote_y[:,1:].reshape(-1,pasos,1)

numero_entradas =1
numero_pasos = 18
numero_neuronas = 120
numero_salidas = 1
tasa_aprendizaje = 0.001
numero_iteraciones = 5000
tamaño_lote = 1

x = tf.placeholder(tf.float32, [None, numero_pasos, numero_entradas])
y = tf.placeholder(tf.float32, [None, numero_pasos, numero_salidas])
capa = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=numero_neuronas, activation=tf.nn.relu), output_size=numero_salidas)
salidas, estados = tf.nn.dynamic_rnn(capa, x, dtype=tf.float32)

funcion_error= tf.reduce_mean(tf.square(salidas-y))
optimizador = tf.train.AdamOptimizer(learning_rate=tasa_aprendizaje)
entrenamiento = optimizador.minimize(funcion_error)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sesion:
    sesion.run(init)
    for iteracion in range(numero_iteraciones):
        lote_x, lote_y = lotes(entrenamiento_normalizado,tamaño_lote, numero_pasos)
        sesion.run(entrenamiento, feed_dict={x:lote_x, y:lote_y})
        if iteracion %100 == 0:
            error = funcion_error.eval(feed_dict={x:lote_x, y:lote_y})
            print(iteracion,"\t Error", error)

        saver.save(sesion,"./modelo_series_temporales2")

with tf.Session() as sesion:
    saver.restore(sesion, "./modelo_series_temporales2")
    entrenamiento_seed = list(entrenamiento_normalizado[-18:])
    for iteracion in range(18):
        lote_x = np.array(entrenamiento_seed[-numero_pasos:]).reshape(1,numero_pasos,1)
        prediccion_y = sesion.run(salidas, feed_dict={x:lote_x})
        entrenamiento_seed.append(prediccion_y[0,-1,0])

resultados = normalizacion.inverse_transform(np.array(entrenamiento_seed[18:]).reshape(18,1))

conjunto_pruebas['predicciones']= resultados
conjunto_pruebas

conjunto_pruebas.plot()
