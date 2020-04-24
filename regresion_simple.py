import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
datos_x = np.linspace(0,10,10) + np.random.uniform(-1,1,10)
datos_y = np.linspace(0,10,10) + np.random.uniform(-1,1,10)

plt.plot(datos_x, datos_y, '*')
# estas variables se incializan con estos valores pero se pueden reemplazar
m = tf.Variable(0.81)
b = tf.Variable(0.87)

error = 0
for x,y in zip(datos_x,datos_y):
    y_pred = m*x + b
    error = error + (y - y_pred)**2

optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.001)
entrenamiento = optimizador.minimize(error)

ini =tf.global_variables_initializer()

with tf.Session() as sesion:
    sesion.run(ini)
    pasos = 1
    for i in range(pasos):
        sesion.run(entrenamiento)
    final_m, final_b = sesion.run([m,b])

x_test = np.linspace(-1,11,10)
y_pred_2 = (final_m * x_test) +final_b

plt.plot(x_test, y_pred_2,'r')
plt.plot(datos_x, datos_y,'+')
