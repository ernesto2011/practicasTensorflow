import tensorflow as tf

mensaje1 =tf.constant("hola")
mensaje2 =tf.constant("mundo")

print(mensaje1)
type(mensaje1)

with tf.Session() as sesion:
    resultado = sesion.run(mensaje1 +  mensaje2)

a = tf.constant(10)
b = tf.constant(5)

a + b

with tf.Session() as sesion:
    resultado = sesion.run(a + b)

resultado

constante = tf.constant(15)
matriz = tf.fill((6,6),10)
matriz2 = tf.random_normal((5,5))
matriz3 = tf.random_uniform((4,4),minval=0,maxval=5)
matriz_ceros = tf.zeros((2,2))
matriz_unos = tf.ones((3,3))
operaciones = [constante, matriz, matriz2, matriz3, matriz_ceros, matriz_unos]
with tf.Session() as sesion:
    for op in operaciones:
        print(sesion.run(op))
        print("\n")
