import pandas as pd
import tensorflow as td

#deben agregar la ruta del archivo que se encuentra en la carpeta resources

ingresos = pd.read_csv('C:/ingresos.csv')
from sklearn.model_selection import train_test_split
datos_x = ingresos.drop('income', axis=1)
datos_y = ingresos('income')
x_train, x_test, y_train, y_test = train_test_split(datos_x, datos_y, test_size=0.3)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female,Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
age =tf.feature_column.numeric_column("age")
fnlwgt =tf.feature_column.numeric_column("fnlwgt")
educational_num =tf.feature_column.numeric_column("educational-num")
capital_gain =tf.feature_column.numeric_column("capital-gain")
capital_loss =tf.feature_column.numeric_column("capital-loss")
hours_per_week =tf.feature_column.numeric_column("hours-per-week")

columnas_cat =[gender, occupation, marital_status, relationship, education, native_country, workclass, age, fnlwgt, educational_num, capital_gain, capital_loss, hours_per_week]
fun_in = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100,num_epochs=None, shuffle=True)
model =tf.estimator.LinearClassifier(feature_columns=columnas_cat)
model.train(input_fn=fun_in, steps=8000)
funcion_pred = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), shuffle=False)
predicciones = model.predict(input_fn=funcion_pred)
list_pred = list(predicciones)
list_pred

final_pred =[prediccion['class_ids'][0] for prediccion in list_pred]
final_pred
