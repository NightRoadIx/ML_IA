'''
 Lectura de información utilizando Pandas

 Para realizar el entrenamiento de las redes neuronales para Machine Learning
 se utilizan una gran cantidad de datos para asegurar que los modelos de ML
 aseguren tener una adecuada convergencia y adaptación a una diversidad de 
 situaciones.
 En la actulidad, la transferencia de grandes cantidades de información se 
 realiza no solo por teclado o dispositivos de entrada clásicos, si no que 
 muchas veces se ocupan dispositivos de almacenamiento masivo (HD, SHD, USBM)
 e incluso los datos almacenados en la "nube", todo ello localizado en archivos.
 Uno de los formatos más utilizados para este fin en el CSV (Comma Separated
 Values, valores separados por coma), archivos de texto plano que usan una 
 estructura específica para organizar datos tabulares.
 Python cuenta con varias liberías para realizar este proceso, entre las que 
 se encuentra Pandas, una de las más utilizadas para este fin
'''
import pandas as pd
import tensorflow as tf

# Descargar el archivo CSV con los datos
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

# Leer el CSV usando pandas
df = pd.read_csv(csv_file)

# Una muestra de los datos
df.head()

# Ver los tipos de los datos
df.dtypes

# Lo siguiente convierte la columna identificada por el encabezado como "thal"
# que es un objeto, a un valor numérico discreto
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

df.head()

# Ahora, se utilizará tf.data.Dataset para leer valores de los datos
# lo interesante de utilizar esto es que se pueden crear líneas de datos altamente eficientes

# esto obtiene los valores etiquetados con "target"
target = df.pop('target')

target.head(10)

# Del set de datos que se tiene, obtener tensores
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

dataset.range

# Recorrer los valores obtenidos (tomar solo 5) para observarlos
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

# Ya que una pd.Series implementa __array__, puede ser utilizado donde use un np.arry o ft.Tensor
tf.constant(df['thal'])

# Entonces se desordena y se obtienen "batch" (como subgrupos)
# Un batch es un término utilizado en ML y se refiere al número de ejemplos 
# de entrenamiento utilizados en una sola iteración
# Estos son utilizados para controlar el número de muestras de entrenamiento
# que se deben trabajar antes de que se actualicen los parámetros del modelo
# (No confundir con épocas, el cual es el número de pases completos sobre TODO
# el set de datos)
# Más información en:
# https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
train_dataset = dataset.shuffle(len(df)).batch(1)

train_dataset.element_spec

for k, z in train_dataset.take(5):
  print('Features: {}, Target: {}'.format(k, z))

# Crear y entrenar un modelo
# Se genera una función que crea el modelo y lo compila
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)

# Sin embargo esto se puede hacer por medio de diccionarios
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
  print (dict_slice)

# y esto mismo se puede ingresar al modelo de ML
model_func.fit(dict_slices, epochs=15)

