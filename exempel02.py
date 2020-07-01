'''
Este ejemplo clasificará revisiones de películas como positivas o negativas
utilizando texto de la BD IMDB
El conjunto de datos contiene 50,000 revisiones de películas y se dividirán
en 25,000 para entrenamiento y 25,000 para prueba
Las revisiones se encuentran balanceadas, esto quiere decir que la mitad
de ellas son positivas y la mitad negativas
'''
import numpy as np

import tensorflow as tf

# Debe instalarse
# pip install -q tensorflow-hub
# pip install -q tfds-nightly para datasets actualizados al día
# pip install -q tensorflow-datasets para datasets actualizados cada mes
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Ver las características del TensorFlow y si es posible hacer procesamiento con la GPU
# Versión del TF
print("Versión TF: ", tf.__version__)
# Modo "ansioso" (Eager mode), el cual es una librería tipo Numpy para computación numérica
# que permite el soporte con la GPU, así como diferenciación automática
print("Modo \"ansioso\": ", tf.executing_eagerly())
# El Hub de TF es una librería y plataforma para transferencia de aprendizaje
# soportada por Google
print("Versión Hub: ", hub.__version__)
print("La GPU está", "disponible" if tf.config.experimental.list_physical_devices("GPU") else "NO DISPONIBLE")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DESCARGA DE LOS DATOS
# Separar los datos en una proporción 60-40%, ya que se trata de un total
# de 25,000 ejemplos, y así dejar 15,000 para entrenamiento y 10,000 para validación
# mas un total de 25,000 ejemplos adicionales para pruebas
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_data.element_spec
train_data.output_shapes
train_data.output_types

# Imprimir los primeros 10 ejemplos
# Utilizar iteradores para obtener los primeros 10 valores
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

print(train_examples_batch.shape)
print(train_labels_batch.shape)

# Ver los primeros 10 ejemplos
train_examples_batch

# Ahora ver las primeras 10 etiquetas
train_labels_batch
# Los 1, 0 representan los comentarios positivos y negativos de las reseñas

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Aquí se utilizará un modelo embebido de texto pre-entrenado del Hub de TensorFlow
# Ya que, una forma de representar el texto es convertir las oraciones en vectores embebidos
# el cual permite olvidarse del preprocesamiento del texto

# Un vector embebido es una técnica de modelado de lenguaje utilizada para mapear 
# palabras a vectores de números reales. Representan palabras o frases en un 
# espacio vectorial de varias dimensiones.

# Es muy utilizada para reconocimiento del habla, bots de respuestas y de información
# aplicaciones de procesamiento de lenguaje natural (NLP)

# Más info:
# https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795

# Lo que se hace es crear una capa de la red neuronal (utilizando Keras)
# Observando que no importa la longitud del texto, el tipo de salida
# siempre se ajusta a: (num_ejemplos, dimension_incrustada)
# el vector embebido tiene un tamaño fijo, por lo que resulta más simple de procesar
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# Hay otros modelos embebidos:
# https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1 -> Más grande con 1M de vocabulario y 50 dimensiones
# https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1 -> Más grande con 1M de vocabulario y 128 dimensiones*
# Todos localizados en el TFHub de Google
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
'''
# Ejemplo de como está funcionando
embiid = hub.load(embedding)
embiids = embiid(["cat is on the green mat", "dog is in the heavy fog"])

print(embiids)
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Construir el modelo
# tipo secuencial
model = tf.keras.Sequential()
# Añadir la capa hub_layer, la cual se adaptará al número de entradas ejemplo
model.add(hub_layer)
# Una capa densa de 16 nodos, FA relu
model.add(tf.keras.layers.Dense(16, activation='relu'))
# La capa de salida que como las etiquetas son "positiva" ó "negativa" es de un solo nodo
model.add(tf.keras.layers.Dense(1))

model.summary()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Compilar el modelo con optimizador "Adam"
# función de perdida entropía cruzada binaria (por el tipo de dato de salida de la red)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# ENTRENAR
# Se ingresarán 10,000 datos de manera aleatoria con un batch de 512
# a entrenar durante 20 épocas
# Usando el mismo batch de los datos de validación
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# EVALUAR EL MODELO
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

print(history.history.keys())

import matplotlib.pyplot as plt

acc2 = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20) # generar un vector de las epocas de entrenamiento de la red

plt.figure()
plt.plot(epochs_range, acc2, label='Precisión de entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de validación')
plt.legend(loc='center right')
plt.title('Precisión de entrenamiento y validación')
plt.grid()
plt.show()

plt.figure()
plt.plot(epochs_range, loss, label='Pérdida de entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de validación')
plt.legend(loc='center right')
plt.title('Pérdida de entrenamiento y validación')
plt.grid()
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# PREDICCIONES
predecir = model.predict(test_data.batch(512))
predecir2 = np.rint(predecir)

print(predecir2)

