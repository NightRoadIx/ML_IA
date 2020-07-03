'''
 Clasificación de textos
'''
# Libreria Tensor Flow
import tensorflow as tf
# Keras
from tensorflow import keras
# De nuevo obtener los datasets
import tensorflow_datasets as tfds
# Quitar las barras de progreso cuando se descarguen y arreglen los datos
tfds.disable_progress_bar()
# Numpy
import numpy as np

# Ver la versión de TF
print(tf.__version__)

(train_data, test_data), info = tfds.load(
    # Utilizar la versión pre-codificada con un ~8k de vocabulario.
    'imdb_reviews/subwords8k', 
    # Regresar los datos de entrenamiento/prueba como una tupla
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Regresar los pares (example, label) de los datos (en lugar de un diccionario).
    as_supervised=True,
    # Regresar la estructura info 
    with_info=True)

train_data

test_data

info

# Probar el encoder
encoder = info.features['text'].encoder
print ('Tamaño del vocabulario: {}'.format(encoder.vocab_size))

# Ver la información que se iene del texto
info.features['text']

# Y ver las palabras que se usan para el codificador
encoder.subwords

# Crear una cadena de prueba
sample_string = 'Hooolis TensorFlow.'

# Codificar
encoded_string = encoder.encode(sample_string)
print ('Cadena codificada es {}'.format(encoded_string))

# Decodificar
original_string = encoder.decode(encoded_string)
print ('La cadena original: "{}"'.format(original_string))

# Lo que hace assert es que si la condición resulta Falsa, se lanza un error AsertionError
# en otro caso no pasa nada
assert original_string == sample_string

# Ver la forma en que se está codificando la cadena
for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

# Ahora va con los datos de entrenamiento
for train_example, train_label in train_data.take(1): # cambiar el índice para ver más elementos
  print('Texto codificado: ', train_example[:10].numpy())
  print('Etiqueta:', train_label.numpy())
  print('Decodificado: ')
  print(encoder.decode(train_example))

# Preparar los datos para el entrenamiento
BUFFER_SIZE = 1000

# Dado que las revisiones son de diferentes longitudes, se usa padded_batch 
# para rellenar con ceros las secuencias y que todas sean de la misma longitud
train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32))

test_batches = (
    test_data
    .padded_batch(32))

# Cada patch tendrá una longitud diferente
for example_batch, label_batch in train_batches.take(4):
  print("Perfil del batch:", example_batch.shape)
  print("Perfil de la etiqueta:", label_batch.shape)

# CONSTRUIR EL MODELO
# El modelo es secuencial
# La primera capa es una del tipo Embedding, muy utilizada para datos de tipo texto
# Requiere que los datos de entrada esten codificados a entero
# que como ya se vio cada dato esta representado por un entero único
# Entrada el tamaño del vocabulario (aprenderá de todos los datos del vocabulario)
# Se especifica también el tamaño de la salida
# Aquí hay datos interesantes acerca de esto:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# Posteriormente, se hace un "pooling" para convertir el espacio vectorial
# en uno de dimensión 1D (un solo dato)
# para aplicar a la salida de "positivo" / "negativo"
# esta última capa tiene como FA una lineal por omisión, aunque podría usarse
# una sigmoide
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1)])

model.summary()

# Compilación del modelo, añadiendo:
# Optimizador: Adam
# Función pérdida: Entropía Cruzada Binaria
# Métrica: Precisión
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ENTRENAR EL MODELO
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

# EVALUAR EL MODELO
loss, accuracy = model.evaluate(test_batches)

print("Pérdida: ", loss)
print("Precisión: ", accuracy)

history_dict = history.history
history_dict.keys()

# Graficar la precisión y pérdida
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'ro', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'ro', label='Precisión de entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.grid()

plt.show()

