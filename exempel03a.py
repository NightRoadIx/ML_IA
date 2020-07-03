'''
  Ver como se codifican los textos
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
import os

# Dirección del cual se tomarán los datos
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
# Los nombres de los archivos
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
# Puede observarse el texto disponible en línea con colocar la URL seguida por cualquiera de los archivos

# Obtener los datos utilizando Keras
for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

# Aquí se muestra el nombre del directorio de trabajo  
parent_dir = os.path.dirname(text_dir)
print(parent_dir)

# Cargar el texto en los datos (dataset)
# Esta función etiquetará el texto con su respectivo índice
def labeler(example, index):
  # Regresa el texto ejemplo y el índice como valor entero de 64 bits
  return example, tf.cast(index, tf.int64)  

# Generar una lista vacía
labeled_data_sets = []

# Recorrer los archivos
for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  # Aquí se mapea con la función labeler, para aplicar una etiqueta
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

labeled_dataset

labeled_data_sets

# Ahora se combinarán estos datasets en uno simple y se "revolverán"
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

labeled_data_sets[1:]

# Se toma entonces una copia de labeled_data_sets (pimer elemento)
all_labeled_data = labeled_data_sets[0]
# Recorrer todo el dataset restante y concatenarlo
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

# Una vez concatenado, revolver
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

# Imprimir 5 de los pares que se obtienen
for ex in all_labeled_data.take(5):
  print(ex)

# Ahora entonces, se codificarán las líneas de texto como números
# o en específico , cada palabra en un entero único

# Se aplica el construir un vocabulario, al tomar cada unidad de texto y convertirla
# en una colección de palabras únicas individuales
# Hay muchas formas de hacerlo, pero en este caso:

# Generar o dividir en "tokens" o símbolos
tokenizer = tfds.features.text.Tokenizer()

# Generar el set de vocabulario
vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  # Iterar sobre cada valor numpy del par (example, label)
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  # el set se actualiza con los valores "token" generados
  vocabulary_set.update(some_tokens)

# Ver entonces el tamaño del vocabulario
vocab_size = len(vocabulary_set)
print("Tamaño del vocabulario: ", vocab_size)

# Ver el set de vocabulario creado
for x in vocabulary_set:
  print(x, " ", end="")

# Crear la codificación al pasar el vocabulario a la función tfds.features.text.TokenTextEncoder
# Este método tomará las cadenas de texto y regresará una lista de enteros
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

encoder

# Una vez que se tiene el encoder, se puede aplicar al texto, por ejemplo
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

# Aplicando el encoder
encoded_example = encoder.encode(example_text)
print(encoded_example)

# Si se desea ver 5 ejemplos para comparar la acción del encoder
for ex in all_labeled_data.take(5):
  print(ex[0].numpy())
  print(encoder.encode(ex[0].numpy()))

# Ahora, se aplicará el encoder a todo el dataset al "envolverlo" en tf.py_function
# y pasarlo al método map del método

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]

vocab_size += 1

print(vocab_size)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
'''
  FIN DEL EJEMPLO
'''