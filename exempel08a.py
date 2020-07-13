import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# Obtener los datos del MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Definir el modelo secuencial en una función
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])
  # Optimizador
  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# Crear una instancia del modelo
model = create_model()

# Mostrar la arquitectura del modelo
model.summary()

# Generar la dirección donde se guardaran los datos
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# Crear una llamada que salva los pesos de los modelos
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Entrenar el modelo con la nueva llamada
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pasar el callback al entrenamiento

ls {checkpoint_dir}

# Crear una instancia de modelo básica
model = create_model()

# Evaluar el modelo
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Modelo no entrenado, exactitud: {:5.2f}%".format(100*acc))

# Cargar los pesos
model.load_weights(checkpoint_path)

# Re-evaluar el modelo
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Modelo restaurado, exactitud: {:5.2f}%".format(100*acc))

# Incluir la época en el nombre del archivo (utilizar 'str.format')
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Crear un callback que salva el peso de los models cada 5 épocas
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Crear una instancia de un nuevo modelo
model = create_model()

# Salvar los pesos utilizando el formato 'checkpoint_path'
model.save_weights(checkpoint_path.format(epoch=0))

# Entrenar el modelo con el nuevo callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)

ls {checkpoint_dir}

# Revisar entonces la última
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
# en general el formato de TF sólo salva los últimos 5 checkpoints

# Crear una nueva instancia del modelo
model = create_model()

# Cargar los pesos previamente salvados
model.load_weights(latest)

# Re-evaluar el modelo
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Modelo restaurado, exactitud: {:5.2f}%".format(100*acc))

