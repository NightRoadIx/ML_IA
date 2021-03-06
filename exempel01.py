#%%
# Clasificación de imágenes
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model

import os
import numpy as np
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DESCARGA DE LAS IMÁGENES Y ORDENAMIENTO DE LAS MISMAS
# Descargar las imagenes de la base de datos de perros vs gatos y almacenarla en el directorio /tmp
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# Esto obtiene el archivo comprimido de una URL y se extrae en una carpeta
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# Se genera la variable con el nombre del directorio 'cats_and_dogs_filtered'
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

'''
El directorio que se genera tiene la siguiente estructura
cats_and_dogs_filtered
|__ train
    |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
    |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
|__ validation
    |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]

'''
#%%
# Una vez que se extraen los datos, se asignan las variables para los grupos de 
# entrenamiento y validación
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Directorios de validación
train_cats_dir = os.path.join(train_dir, 'cats')  # imágenes de gatos
train_dogs_dir = os.path.join(train_dir, 'dogs')  # imágenes de perros
# Directorios de entrenamiento
validation_cats_dir = os.path.join(validation_dir, 'cats')  # imágenes de gatos
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # imágenes de perros

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Analizar los datos que se obtuvieron

# Número total de imágenes de entrenamiento
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

# Número total de imágenes de validación
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

# Total de imágenes
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

# Se colocan algunas variables para su uso posterior en el programa
# batch_size o número de lotes, que permite realizar la actualización de los pesos sinápticos terminando la época
# o al realizar un determinado paso de este número de elementos
batch_size = 128
# Épocas, el número de iteraciones a realizar para el entrenamiento de la red
epochs = 50
# Tamaño de las imágenes
IMG_HEIGHT = 150
IMG_WIDTH = 150

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Preparación de los datos

# La clase proporcionada por tf.keras ImageDataGenerator puede leer imágenes y pre-procesarlas
# para convertirlas en tensores, también configura los generadores que convierten las imágenes
# en grupos de tensores (batches of tensors)
train_image_generator = ImageDataGenerator(rescale=1./255) # Generador para los datos de entrenamiento
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generador para los datos de validación

# Se cargan las imágenes, se rescalan y redimensionan
# shuffle permite "revolver" las imágenes de manera aleatoria
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Visualizar las imágenes

# La función next regresa un lote de imágenes del set de datos
# regresa valores de la forma (x_train, y_train), donde:
# x_train son los características de entrenamiento (imágenes)
# y_train son las etiquetas {Estas se descartan, no se asignan a variable alguna}
sample_training_images, _ = next(train_data_gen)

# Esta función graficará las imágenes en forma de una rejilla 1 x 5
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Crear el modelo

# Aquí se genera un modelo que consiste en tres bloques de convolución 
# con las capas máximas pool en cada uno de ellos.
# Las capas de convolución (Conv2D) extraen características de la 
# imagen. La convolución ayuda al emborronamiento, agudizado, detección de bordes
# reducción de ruido y otras operaciones que ayudan a obtener las características
# específicas de la imagen.
# Max pooling es un proceso de discretización basado en muestras
# su objetivo es disminuir el número de muestras como representación de entrada,
# reduciendo así su dimensionalidad y permitiendo que se realicen suposiciones
# acerca de las características contenidas en las sub-regiones.
# La instrucción Flatten lo que realiza es convertir los datos de salida
# en un vectro simple de n elementos
# Hay una capa completamente conectada con 512 unidades que son activadas (Dense)
# por una FA del tipo relu (Rectified Linear Unit, y = max(0, x))
# La salida del modelo es una clasificación binaria mediante la FA sigmoide
# (binaria pues se están clasificando dos clases de fotografías)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Observar el resumen del modelo generado 
print(model.summary())

#%%
# Entrenar el modelo

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

print(history.history.keys())

#%%
# Visualizar los resultados del entrenamiento
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# En las gráficas anteriores se observa que la precisión del entrenamiento crece
# de forma lineal con el tiempo, mientras que la precisión de validación se estanca
# en alrededor del 70% en el proceso de entrenamiento. Además la diferencia en precisión
# entre el entrenamiento y la validación es evidente -un signo del overfitting
# Eliminar el overfitting por medio del aumento de los datos de entrenamiento
# lo que se hace es, en primer lugar, evitar que la red sea entrenada
# o que vea la misma imagen dos veces durante el entrenamiento
# por lo que se tiene que hacer ligeras modificaciones a las imágenes que
# se tienen de entrenamiento

# Aplicar una rotación horizontal
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

# Tomar una muestra de los ejemplos de entrenamiento y repetirla 5 veces
# para que se observe el resultado sobre la misma imagen
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Visualizar las nuevas imágenes
plotImages(augmented_images)

#%%
# Rotar la imagen de manera aleatoria
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

#%%
# Ahora hacer zoom
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

#%%
# Juntar todos estos grupos de datos
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

# Visualizar una sola imagen con todas las transformaciones
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Ahora para la validación
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')
#%%
# Otra técnica para reducir el overfitting es la introducción de dropouts a la red
# Es una forma de regularización que fuerza a los pesos neuronales a tomar
# solo valores pequeños, lo que hace la distribución de los pesos más regulares
# y la red puede reducir el overfitting en entrenamientos con muestras pequeñas

# Crear una nueva red con Dropouts
# esto hace, de manera aleatoria, que el 20% de las neuronas se vayan a 0
model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el nuevo modelo
model_new.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Ver el resumen del modelo
model_new.summary()

# Entrenar el modelo
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# Visualizar el nuevo modelo
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


'''
predictions = model.predict(test_images)
# Se puede observar la primera predicción
predictions[0]
# Se observa cual es el que tiene el valor máximo, que representaría el que tiene la mayor
# probabilidad de que sea la etiqueta
np.argmax(predictions[0])
'''