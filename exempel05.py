'''
  Overfitting y Underfitting
  En los problemas de clasificación o predicción, se puede observar que tras 
  una cierta cantidad de épocas en el entrenamiento, la precisión del modelo
  en los datos de validación llega a un pico y ocurren dos cosas, o se 
  estanca o comienza a disminuir; a esto se le llama Overfit, lo cual afecta
  que tan bien se comportará (o generalizará) el modelo a datos nuevos.

  Por otro lado, Underfitting se refiere a que "aún queda espacio para mejorar"
  el modelo, ya sea por que el modelo no está bien estructurado, esta sobre regulado
  o simplemente no se ha entrenado lo suficiente. Esto implica que el modelo
  no ha aprendido los patrones relevantes en el entrenamiento.
'''

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)

# Instalar los documentos de github
!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from  IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

# Generar una carpeta temporal para el trabajo
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
# Esta función, va borrando los directorios de manera recursiva, para que una
# vez que termina el programa, se elimine el directorio temporal
shutil.rmtree(logdir, ignore_errors=True)

print(logdir)

# Se descarga el grupo de datos de Higgs, el cual contiene 11 000 000 ejemplos
# cada uno con 28 características y una clase etiqueta binaria
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

gz

# Se coloca el número de características
FEATURES = 28

# Esto se puede aplicar para leer los archivos CSV directamente de un archivo gzip
# sin un paso de descomprim
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

ds

# Esto regresará una lista de escalares para cada registro
# La siguiente función rempaquetarpa esa lista de escalares en un par:
# (vector_caracteristicas, etiqueta)
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

# Debido a que TF es más eficiente con grupos de datos grandes, se empaquetará
# cada fila individual en un paquete de 10000 ejemplos
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# Se observan los datos
for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)

# Se acomodan los datos, considerando
# 1000 muestras para validación
N_VALIDATION = int(1e3)
# 10000 de entrenamiento
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
# Tamaño del batch 500
BATCH_SIZE = 500
# Y los pasos que se debe tomar por cada época
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

# Esto tomará los datos requeridos
# Además de que con cache() se asegura que no se requiere
# recargar los datos en cada época
# lo cual optimiza tanto el espacio como la velocidad
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

train_ds

# Estos grupos de datos regresan ejemplso individuales, por lo que se usa el método
# batch() para crear grupos de tamaño apropiado para el entrenamiento
# Usar shuffle() para "revolver" los datos
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# PROCESO PARA OBSERVAR EL OVERFITTING
# Hay que recordar siempre que estos modelos tienden a ser buenos para ajustar 
# los datos de entrenamiento, pero el reto es la generalización con datos 
# externos

# Muchos modelos realizan un mejor entrenamiento si gradualmente se reduce la
# velocidad de aprendizaje durante el entrenamiento
# esto indica que será un poco más lento conforme el modelo va aprendiendo
# Esto se logra con optmizers.schedules, en este caso un modelo de optimización
# que decae en la velocidad de aprendizaje de manera inversa al tiempo
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

# Lo anterior genera que el ritmo de aprendizaje disminuya hiperbólicamente
# a 1/2 de la velocidad base a las 1000 épocas, 1/3 a las 2000 y así sucesivimamente
step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoca')
plt.grid()
_ = plt.ylabel('Velocidad de aprendizaje')

# Se generararán "call backs" para el monitoreo de la entropía cruzada binaria
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

# Se genera una función para compilar y y ajustar el modelo
# puese harán pruebas de diferentes tipos de modelos con un número de capas
# y nodos diferentes
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

# MODELO PEQUEÑITO
# Este modelo solo contará con una capa de 16 nodos
# y una de salida de un nodo
# FA es "elu" (exponential linear unit)
# la cual intenta converger de manera más rápida a cero con resultados más precisos
# R(z) = 
#        si  z >  0  ->  z
#        si  z <= 0  ->  alpha(e^z - 1)
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

# Aquí se irán creando las "historias" o los datos de los modelos a probar
size_histories = {}

# compilar y ejecutar el modelo
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# Generar una gráfica que compare los valores de entrenamiento con los de validación
# utilizando para ello la entropía cruzada binaria
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

# MODELO PEQUEÑO
# Ahora se aumenta una capa de 16 nodos adicionales
small_model = tf.keras.Sequential([
    # `input_shape` es solo requerido para que `.summary` funcione corectamente.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# MODELO MEDIANO
# Se crea con 3 capas de 64 nodos y una de salida de 1 nodo
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])

size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

# MODELO GRANDE
# Se genera con 4 capas de 512 nodos y una de salida con un solo nodo
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])

size_histories['large'] = compile_and_fit(large_model, "sizes/large")

# COMPARAR LOS MODELOS DE MANERA GRÁFICA
# En esta gráfica las líneas sólidas representan la pérdida (loss) del entrenamiento
# las doscontinuas la pérdida en validación (entre menor sea este valor, mejor modelo es)
plotter.plot(size_histories)
# La escala en el eje x será logarítimica
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epocas [Log]")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Estrategias para prevenir el overfitting
# Debido a que en este caso, el modelo pequeñito es el que presenta
# un mejor comportamiento y no presenta overfitting, se hace una
# copia de los datos de entrenamiento para usarlos como comparación
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

# Primera forma de reducir el overfitting
# Añadir regularización de los pesos
# De la misma forma que en el principio de la Navaja de Occam
# la solución correcta es la más simple a un problema
# Entonces aplicando este principio, se colocan restricciones a la complejidad
# de un modelo al forzar a los pesos a solo tomar valores pequeños, lo que hace
# que la distribución de pesos sea más regular
# A esto se le conoce como regularización de pesos y se puede lograr
# al añadir a la función loss un costo asociado a tener pesos grandes
# Se puede hacer de dos formas:
# 1.- Regularización L1, donde el costo añadido es proporcional al valor absoluto
#     de los coeficientes de los pesos (esto pone los pesos hacia 0 alentando un
#     modelo disperso)
# 2.- Regularización L2, donde el costo añadido es proporcional al cuadrado del
#     valor de los coeficientes de los pesos (en este caso L2 penaliza los
#     parámetros de los pesos sin dispersar los valores ya que la penalización
#     lleva a cero los pesos pequeños)
# El caso del parámetro 0.001 es que cada coeficiente en la matriz de pesos de la
# capa añade 0.001 * valor_del_coeficiente^2 al total de pérdida de la red
# Se usa esta regularización en el modelo grande
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

# Una vez que se entrena, se observa la gráfica
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

result = l2_model(features)
regularization_loss=tf.add_n(l2_model.losses)

# Segunda forma de reducir el overfitting
# Añadir Dropouts
# Esta es la forma más eficiente de regularizar los modelos.
# Dropout consiste en llevar a 0 de manera aleatoria  un número de salidas de 
# cada una de las capas durante el entrenamiento, o sea, lo que realiza es que
# si la capa entrega un vector [0.2, 0.5, 1.3, 0.8, 1.1] tras aplicar el Dropout
# se obtiene [0, 0.5, 1.3, 0, 1.1]
# El "Dropout rate" es la fracción de datos de salida que se llevarán a 0
# Usualmente es colocada entre 0.2 y 0.5
dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

# Comparar
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

# Sin embargo al combinar ambas técnicas
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

# La mejora es realmente significativa
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])