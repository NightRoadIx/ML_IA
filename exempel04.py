'''
Solución a un problema de regresión utilizando TF
En este caso lo que se desea es predecir salidas de valores continuos, como un precio o una probabilidad
A diferencia de los problemas de clasificación, en donde la idea es seleccionar una clase de untre un grupo

Se utiliza lo siguiente:
# seaborn para la graficación
pip install -q seaborn

# Algunas funciones de tensorflow_docs
pip install -q git+https://github.com/tensorflow/docs
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tf.__version__)

# Descargar los datos
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

# Se obtendrán los datos para su manipulación haciendo uso de la librería pandas
# Se van a obtener los datos a partir del nombre las columnas
# Ya que estos datos se encuentran en un archivo CSV
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# Leer los datos
# Los valores que no existan (o NA) se cambian por "?"
# el separador de los datos es " "
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

# Se hace una copia de los datos
dataset = raw_dataset.copy()
# Mostrar los últimos datos
dataset.tail()

# Se contabilizan los datos que se tienen
dataset.count()

# Se revisa si hay datos perdidos o NA
dataset.isna().sum()

# se pueden realizar varias cosas con estos datos para que no afecten al momento de procesar
# en este caso lo que se hará es eliminarlos
dataset = dataset.dropna()

# Se contabilizan los datos que se tienen
dataset.count()

# en el caso de la columna "Origin", los datos son categorias, no numéricos
# Por lo que se convierten en varias columnas
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

# Se separan los datos para obtener los grupos de entrenamiento y prueba
# La proporción será 80-20 %, sin "revolverlos" aleatoriamente
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# contabilizar los datos del grupo de entrenamiento
train_dataset.count()

# Ver los datos
train_dataset.tail()

# Se pueden observar los datos
# QUE ES LO QUE HACE?
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# y las estadísticas
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# separar el valor objetivo o "etiqueta" de las características
# esta etiqueta es el valor que tomará para entrenar y a predecir
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Observar algunas de las etiquetas tomadas
train_labels.tail()

# Crear una función para la normalización de los datos
# Es una buena práctica el normlizar los datos pues generalmente, estos datos 
# tienen diferentes escalas e intervalos
# y a pesar de que el modelo puede entrenarse sin normalizar, esto vuelve más
# complejo el proceso y lo más importante, hace el modelo dependiente de las
# unidades que maneja la entrada
def norm(x):
  # la normalización se hace al dato restando el promedio y dividiendo entre la desviación estándar
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# CUIDADO!
# es extremadamente importante que los datos que vayan a ingresarse al modelo
# para predicción, se tienen que ingresar de la misma forma normlizados 
# ene ste caso por el promedio y la desviación estándar

normed_train_data.tail()

len(train_dataset.keys())

# EL MODELO
# Se usa un modelo secuencial
# con dos capas ocultas de tipo Dense
# usando como FA una de tipo relu
# observar que los datos de entrada serán de acuerdo a la longitud de train_dataset
# la salida es un valor único continuo
# Se compila con lo siguente:
# función de pérdida loss mediante MSE (Mean Squared Error, Error Cuadrático Medio)
#  métrica muy utilizada en estos problemas de regresión
# Optimizador, La esencia de RMSprop es:
#  *Mantener un promedio móvil (con descuento) del cuadrado de gradientes
#  *Divide el gradiente entre la raíz de este promedio
#  utiliza como argumento el ritmo de aprendizaje
#  https://keras.io/api/optimizers/rmsprop/
# Metricas por supuesto serán MSE y MAE (Mean Absolute Error, error absoluto promedio)
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Llamar a la función que genera el modelo
model = build_model()

# inspeccionar el modelo
model.summary()

# Ahora probar el modelo, tomando un conjunto de 10 ejemplos de los datos de entrenamiento
example_batch = normed_train_data[:10]
# Y utilizar predict(), para realizar las predicciones
example_result = model.predict(example_batch)
example_result

# Entrenar el modelo
EPOCHS = 1000

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])

# Una vez entrenado, se puede ver en el historial como fue el entrenamiento
# y el movimiento de las métricas que se utilizaron
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Se generará una gráfica de este historial
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# La métrica que se tomará es el MAE
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

# La métrica que se tomará es el MSE
plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG$^2$]') # se usa LaTEX para mostrar el cuadrado en superíndice

# Se utilizará algo para detener automáticamente cuando los valores de validación ya no mejoren
# Lo que se usa es EarlyStopping

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

# Se rehace el modelo
model = build_model()

# El parámetro "patience" es la cantidad de épocas para revisar la mejora
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 
early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

# Graficar la métrica MAE
plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')

# Ahora se verá que tan bien el modelo realiza generalización al utilizar el grupo de datos de prueba
# Esto nos permitirá saber que tan bien funcionará el modelo al ponerlo en fucionamiento
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("El valor de MAE para las pruebas: {:5.2f} MPG".format(mae))

# Hacer predicciones con los datos de prueba
test_predictions = model.predict(normed_test_data).flatten()

# Esta gráfica muestra esos valores predichos por el modelo
# y compararlos con los valores reales
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('Valores reales [MPG]')
plt.ylabel('Predicciones [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.grid()
_ = plt.plot(lims, lims, color="red")

# Ahora se verá cual es la distribución de los errores
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Error de predicción [MPG]")
_ = plt.ylabel("Conteo")

