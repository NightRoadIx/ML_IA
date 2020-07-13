# Descargar los datos más actuales
!pip install -q sklearn
!pip install -q -U tf-estimator-nightly
!pip install -q -U tf-nightly

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Cargar el set de datos del Titanic para predecir la supervivencia de los 
# pasajeros, tomando en cuenta las características como su gpenero, edad
# clase, etc.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Ver como están los datos
dftrain.head()

# Descripción de los datos (las estadísticas de los datos numéricos)
dftrain.describe()

# Cuantos datos son
dftrain.shape[0], dfeval.shape[0]

# Observar gráficamente la distribución de los pasajeros con respecto de su edad
we = dftrain.age.hist(bins=20)
we.set_xlabel("Edad")
we.set_ylabel("Pasajeros")

we2 = dftrain.sex.value_counts().plot(kind='barh')
we2.set_xlabel("Pasajeros")

we3 = dftrain['class'].value_counts().plot(kind='barh')
we3.set_xlabel("Pasajeros")

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% supervivencia')

'''
  Los estimadores usan un sistema llamado columnas de características para 
  describir cómo el modelo debe interpretar cada una de las características de 
  entrada sin formato. Un estimador espera un vector de entradas numéricas, y 
  las columnas de características describen cómo el modelo debe convertir cada 
  característica.
  Seleccionar y crear el conjunto correcto de columnas de características es 
  clave para aprender un modelo efectivo. Una columna de entidades puede ser 
  una de las entradas sin procesar en el dict de entidades original 
  (una columna de entidades base), o cualquier columna nueva creada usando 
  transformaciones definidas sobre una o múltiples columnas base 
  (columnas de entidades derivadas).
  El estimador lineal utiliza características numéricas y categóricas. 
  Las columnas de características funcionan con todos los estimadores de 
  TensorFlow y su propósito es definir las características utilizadas para 
  el modelado. Además, proporcionan algunas capacidades de ingeniería de 
  características como codificación en caliente, normalización y bucketización.
'''
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

vocabulary

feature_columns

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

# Función de entrenamiento
train_input_fn = make_input_fn(dftrain, y_train)
# Función de evaluación
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

train_input_fn

ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
  print('Algunas características clave:', list(feature_batch.keys()))
  print()
  print('Un lote de clase:', feature_batch['class'].numpy())
  print()
  print('Un lote de etiquetas:', label_batch.numpy())

# Se puede inspeccionar el resultado de una columna característica específica 
# utilizando tf.keras.layers.DenseFeatures
# La columna de edad
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()

# La columna de género
gender_column = feature_columns[0]
# Para este casose transforma en una columna indicador, ya que es categórica
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()

# Una vez que ya se tienen todas las características base, se entrena el modelo
# pero en el caso del estimador sólo se utilza tf.estimator
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

# Ahora para asegurar que el modelo aprenda de las diferencias entre cada una 
# de las diferentes combinaciones de características, se puede añadir una 
# crossed_column al modelo
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

age_x_gender

# Una vez añadido esto, se vuelve a enrenar el modelo
derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result)

# Ya que antes se definió con un set de datos de evaluación sin revolver
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

we5 = probs.plot(kind='hist', bins=20, title='Probabilidades predichas')
we5.set_xlabel("Probabilidad de supervivencia")
we5.set_ylabel("Frecuencia")

'''
  Finalmente, se observa la Receiver Operating Characteristic
  { característica operativa del receptor (ROC) }
  de los resultados, lo que proporciona una mejor idea de la compensación 
  entre la tasa positiva verdadera y la tasa falsa positiva.

  La curva ROC es una representación gráfica de la sensibilidad frente a la
  especificidad para un sistema clasificador binario al variar el umbral de
  discriminación.
  Es muy utilizada en pruebas de diagnóstico, en el que tras realizar alguna
  prueba o examen, obteniendo un valor positivo o negativo,
  se observa el porcentaje de diagnósticos que son verdaderos positivos o
  falsos positivos, esto es que si una prueba es positiva puede que la prueba
  acierte o falle y lo mismo si la prueba es negativa.
'''
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('Curva ROC curve')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.xlim(0,)
plt.ylim(0,)
plt.grid()

