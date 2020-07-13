'''
    Un modelo Keras consiste en:
    
    * La arquitectura o configuración, la cual especifícia que capas del modelo
      hay y como están conectadas
    * Un grupo de valores de pesos (el estado del modelo)
    * Un optmizador (definido por el compilado del modelo)
    * Un grupo de pérdidas y métricas (definido al compilar el modelo)
    
    Keras permite guardar y posteriormente recuperar todos estos datos en disco
    todo esto, mediante las funciones:
    model.save('path')
    keras.models.load_model('path')
'''
#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generar la función para la creación del modelo
def get_model():
    # Creear un modelo simple con entrada de 32 datos
    inputs = keras.Input(shape=(32,))
    # la salida
    outputs = keras.layers.Dense(1)(inputs)
    # generar el modelo
    model = keras.Model(inputs, outputs)
    # compilar con su respectivo optmizador y función de pérdida
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Generar el modelo
model = get_model()
# Ver el resumen del modelo
model.summary()

#%%
# Entrenarlo
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

#%%
# Ahora salvar el modelo, en un archivo que se llame "mi_modelo"
model.save("mi_modelo")

#%%
# Ahora la parte interesante, se puede mandar a llamar al modelo
reconstructed_model = keras.models.load_model("mi_modelo")

#%%
# Revisar el modelo
reconstructed_model.summary()
# Revisar al predecir con ambos modelos utilizando los datos de prueba:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

#%%
# El modelo cargado ya está compilado y retiene el estado del optimizador
# por lo que el entrenamiento se puede reanudar
reconstructed_model.fit(test_input, test_target)