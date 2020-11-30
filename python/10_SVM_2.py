"""
@author: Tinguaro Rodríguez
"""

from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

dataset = load_boston() # Carga del dataset Boston housing
X, y = dataset.data, dataset.target;

# Normalización de los inputs
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_entrenamiento, X_test, y_entrenamiento, y_test =train_test_split(X,y,test_size=0.4,
                                                                   random_state=13)
# Ajuste de la regresión de vector soporte
comienzo=time.process_time()
regresion = SVR();
regresion.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Error cuadrático medio:",
      metrics.mean_squared_error(y_test,regresion.predict(X_test)))
print("Error absoluto medio:",
      metrics.mean_absolute_error(y_test,regresion.predict(X_test)))
print("Error absoluto mediano:",
      metrics.median_absolute_error(y_test,regresion.predict(X_test)))
print("Coef. de determinación:",
      metrics.r2_score(y_test,regresion.predict(X_test)))


# Transformación de la variable target, en entrenamiento y en test
import numpy as np
y_entrenamiento_log=np.log(y_entrenamiento)
y_test_log=np.log(y_test)

# Nuevo ajuste del modelo con la variable transformada
comienzo=time.process_time()
regresion.fit(X_entrenamiento,y_entrenamiento_log)

# Salida de resultados
# Los valores predichos han de «destransformarse»
print("Duración del proceso:",time.process_time()-comienzo)
print("Error cuadrático medio:",
      metrics.mean_squared_error(y_test,np.exp(regresion.predict(X_test))))
print("Error absoluto medio:",
      metrics.mean_absolute_error(y_test,np.exp(regresion.predict(X_test))))
print("Error absoluto mediano:",
      metrics.median_absolute_error(y_test,np.exp(regresion.predict(X_test))))
print("Coef. de determinación:",
      metrics.r2_score(y_test,np.exp(regresion.predict(X_test))))