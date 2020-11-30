"""
@author: Tinguaro Rodríguez
"""

import numpy as np
from sklearn.datasets import load_breast_cancer

# Carga del dataset Breast cancer en un objeto de tipo bunch
mi_bunch = load_breast_cancer()

# En los atributos data y target de este bunch se encuentran los arrays
# con los valores de inputs y target de cada ejemplo, los cuales se cargan en X e y
X, y = mi_bunch.data, mi_bunch.target

# Lista de nombres de variables o columnas para ser exportada.
# A los nombres de los inputs obtenidos con el atributo feature_names
# se les añade el nombre ‘clase’ para la variable target
columnas=mi_bunch.feature_names
columnas=np.append(columnas,'clase')
print(columnas)

# Creación de un único array con los datos de inputs y targets de todos los ejemplos, uno por fila
Xy=np.hstack((X,np.reshape(y,(X.shape[0],1))))

import csv
archivo=open('breast_cancer.csv','w',newline='')
writer = csv.writer(archivo,delimiter=';')
writer.writerow(columnas)
writer.writerows(Xy)
archivo.close()

# Importación del archivo CSV a un pandas dataframe
import pandas as pd
filename='breast_cancer.csv'
datos = pd.read_csv(filename,sep=';')

# Obtención de los nombres de variable
columnas=datos.columns.get_values()
print(columnas[columnas.shape[0]-1])

# Separación de los datos de inputs y target
X2=datos.get_values()[:,:columnas.shape[0]-1]
y2=datos.get_values()[:,columnas.shape[0]-1]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(X2,y2)
print("Tasa de acierto:", rf.score(X2, y2))

from sklearn.model_selection import train_test_split
import time
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X,y, test_size=0.4, 
                                                                    random_state=13)
comienzo=time.process_time()
rf = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=0);
rf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tasa de acierto en entrenamiento:", rf.score(X_entrenamiento,y_entrenamiento))
print("Estimación del rendimiento real:", rf.score(X_test, y_test))
print("Estimación out-of-bag",rf.oob_score_)

comienzo=time.process_time()
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0);
rf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tasa de acierto en entrenamiento:", rf.score(X_entrenamiento,y_entrenamiento))
print("Estimación del rendimiento real:", rf.score(X_test, y_test))
print("Estimación out-of-bag",rf.oob_score_)

comienzo=time.process_time()
rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=0);
rf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tasa de acierto en entrenamiento:", rf.score(X_entrenamiento,y_entrenamiento))
print("Estimación del rendimiento real:", rf.score(X_test, y_test))
print("Estimación out-of-bag",rf.oob_score_)

comienzo=time.process_time()
rf = RandomForestClassifier(n_estimators=10000, oob_score=True, random_state=0);
rf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tasa de acierto en entrenamiento:", rf.score(X_entrenamiento,y_entrenamiento))
print("Estimación del rendimiento real:", rf.score(X_test, y_test))
print("Estimación out-of-bag",rf.oob_score_)

# Obtención de importancias y reordenación de los índices
importancia = rf.feature_importances_
indices = np.argsort(importancia)[::-1]

# Ranking de importancia
print("Ranking de importancia de las variables:")
for i in range(X.shape[1]):
    print("%d. %s  (%f)" %(i + 1,mi_bunch.feature_names[indices[i]],importancia[indices[i]]))

import matplotlib.pyplot as plt
variables=4
lista=[]
for i in range(variables):
    lista.append(mi_bunch.feature_names[indices[i]])
print(lista[:variables])
plt.figure()
plt.title("Importancia de variables")
plt.bar(range(variables), importancia[indices[:variables]],
       color="r", align="center")
plt.xticks(range(variables), lista[:variables], size=7)
plt.xlim([-1, variables])
plt.show()

from sklearn.datasets import make_classification
X1, y1 = make_classification(n_samples=1000000, n_features=10,n_informative=3,
                             n_redundant=0, n_repeated=0, n_classes=2,
                             random_state=0, shuffle=False)

comienzo=time.process_time()
comienzo2=time.time()
rf = RandomForestClassifier(n_estimators=10,n_jobs=1,random_state=0);
rf.fit(X1,y1)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tiempo transcurrido:",time.time()-comienzo2)

comienzo=time.process_time()
comienzo2=time.time()
rf = RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=0);
rf.fit(X1,y1)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tiempo transcurrido:",time.time()-comienzo2)
