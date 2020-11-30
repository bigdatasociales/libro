"""
@author: Tinguaro Rodríguez
"""

# Importación de módulos y funciones
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

# Carga del dataset Iris
iris = load_iris()
X, y = iris.data, iris.target;

# Escalamiento de los inputs
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# División del dataset en entrenamiento y test
# Nótese la opción stratify
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X , y,test_size=0.4, 
                                                                    random_state=13, stratify=y)
# Ajuste del algoritmo con la muestra de entrenamiento
# nótese que y_entrenamiento no interviene en el ajuste del clasificador no supervisado
comienzo=time.process_time()
kmeans = KMeans(n_clusters=3, init='random', n_init=1,
                random_state=0, algorithm='full')
kmeans.fit(X_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Centroides:\n%s" % kmeans.cluster_centers_) # centroides
print("Función objetivo:",kmeans.inertia_) # función objetivo

comienzo=time.process_time()
kmeans = KMeans(n_clusters=3, init='random', n_init=10,
                random_state=0, algorithm='full')
kmeans.fit(X_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Centroides:\n%s" % kmeans.cluster_centers_) # centroides
print("Función objetivo:",kmeans.inertia_) # función objetivo

from sklearn import metrics
predichos=kmeans.predict(X_entrenamiento)
print("Matriz de confusión:\n%s" % metrics.confusion_matrix(y_entrenamiento, predichos))

# Esta función toma las etiquetas de clase reales (obs) # y predichas (pred) y realiza
# una permutación de las últimas para que correspondan con la clase real mayoritaria
# en cada etiqueta predicha
import numpy as np
def cambia_etiquetas(obs,pred):
    N=obs.shape[0] # Número de ejemplos
    C=pred.max() # Número de clases y grupos
    # Obtención de la matriz de confusión
    matriz=np.zeros((C+1,C+1))
    for i in range(N):
        matriz[obs[i],pred[i]]+=1
    # Intercambio de etiquetas
    aux=pred.copy()
    for c in range(C+1):
        aux[pred==c]=matriz.argmax(axis=0)[c]
    return aux # Output de la función

predichos=cambia_etiquetas(y_entrenamiento,predichos)
print("Matriz de confusión:\n%s" 
      % metrics.confusion_matrix(y_entrenamiento,predichos))
print("Tasa de acierto:", metrics.accuracy_score(y_entrenamiento,predichos))

predichos=kmeans.predict(X_test)
predichos=cambia_etiquetas(y_test,predichos)
print("Matriz de confusión:\n%s" 
      % metrics.confusion_matrix(y_test, predichos))
print("Tasa de acierto:", metrics.accuracy_score(y_test,predichos))


import matplotlib.pyplot as plt
max_K=10
J=[]
for K in range(1,max_K+1):
    kmeans = KMeans(n_clusters=K, init='random', n_init=10,
                random_state=0, algorithm='full')
    kmeans.fit(X)
    J.append(kmeans.inertia_)
    
plt.figure()
plt.title("Seleccción de K mediante el método del codo")
plt.plot(range(1,max_K+1), J, color='darkorange',
         lw=2, label='Función objetivo J(K)')
plt.xticks(range(1,max_K+1), range(1,max_K+1), size=10)
plt.xlim([0.5, max_K+0.5])
plt.ylim([0.0, max(J)+1])
plt.xlabel('Valores de K')
plt.ylabel('Valores de J')
plt.legend(loc="upper right")
plt.show()