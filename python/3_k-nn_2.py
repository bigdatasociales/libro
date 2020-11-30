"""
@author: Tinguaro Rodríguez
"""

# Importación de módulos y funciones
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

# Carga y normalización de los datos
iris = datasets.load_iris()
X = iris.data
y = iris.target
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# División de los ejemplos entre entrenamiento y test
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X , y,test_size=0.4, random_state=13)

max_validacion = 0
# Bucle para la selección de k
for k in range(1,16): 
    clasificador = neighbors.KNeighborsClassifier(k)
    # Validación cruzada con cv=10 iteraciones
    val_cruzada = cross_val_score(clasificador, X_entrenamiento, y_entrenamiento, cv=10)
    # val_cruzada = tasas de acierto de las 10 iteraciones
    validacion = val_cruzada.mean() # Media de las 10 tasas de acierto
    # Estimación para k
    print("Tasa de acierto para k = %s: %s" %(k,validacion))
    if validacion > max_validacion: # Si mejora las anteriores, se almacena con k
        max_validacion = validacion
        mejor_k = k
        
print("Mejor k =", mejor_k)
print("Tasa de acierto para mejor k =", max_validacion)

# Entrenar k-NN con el mejor k y la muestra de entrenamiento completa
clasificador = neighbors.KNeighborsClassifier(mejor_k)
clasificador.fit(X_entrenamiento, y_entrenamiento)
# Estimación final del rendimiento del k-NN con la muestra de test
print("Estimación del rendimiento real:", clasificador.score(X_test, y_test))