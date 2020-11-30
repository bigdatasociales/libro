"""
@author: Tinguaro Rodríguez
"""

# Importación de módulos y funciones
from sklearn import datasets
from sklearn import naive_bayes, neighbors, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

vinos=datasets.load_wine() #carga del dataset Wine

# Normalización de los datos
X = vinos.data
y = vinos.target
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# División del dataset en entrenamiento y test
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X , 
                        y, test_size=0.4, random_state=13)

# k vecinos más cercanos
# Definición de los rangos de los parámetros
parametros_knn = [{'n_neighbors': range(1, 15)}]

# Búsqueda en rejilla con validación cruzada sobre la muestra de entrenamiento
knn = GridSearchCV(neighbors.KNeighborsClassifier(),parametros_knn,cv=5)
knn.fit(X_entrenamiento, y_entrenamiento)

# Salida de resultados
print("Mejor configuración paramétrica del K-NN:",knn.best_params_)
print("Tasa de acierto en validación de la mejor configuración del K-NN:",knn.best_score_)
print("Estimación del rendimiento real del K-NN:", knn.score(X_test, y_test))


# Árbol de clasificación
# Definición de los rangos de los parámetros
parametros_tree = [{'max_depth': [3,4,5], 
                         'min_samples_split': range(2, 51)}]

# Búsqueda en rejilla con validación cruzada sobre la muestra de entrenamiento
arbol = GridSearchCV(tree.DecisionTreeClassifier(random_state=1),parametros_tree,
                     cv=5)
arbol.fit(X_entrenamiento, y_entrenamiento)

# Salida de resultados
print("Mejor configuración paramétrica del árbol:",arbol.best_params_)
print("Tasa de acierto en validación de la mejor configuración del árbol:",arbol.best_score_)
print("Estimación del rendimiento real del árbol:", arbol.score(X_test, y_test))

# Clasificador naive Bayes gaussiano
NB=naive_bayes.GaussianNB()
NB.fit(X_entrenamiento, y_entrenamiento)
print("Estimación del rendimiento real del Naive Bayes:", NB.score(X_test, y_test))