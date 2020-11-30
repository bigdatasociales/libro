"""
@author: Tinguaro Rodríguez
"""

# Importación de módulos y funciones
from sklearn.datasets import load_iris
from sklearn import tree 
from sklearn.model_selection import train_test_split, GridSearchCV


iris = load_iris() #carga del dataset Iris

# División del dataset en entrenamiento y test
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(iris.data , 
                        iris.target, test_size=0.4, random_state=13)

# Definición de los rangos de los parámetros
parametros_a_ajustar = [{'max_depth': [3,4,5], 
                         'min_samples_split': range(2, 51)}]

# Búsqueda en rejilla con validación cruzada sobre la muestra de entrenamiento
arbol = GridSearchCV(tree.DecisionTreeClassifier(),parametros_a_ajustar,cv=5)
arbol.fit(X_entrenamiento, y_entrenamiento)

# Salida de resultados de la búsqueda
print("Mejor configuración paramétrica:",arbol.best_params_)
print("Tasa de acierto en validación de la mejor configuración:",arbol.best_score_)
print("Estimación del rendimiento real:", arbol.score(X_test, y_test))

# Ajuste del árbol con los mejores parámetros sobre la muestra de entrenamiento
arbol = tree.DecisionTreeClassifier(max_depth=arbol.best_params_['max_depth'],
                    min_samples_split=arbol.best_params_['min_samples_split'])
arbol = arbol.fit(X_entrenamiento, y_entrenamiento)

# Visualización del mejor árbol
import graphviz
datos_grafico = tree.export_graphviz(arbol, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
grafico = graphviz.Source(datos_grafico)
grafico