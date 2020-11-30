"""
@author: Tinguaro Rodríguez
"""

# Importación de módulos
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris() # Carga del dataset Iris

# Definición y ajuste de un árbol de clasificación
arbol = tree.DecisionTreeClassifier()
arbol = arbol.fit(iris.data, iris.target)
# Visualización del árbol
import graphviz
datos_grafico = tree.export_graphviz(arbol,out_file=None)
grafico = graphviz.Source(datos_grafico)
grafico.render("iris") # Crea el archivo iris.pdf con el árbol
grafico # Esta sentencia requiere la salida del gráfico por pantalla

# Visualización del árbol con apariencia más "amable"
datos_grafico = tree.export_graphviz(arbol, out_file=None,
                                      feature_names=iris.feature_names,
                                      class_names=iris.target_names,
                                      filled=True, rounded=True,
                                      special_characters=True)
grafico = graphviz.Source(datos_grafico)
grafico.render("iris") 
grafico 

print("Tasa de acierto del árbol:",arbol.score(iris.data, iris.target))