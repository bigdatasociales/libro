"""
@author: Tinguaro Rodríguez
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
import time

dataset = load_breast_cancer() # Carga del dataset Breast cancer
X, y = dataset.data, dataset.target;

X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X,y, 
                                        test_size=0.4, random_state=13)
# Definición de los rangos de los parámetros
parametros_a_ajustar = [{'max_depth': [4,5,6], 
                         'min_samples_split': range(2, 51)}]

# Búsqueda en rejilla con validación cruzada sobre la muestra de entrenamiento
comienzo = time.process_time()
arbol = GridSearchCV(tree.DecisionTreeClassifier(),parametros_a_ajustar,cv=5)
arbol.fit(X_entrenamiento, y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
      
# Salida de resultados de la búsqueda
print("Mejor configuración paramétrica:",arbol.best_params_)
print("Tasa de acierto en validación de la mejor configuración:",arbol.best_score_)
print("Estimación del rendimiento real:", arbol.score(X_test, y_test))

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Normalización de los inputs
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Conjuntos de entrenamiento y test normalizados
X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X,y,test_size=0.4, random_state=13)

# Entrenamiento de la red
comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic", solver="sgd",
                    batch_size=1, max_iter=50, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

# Gestión de los mensajes de aviso
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
# Ejecutando esta sentencia se suprimen los avisos sobre convergencia de la red neuronal
warnings.filterwarnings('ignore', category=UndefinedMetricWarning) 
# Ejecutando esta sentencia se restauran los avisos sobre convergencia de la red neuronal
warnings.filterwarnings('always', category=ConvergenceWarning)

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",solver="sgd",
                    batch_size=1, max_iter=10, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

from sklearn import metrics
print("Matriz de confusión:\n%s" % metrics.confusion_matrix(y_test,red.predict(X_test)))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",solver="sgd",
                    batch_size=1, max_iter=500, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))
print("Matriz de confusión:\n%s" % metrics.confusion_matrix(y_test,red.predict(X_test)))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes = (600,), activation="logistic",solver="sgd",
                    batch_size=1, max_iter=500, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes = (1,), activation="logistic",solver="sgd",
                    batch_size = 341, max_iter=500, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes = (10,), activation="logistic",solver="sgd",
                    batch_size = 341, max_iter=500, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes = (1,), activation="logistic",solver="sgd",
                    batch_size = 20, max_iter=5000, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",solver="sgd",
                    batch_size=1,learning_rate_init=0.0001,max_iter=50, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",solver="sgd",
                    batch_size=1,learning_rate_init=0.01,max_iter=50, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))

comienzo=time.process_time()
red = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",solver="sgd",
                    batch_size=1,learning_rate_init=0.1,max_iter=50, random_state=0);
red.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Estimación del rendimiento real:", red.score(X_test, y_test))