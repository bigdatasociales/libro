"""
@author: Tinguaro Rodríguez
"""

from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X , y,test_size=0.4, random_state=13)

for k in range(1,16):
    clasificador = neighbors.KNeighborsClassifier(k)
    clasificador.fit(X_entrenamiento, y_entrenamiento)
    print("Tasa de acierto para k = %s: %s" %(k,clasificador.score(X_test, y_test)))
   