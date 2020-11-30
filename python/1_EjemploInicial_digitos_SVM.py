"""
@author: Tinguaro Rodríguez
"""

from sklearn import svm, datasets, metrics


digitos = datasets.load_digits()


X = digitos.data
y = digitos.target
clasificador = svm.SVC(gamma=0.001)
clasificador.fit(X, y)


observados = digitos.target
predichos = clasificador.predict(X)
print("Informe de rendimiento del clasificador:\n%s\n"
      % ( metrics.classification_report(observados, predichos)))
print("Matriz de confusión:\n%s" % metrics.confusion_matrix(observados, predichos))


from sklearn.model_selection import train_test_split, cross_val_score

X_entrenamiento, X_test, y_entrenamiento, y_test = train_test_split(X , y, test_size=0.4, random_state=1)
clasificador.fit(X_entrenamiento, y_entrenamiento)
observados = y_test
predichos = clasificador.predict(X_test)

print("Informe de rendimiento del clasificador:\n%s\n"
      % ( metrics.classification_report(observados, predichos)))
print("Matriz de confusión:\n%s" % metrics.confusion_matrix(observados, predichos))


import numpy as np
val_cruzada = cross_val_score(clasificador, X, y, cv=5)
print("Tasas de acierto en test para cada iteración de validación cruzada: ", val_cruzada)
print("Tasas de acierto media en la validación cruzada: ", val_cruzada.mean())
print("Desviación típica de las tasas de acierto: ", val_cruzada.std())


