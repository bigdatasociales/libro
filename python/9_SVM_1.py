"""
@author: Tinguaro Rodríguez
"""

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer() # Carga del dataset Breast cancer
X, y = dataset.data, dataset.target;

# Normalización de los inputs
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Partición en muestras de entrenamiento y test
X_entrenamiento, X_test, y_entrenamiento, y_test =train_test_split(X,y,test_size=0.4, random_state=13)

import time
comienzo=time.process_time()
clf = SVC()
clf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)
print("Tasa de acierto en entrenamiento:",clf.score(X_entrenamiento,y_entrenamiento))
print("Estimación del rendimiento real:", clf.score(X_test, y_test))

from sklearn import metrics
print("Kappa de Cohen:",metrics.cohen_kappa_score(y_test,clf.predict(X_test)))
print("Precisión clase positiva:",metrics.precision_score(y_test,clf.predict(X_test),pos_label=0))
print("Exhaustividad clase positiva:",metrics.recall_score(y_test,clf.predict(X_test),pos_label=0))
print("AUC:",metrics.roc_auc_score(y_test,clf.decision_function(X_test)))

import numpy as np
rango_gamma=(np.linspace(0.0001,0.01,num=15))
parametros_a_ajustar = [{'kernel': ['linear']},
  {'kernel': ['rbf'], 'gamma': rango_gamma},
  {'kernel': ['poly'], 'gamma': rango_gamma, 'degree': [1,2,3,4,5],
   'coef0': range(-1,2)},
  {'kernel': ['sigmoid'],'gamma': rango_gamma,'coef0': range(-1,2)}]

medidas={'acierto' : 'accuracy', 
         'precisión' : metrics.make_scorer(metrics.precision_score,pos_label=0) , 
         'exhaustividad' : metrics.make_scorer(metrics.recall_score,pos_label=0),
         'kappa' : metrics.make_scorer(metrics.cohen_kappa_score),
         'auc' : 'roc_auc'}

from sklearn.model_selection import GridSearchCV
comienzo=time.process_time()
clf = GridSearchCV(SVC(),param_grid=parametros_a_ajustar,
                  scoring=medidas, cv=5, refit='exhaustividad')
clf.fit(X_entrenamiento,y_entrenamiento)
print("Duración del proceso:",time.process_time()-comienzo)

# Salida de resultados de la búsqueda
print("Mejor configuración paramétrica:",clf.best_params_)
print("Exhaustividad media de la mejor configuración:",clf.best_score_)
print("Tasa de acierto media de la mejor configuración:",
      clf.cv_results_['mean_test_acierto'][clf.best_index_])
print("Precisión media de la mejor configuración:",
      clf.cv_results_['mean_test_precisión'][clf.best_index_])
print("AUC media de la mejor configuración:",
      clf.cv_results_['mean_test_auc'][clf.best_index_])

clf = SVC(kernel='linear')
clf = clf.fit(X_entrenamiento, y_entrenamiento)
print("Tasa de acierto:", clf.score(X_test, y_test))
print("Kappa de Cohen:", metrics.cohen_kappa_score(y_test,clf.predict(X_test)))
print("Precision clase positiva:", metrics.precision_score(y_test,clf.predict(X_test),pos_label=0))
print("Exhaustividad clase positiva:", metrics.recall_score(y_test,clf.predict(X_test),pos_label=0))
print("AUC:", metrics.roc_auc_score(y_test,clf.decision_function(X_test)))

import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, -1*clf.decision_function(X_test),
                                         pos_label=0)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Curva ROC (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
