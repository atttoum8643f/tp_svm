# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 03:04:02 2025

@author: ATTOUMANI Ibrahim
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

##########################################################################
#                       Chargement et filtrage des données
##########################################################################
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X = X[Y != 0, :2]  # classes 1 et 2, 2 premières features
Y = Y[Y != 0]

##########################################################################
#                       Séparation train/test
##########################################################################
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.9, random_state=18
)

##########################################################################
#                       Entraînement du SVM polynomial
##########################################################################
clf_poly = SVC(kernel="poly", degree=3)  # noyau polynomial de degré 3
clf_poly.fit(X_train, Y_train)

##########################################################################
#               Création d'une grille pour visualiser la frontière
##########################################################################
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))


Z_poly = clf_poly.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

##########################################################################
#                       Affichage
##########################################################################
plt.contourf(xx, yy, Z_poly, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=50, marker='o', label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=50, marker='x', label="Test")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("SVM polynomial sur Iris (classes 1 et 2)")
plt.legend()
plt.show()

##########################################################################
#                       Évaluation
##########################################################################
accuracy_poly = clf_poly.score(X_test, Y_test)
print(f"Précision sur le test (polynomial) : {accuracy_poly:.2f}")
