# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:28:00 2025

@author: ATTOUMANI Irahim
"""

# Générer un jeu de données binaire fortement déséquilibré (90% vs 10%)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from collections import Counter
import numpy as np

# paramètres
n_samples = 1000
n_features = 2
weights = [0.9, 0.1]
class_sep = 1.0
random_state = 0

X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           weights=weights,
                           flip_y=0.01,
                           class_sep=class_sep,
                           random_state=random_state)

print("Répartition des classes :", Counter(y))

# Deux modèles : non pondéré vs pondéré
models = {
    "non pondéré": SVC(kernel="linear", C=1.0, class_weight=None),
    "pondéré": SVC(kernel="linear", C=1.0, class_weight="balanced")
}

# Entraînement
for model in models.values():
    model.fit(X, y)

# Tracé des points
plt.figure(figsize=(7, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=20, c="lightblue", label="Classe 0 (majoritaire)")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=30, c="brown", marker="x", label="Classe 1 (minoritaire)")

# Grille pour la frontière
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 500),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 500)
)

# Frontières de décision
colors = {"non pondéré": "black", "pondéré": "red"}

for name, model in models.items():
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors=colors[name], linewidths=2, label=name)

# Légende personnalisée
handles = [
    plt.Line2D([0], [0], color="black", label="non pondéré"),
    plt.Line2D([0], [0], color="red", label="pondéré")
]
plt.legend(handles=handles + plt.gca().get_legend_handles_labels()[0], loc="best")

plt.title("SVM linéaire : comparaison non pondéré vs pondéré")
plt.show()