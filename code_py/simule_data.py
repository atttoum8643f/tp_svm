# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:23:03 2025

@author: ATTOUMANI Ibrahim
"""

# Générer un jeu de données binaire fortement déséquilibré (90% vs 10%)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from collections import Counter
import numpy as np

# paramètres
n_samples = 1000
n_features = 2          # pour pouvoir tracer en 2D
weights = [0.9, 0.1]    # 90% de la classe 0, 10% de la classe 1
class_sep = 1.0         # séparation entre les classes (ajuster si nécessaire)
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

# Créer deux modèles SVM linéaires avec C différent
models = {
    "SVM linéaire, C=1.0": SVC(kernel="linear", C=1.0),
    "SVM linéaire, C=0.001": SVC(kernel="linear", C=0.001, probability=True)
}

# Entraîner les modèles
for name, model in models.items():
    model.fit(X, y)

# Tracer les frontières de décision
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    # Grille pour visualiser la frontière
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 500),
        np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 500)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Affichage
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=20, label="Classe 0 (majoritaire)")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=30, marker="x", label="Classe 1 (minoritaire)")
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
