# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 20:35:41 2025

@author: ATTOUMANI Ibrahim
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Charger le dataset LFW (visages)
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw.data
y = lfw.target

# Ajout de variables de nuisance (bruit gaussien)
n_samples, n_features = X.shape
noise = np.random.randn(n_samples, 500)  # par ex. +500 variables de bruit
X_noisy = np.hstack((X, noise))

# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.3, random_state=0, stratify=y
)

# Normalisation (important pour SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Valeurs de C à tester (logarithmique)
C_values = np.logspace(-5, 5, 11)  # de 1e-5 à 1e5
errors = []

# Entraînement et évaluation
for C in C_values:
    clf = SVC(kernel="linear", C=C, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    errors.append(1 - acc)  # taux d'erreur

# Affichage
plt.figure(figsize=(7,5))
plt.semilogx(C_values, errors, marker="o", linestyle="-", color="red")
plt.xlabel("Paramètre C (échelle logarithmique)")
plt.ylabel("Erreur de prédiction")
plt.title("Impact des variables de nuisance sur la performance du SVM")
plt.grid(False)
plt.show()


# Fonction pour entraîner et calculer erreur
def erreur_svm(X, y, n_noise=0, seed=0):
    np.random.seed(seed)  # fixer la graine pour le bruit
    
    n_samples, n_features = X.shape
    if n_noise > 0:
        noise = np.random.randn(n_samples, n_noise)
        X = np.hstack((X, noise))

    # Split (graine fixée)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM avec C=1.0
    clf = SVC(kernel="linear", C=1.0, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return 1 - accuracy_score(y_test, y_pred)

# Exemple
err_clean = erreur_svm(X, y, n_noise=0, seed=18)
err_noisy = erreur_svm(X, y, n_noise=500, seed=18)

print(f"Erreur sans nuisance : {err_clean:.3f}")
print(f"Erreur avec nuisance : {err_noisy:.3f}")
