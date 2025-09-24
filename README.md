# TP : Support Vector Machines (SVM) avec scikit-learn

**Auteur : ATTOUMANI Ibrahim**

---

## Description
Ce projet illustre l’utilisation des **Support Vector Machines (SVM)** avec la librairie scikit-learn.  
Le rapport est rédigé en LaTeX.

---

## Prérequis

- Python 3.x  
- scikit-learn  
- matplotlib  
- numpy  

---

## Étapes du TP

### 1. Découverte de `sklearn.svm.SVC`
Un script d’exemple (`svm_script.py`) est fourni pour illustrer l’utilisation de `SVC`. Nous allons nous en inspiré de ce code pour réaliser nos propres expériences.

### 2. Classification sur le dataset *iris*
- Nous allons charger le dataset `iris` depuis `sklearn.datasets`.
- Puis, sélectionner uniquement les classes 1 et 2, ainsi que les deux premières variables.  
- Nous allons entraîner un **SVM linéaire** et garder 50% des données pour le test.  

### 3. Interface SVM (optionnel)
- Lancer le script [`svm_gui.py`](https://scikit-learn.org/1.2/auto_examples/applications/svm_gui.html) pour tester visuellement l’impact du choix du noyau et du paramètre **C**.  

### 4. Classification de visages
- Télécharger la base [LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz).  
- Modifier l’exemple dans `svm_script.py` pour :  
  - Étudier l’influence du paramètre **C** (échelle logarithmique de `1e5` à `1e-5`).  
  - Ajouter des variables parasites et constater la baisse de performance.  
  - Améliorer les résultats grâce à une réduction de dimension avec **PCA** (`sklearn.decomposition.PCA`).  
- Réfléchir au biais introduit par le prétraitement (voir lignes 215–241 de `svm_script.py`).  

---

## Références

- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*, Springer, 2009.  
- Schölkopf, Smola. *Learning with Kernels*, MIT Press, 2002.  
- Vapnik. *Statistical Learning Theory*, Wiley, 1998.  

---
