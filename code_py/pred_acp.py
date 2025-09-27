import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Charger le dataset LFW (visages)
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw.data
y = lfw.target

# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA initiale (toutes les composantes possibles)
pca_full = PCA(svd_solver="randomized", whiten=True, random_state=42)
pca_full.fit(X_train)

# Valeurs propres
eigenvalues = pca_full.explained_variance_

# Critère de Kaiser : garder uniquement les composantes avec variance >= 1
mask = eigenvalues >= 1
n_kaiser = np.sum(mask)

print(f"Nombre de composantes retenues par le critère de Kaiser : {n_kaiser}")

# Refaire une PCA avec ce nombre de composantes
pca = PCA(n_components=n_kaiser, svd_solver="randomized", whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Dimension initiale : {X.shape[1]}")
print(f"Dimension après PCA (Kaiser) : {X_train_pca.shape[1]}")


# --------- SCREE PLOT (Variance expliquée) ---------
var_exp = pca_full.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(8,5))
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label="Variance par composante")
plt.plot(range(1, len(cum_var_exp)+1), cum_var_exp, marker="o", color="red", label="Variance cumulée")
plt.axhline(y=1/len(X_train), color="green", linestyle="--", label="Seuil bruit (théorique)")
plt.axhline(y=1, color="orange", linestyle="--", label="Seuil Kaiser (=1)")
plt.xlabel("Nombre de composantes principales")
plt.ylabel("Variance expliquée")
plt.title("Scree Plot - Variance expliquée par les composantes PCA")
plt.legend()
plt.grid(True)
plt.show()

# SVM (linéaire)
clf = SVC(kernel="linear", C=1.0, random_state=42)
clf.fit(X_train_pca, y_train)

# Évaluation
y_pred = clf.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"Précision avec PCA (critère de Kaiser) : {acc:.3f}")

# Inertie expliquée (variance cumulée captée par PCA)
cum_var_exp_pca = np.cumsum(pca.explained_variance_ratio_)
print(f"Inertie expliquée par PCA (critère de Kaiser) : {cum_var_exp_pca[-1]:.5f}")
