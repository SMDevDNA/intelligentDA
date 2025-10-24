import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


np.random.seed(42)
os.makedirs("figs", exist_ok=True)


def savefig(fig, name):
    path = os.path.join("figs", name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- 1) Synthetic PV data
N = 2000
G = np.clip(np.random.gamma(5, 120, N), 0, 1100)                # irradiance W/m^2
Ta = np.random.normal(20, 8, N)                                  # ambient temp
W = np.clip(np.random.normal(3, 1.2, N), 0, None)                # wind m/s
cos_theta = np.clip(np.random.beta(2, 2, N), 0, 1)               # angular factor
# PV cell temperature (simple NOCT-like)
T_cell = Ta + (G/800)*25 - 3*W                                   # °C
# Power model: P ≈ η_ref * (1 - γ*(T_cell-25)) * G * A * cosθ + noise
eta_ref, gamma, A = 0.19, 0.0045, 1.6                            # nominal eff, temp coeff, area m^2
P = eta_ref * (1 - gamma*(T_cell-25)) * G * A * cos_theta
P = np.clip(P + np.random.normal(0, 60, N), 0, None)             # add noise, ensure non-negative


df = pd.DataFrame(dict(G=G, Ta=Ta, W=W, cos_theta=cos_theta, T_cell=T_cell, P=P))


# Define classes by quantiles of P
q1, q2 = np.quantile(P, [0.33, 0.66])
y = np.where(P < q1, "Low", np.where(P < q2, "Med", "High"))


X = df[["G","T_cell","cos_theta","W"]].copy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ---- 2) Train Decision Tree
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
tree.fit(Xtr, ytr)


# ---- 3) Evaluation
pred = tree.predict(Xte)
acc = accuracy_score(yte, pred)
print("Decision Tree accuracy:", round(acc, 3))
print(classification_report(yte, pred))


cm = confusion_matrix(yte, pred, labels=["Low","Med","High"])
fig = plt.figure(figsize=(4,3))
ConfusionMatrixDisplay(cm, display_labels=["Low","Med","High"]).plot(values_format='.0f')
plt.title("Decision Tree – Confusion Matrix"); plt.tight_layout()
savefig(fig, "dt_confusion.png")
print("[FIG: dt_confusion.png]")


# ---- 4) Feature importances
imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
fig = plt.figure(figsize=(4,3))
imp.plot(kind="bar")
plt.title("Decision Tree – Feature Importances"); plt.tight_layout()
savefig(fig, "dt_importances.png")
print("[FIG: dt_importances.png]")


# ---- 5) Tree plot
fig = plt.figure(figsize=(10,6))
plot_tree(tree, feature_names=X.columns, class_names=["High","Med","Low"], filled=True, rounded=True)
plt.tight_layout()
savefig(fig, "dt_structure.png")
print("[FIG: dt_structure.png]")
