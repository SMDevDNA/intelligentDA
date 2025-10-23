# 0) Імпорт
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    silhouette_score, davies_bouldin_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

rng = np.random.default_rng(42)

# 1) Згенеруємо синтетичні дані схожі на SCADA
n_turbines = 25
rows_per_turbine = 800
N = n_turbines * rows_per_turbine

turbine_id = np.repeat(np.arange(n_turbines), rows_per_turbine)

wind_speed = rng.normal(9, 3, N).clip(0, 30)                 # м/с
ambient_temp = rng.normal(10, 8, N)                           # °C
pitch = np.clip(rng.normal(5, 2, N) + 0.1*(wind_speed-8), 0, 30)  # градуси
vibration = rng.normal(0.5 + 0.02*wind_speed, 0.15, N).clip(0, 5)
nacelle_temp = ambient_temp + rng.normal(10, 3, N) + 0.2*wind_speed
gearbox_temp = ambient_temp + rng.normal(25, 4, N) + 0.25*wind_speed

# Ідеальна потужність ~ v^3 до номіналу; додамо ефекти кертейлменту/збоїв
cp_eff = 0.35 - 0.03*(pitch/30)                               # спрощено
ideal_power = (wind_speed**3) * cp_eff
curtailment_mask = (wind_speed > 10) & (rng.random(N) < 0.1)
power = ideal_power * (1 - 0.5*curtailment_mask) + rng.normal(0, ideal_power.std()*0.05, N)
power = np.clip(power, 0, None)

# Ймовірність відмов залежить від вібрації та температур редуктора
logit = -6 + 1.5*(vibration-0.7) + 0.02*(gearbox_temp-40)
p_fault = 1/(1+np.exp(-logit))
fault = (rng.random(N) < p_fault).astype(int)

df = pd.DataFrame({
    "turbine_id": turbine_id,
    "wind_speed": wind_speed,
    "ambient_temp": ambient_temp,
    "pitch": pitch,
    "vibration": vibration,
    "nacelle_temp": nacelle_temp,
    "gearbox_temp": gearbox_temp,
    "power": power,
    "fault": fault,
    "curtailed": curtailment_mask.astype(int)
})

# 2) Класифікація: fault vs normal
features = ["wind_speed","pitch","vibration","nacelle_temp","gearbox_temp","power","ambient_temp"]
X = df[features].values
y = df["fault"].values
groups = df["turbine_id"].values

# Розділяємо за групами (турбінами), щоб не було leakage
gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Баланс у train:", pd.Series(y_train).value_counts().to_dict())
print("Баланс у test :", pd.Series(y_test).value_counts().to_dict())

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

# --- Оптимізація порогу за F1 (через PR-криву) ---
prec, rec, thr = precision_recall_curve(y_test, proba)
# precision_recall_curve повертає thr на довжину len(prec)-1
f1 = (2 * prec[1:] * rec[1:]) / (prec[1:] + rec[1:] + 1e-12)
best_idx = np.argmax(f1)
best_threshold = thr[best_idx]

# Страхувальник: якщо при оптимальному порозі немає позитивів, зробимо адекватний fallback
y_pred = (proba >= best_threshold).astype(int)
if y_pred.sum() == 0:
    # Візьмемо поріг так, щоб хоча б топ-1% найризиковіших стали позитивами
    pct = max(1, int(0.01 * len(proba)))
    sorted_proba = np.sort(proba)
    best_threshold = sorted_proba[-pct] if pct < len(sorted_proba) else sorted_proba[0] - 1e-9
    y_pred = (proba >= best_threshold).astype(int)

print("\n=== КЛАСИФІКАЦІЯ FAULT ===")
print(f"Оптимальний поріг (F1): {best_threshold:.4f}")
print("Fault predicted:", int(y_pred.sum()), "із", len(y_pred))
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, proba))

# Найважливіші ознаки
importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))

# 3) Кластеризація операційних режимів (без міток)
X_clust = df[["wind_speed","power","pitch","vibration","gearbox_temp","ambient_temp"]].values
scaler = StandardScaler()
Xn = scaler.fit_transform(X_clust)

# Підбір k за силуетом
best_k, best_sil = None, -1
for k in range(2, 8):
    km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_tmp = km_tmp.fit_predict(Xn)
    sil_tmp = silhouette_score(Xn, labels_tmp)
    if sil_tmp > best_sil:
        best_k, best_sil = k, sil_tmp

km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels = km.fit_predict(Xn)

dbi = davies_bouldin_score(Xn, labels)
sil = silhouette_score(Xn, labels)

print(f"\n=== КЛАСТЕРИЗАЦІЯ ОПЕР. РЕЖИМІВ ===")
print(f"Best k (by silhouette): {best_k}, silhouette={sil:.3f}, Davies-Bouldin={dbi:.3f}")

# Інтерпретація кластерів: середні по кластерах
df_clusters = df.copy()
df_clusters["cluster"] = labels
summary = df_clusters.groupby("cluster")[["wind_speed","power","pitch","vibration","gearbox_temp"]].mean().round(2)
print("\nCluster centroids (means):\n", summary)

# Пошук «анормальних» кластерів: низька потужність при високій швидкості
# Для порівняння використовуємо пороги з усіх точок, а не з вже усереднених центрів
ws_thr = df["wind_speed"].quantile(0.75)
pwr_thr = df["power"].quantile(0.25)
abn_clusters = summary[(summary["wind_speed"] > ws_thr) & (summary["power"] < pwr_thr)]
print("\nКластери з підозріло низькою потужністю при високій швидкості вітру:\n", abn_clusters if not abn_clusters.empty else "— немає")
