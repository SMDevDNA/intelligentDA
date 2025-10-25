import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1. Генерація синтетичних даних "Сонячна панель" ----------
np.random.seed(42)
N = 1500
G = np.clip(np.random.gamma(5, 120, N), 0, 1100)       # сонячна радіація, Вт/м²
Ta = np.random.normal(20, 8, N)                        # температура навколишнього повітря, °C
W = np.clip(np.random.normal(3, 1.2, N), 0, None)      # швидкість вітру, м/с
cos_theta = np.clip(np.random.beta(2, 2, N), 0, 1)     # косинус кута падіння

# температура елемента та потужність
T_cell = Ta + (G/800)*25 - 3*W
eta_ref, gamma, A = 0.19, 0.0045, 1.6
P = eta_ref * (1 - gamma*(T_cell-25)) * G * A * cos_theta
P = np.clip(P + np.random.normal(0, 50, N), 0, None)

df = pd.DataFrame(dict(G=G, T_cell=T_cell, cos_theta=cos_theta, W=W, P=P))

# ---------- 2. Побудова регресійної моделі ----------
X = df[["G", "T_cell", "cos_theta", "W"]]
y = df["P"]

lin = LinearRegression().fit(X, y)
pred = lin.predict(X)

# ---------- 3. Метрики моделі ----------
rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)

# ---------- 4. Коефіцієнти ----------
coef = lin.coef_
inter = lin.intercept_
print("=== Формальне рівняння регресії ===")
print(f"P = {inter:.2f}"
      f" + ({coef[0]:.3f})*G"
      f" + ({coef[1]:.3f})*T_cell"
      f" + ({coef[2]:.3f})*cosθ"
      f" + ({coef[3]:.3f})*W")

print(f"\nRMSE = {rmse:.2f},   R² = {r2:.3f}")

# ---------- 5. Стандартизовані коефіцієнти (β) ----------
Xs = (X - X.mean()) / X.std()
ys = (y - y.mean()) / y.std()
beta = LinearRegression().fit(Xs, ys).coef_
betas = pd.Series(beta, index=X.columns).sort_values(key=abs, ascending=False)

print("\n=== Стандартизовані β-коефіцієнти (вплив змінних) ===")
print(betas)

# ---------- 6. Візуалізація залишків ----------
residuals = y - pred
plt.figure(figsize=(5,3))
plt.scatter(pred, residuals, s=10, alpha=0.5)
plt.axhline(0, color='r', lw=1)
plt.xlabel("Передбачене P")
plt.ylabel("Залишок")
plt.title("Залишки vs передбачення (лінійна модель)")
plt.tight_layout()
plt.show()

# ---------- 7. Вплив cosθ при фіксованих середніх значеннях ----------
grid = np.linspace(0, 1, 100)
X_mean = X.mean()
Xp = pd.DataFrame({
    "G": X_mean["G"],
    "T_cell": X_mean["T_cell"],
    "cos_theta": grid,
    "W": X_mean["W"]
})
yp = lin.predict(Xp)

plt.figure(figsize=(5,3))
plt.plot(grid, yp, color="blue")
plt.xlabel("cosθ")
plt.ylabel("Передбачена потужність P, Вт")
plt.title("Вплив кута падіння променів на потужність (інші змінні – середні)")
plt.grid(True)
plt.tight_layout()
plt.show()
