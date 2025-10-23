import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from itertools import product
np.random.seed(42)

# -----------------------------
# БАЗОВІ ФУНКЦІЇ НАЛЕЖНОСТІ
# -----------------------------
@dataclass
class GaussianMF:
    c: float   # центр
    s: float   # σ (>0)

    def __call__(self, x):
        # μ(x) = exp( - (x-c)^2 / (2 s^2) )
        return np.exp(-0.5 * ((x - self.c) / (self.s + 1e-12)) ** 2)

    def dmu_dc(self, x):
        # ∂μ/∂c = μ * (x-c) / s^2
        mu = self.__call__(x)
        return mu * ((x - self.c) / (self.s**2 + 1e-12))

    def dmu_ds(self, x):
        # ∂μ/∂s = μ * ((x-c)^2) / (s^3)
        mu = self.__call__(x)
        return mu * (((x - self.c) ** 2) / (self.s**3 + 1e-12))

# -----------------------------
# ANFIS (Sugeno 1-го порядку)
# -----------------------------
class ANFIS:
    def __init__(self, n_inputs, mfs_per_input=2, lr_premise=1e-2, lr_conseq=1e-2):
        self.n_inputs = n_inputs
        self.mfs_per_input = mfs_per_input
        self.lr_premise = lr_premise
        self.lr_conseq = lr_conseq

        # 1) Ініціалізація МФ: рівномірно за [0,1] (дані масштабуємо)
        self.mf_params = [
            [GaussianMF(c, 0.2) for c in np.linspace(0.2, 0.8, mfs_per_input)]
            for _ in range(n_inputs)
        ]

        # 2) Правила: усі комбінації МФ по входах
        self.rules = list(product(*[range(mfs_per_input) for _ in range(n_inputs)]))
        self.n_rules = len(self.rules)

        # 3) Наслідки: [a1..aD, b] для кожного правила
        self.conseq = np.random.randn(self.n_rules, n_inputs + 1) * 0.01

        # Масштаби
        self.x_min = None
        self.x_max = None

    # ---------- масштабування ----------
    def _fit_scaler(self, X):
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        self.x_max[self.x_max == self.x_min] += 1e-6  # уникнути нульового діапазону

    def _scale(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-12)

    # ---------- прямий прохід ----------
    def _firing_strengths(self, Xs):  # Xs: scaled (N,D)
        N, D = Xs.shape
        W = np.ones((N, self.n_rules))
        for r_idx, combo in enumerate(self.rules):
            for d, mf_idx in enumerate(combo):
                W[:, r_idx] *= self.mf_params[d][mf_idx](Xs[:, d])
        return W  # (N,R)

    def predict(self, X):
        assert self.x_min is not None, "Модель ще не навчена"
        Xs = self._scale(X)
        N, D = Xs.shape
        W = self._firing_strengths(Xs)                # (N,R)
        W_sum = np.sum(W, axis=1, keepdims=True) + 1e-12
        W_norm = W / W_sum                            # (N,R)

        X_lin = np.hstack([Xs, np.ones((N, 1))])      # (N, D+1)
        # y_hat[n] = sum_r W_norm[n,r] * (conseq[r] · X_lin[n])
        y_hat = np.sum(W_norm * (X_lin @ self.conseq.T), axis=1)
        return y_hat

    # ---------- навчання (виправлено) ----------
    def fit(self, X, y, epochs=200, batch_size=64, verbose=True):
        self._fit_scaler(X)
        Xs = self._scale(X)
        N, D = Xs.shape
        y = y.reshape(-1)

        losses = []

        for ep in range(1, epochs + 1):
            idx = np.random.permutation(N)
            Xs_shuf, y_shuf = Xs[idx], y[idx]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb = Xs_shuf[start:end]
                yb = y_shuf[start:end]
                B = xb.shape[0]

                # --- прямий прохід ---
                W = self._firing_strengths(xb)                   # (B,R)
                W_sum = np.sum(W, axis=1, keepdims=True) + 1e-12
                W_norm = W / W_sum                               # (B,R)
                X_lin = np.hstack([xb, np.ones((B, 1))])         # (B,D+1)
                y_rule_out = X_lin @ self.conseq.T               # (B,R)
                y_hat = np.sum(W_norm * y_rule_out, axis=1)      # (B,)

                # --- похибка ---
                e = y_hat - yb                                    # (B,)
                loss = 0.5 * np.mean(e ** 2)

                # --- градієнти за наслідками ---
                # dL/dconseq[r,:] = mean_b (e[b] * W_norm[b,r] * X_lin[b,:])
                grad_conseq = (e[:, None, None] * W_norm[:, :, None] * X_lin[:, None, :]).mean(axis=0)

                # --- похідні по W (для premise-параметрів) ---
                # d y_hat / d W[b,r] = (y_rule_out[b,r] - y_hat[b]) / W_sum[b]
                dY_dW = (y_rule_out - y_hat[:, None]) / W_sum     # (B,R)

                # Попередньо обчислимо μ і їх похідні для кожного (d,mf)
                MU = {}
                dMU_dc = {}
                dMU_ds = {}
                for d in range(self.n_inputs):
                    x_d = xb[:, d]
                    for mf in range(self.mfs_per_input):
                        f = self.mf_params[d][mf]
                        MU[(d, mf)] = f(x_d)                         # (B,)
                        dMU_dc[(d, mf)] = f.dmu_dc(x_d)              # (B,)
                        dMU_ds[(d, mf)] = f.dmu_ds(x_d)              # (B,)

                # Градієнти для c та s
                grad_c = [[0.0 for _ in range(self.mfs_per_input)] for _ in range(self.n_inputs)]
                grad_s = [[0.0 for _ in range(self.mfs_per_input)] for _ in range(self.n_inputs)]

                # Для кожного правила r і кожного входу d:
                for r, combo in enumerate(self.rules):
                    # прод_без_d = добуток μ по всіх входах, крім d
                    # рахуємо напряму (простота > оптимальність)
                    for d in range(self.n_inputs):
                        mf_d = combo[d]
                        prod_others = np.ones(B)
                        for dd, mf_dd in enumerate(combo):
                            if dd == d:
                                continue
                            prod_others *= MU[(dd, mf_dd)]

                        dW_dc = prod_others * dMU_dc[(d, mf_d)]  # (B,)
                        dW_ds = prod_others * dMU_ds[(d, mf_d)]  # (B,)

                        # dL/dθ = mean_b ( e[b] * dY_dW[b,r] * dW[b,r]/dθ )
                        g_c = np.mean(e * dY_dW[:, r] * dW_dc)
                        g_s = np.mean(e * dY_dW[:, r] * dW_ds)

                        grad_c[d][mf_d] += g_c
                        grad_s[d][mf_d] += g_s

                # --- оновлення параметрів ---
                self.conseq -= self.lr_conseq * grad_conseq

                for d in range(self.n_inputs):
                    for mf in range(self.mfs_per_input):
                        self.mf_params[d][mf].c -= self.lr_premise * grad_c[d][mf]
                        self.mf_params[d][mf].s -= self.lr_premise * grad_s[d][mf]
                        # σ не даємо впасти нижче мінімуму
                        self.mf_params[d][mf].s = float(max(self.mf_params[d][mf].s, 0.05))

            losses.append(loss)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"Epoch {ep:4d}/{epochs}: loss={loss:.6f}")

        return np.array(losses)

# -----------------------------
# ДЕМО: апроксимація функції
# -----------------------------
def make_dataset(n=800, noise=0.05):
    X = np.random.rand(n, 2)
    y = np.sin(np.pi * X[:, 0]) + 0.5 * np.cos(np.pi * X[:, 1]) + X[:, 0] * X[:, 1]
    y += noise * np.random.randn(n)
    return X, y

def train_demo():
    X, y = make_dataset(n=1200, noise=0.05)
    n_tr = int(0.8 * len(X))
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr:], y[n_tr:]

    model = ANFIS(n_inputs=2, mfs_per_input=3, lr_premise=5e-3, lr_conseq=1e-2)
    losses = model.fit(Xtr, ytr, epochs=200, batch_size=128, verbose=True)

    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    rmse_tr = math.sqrt(np.mean((yhat_tr - ytr) ** 2))
    rmse_te = math.sqrt(np.mean((yhat_te - yte) ** 2))
    print(f"\nRMSE train: {rmse_tr:.4f} | RMSE test: {rmse_te:.4f}")

    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE/2)")
    plt.title("ANFIS training loss")
    plt.tight_layout()
    plt.savefig("anfis_loss.png", dpi=150)
    print("Збережено: anfis_loss.png")

    plt.figure(figsize=(6,4))
    plt.scatter(yte, yhat_te, s=10, alpha=0.6)
    lims = [min(yte.min(), yhat_te.min()), max(yte.max(), yhat_te.max())]
    plt.plot(lims, lims)
    plt.xlabel("True y")
    plt.ylabel("Pred y")
    plt.title("ANFIS: True vs Pred (test)")
    plt.tight_layout()
    plt.savefig("anfis_true_vs_pred.png", dpi=150)
    print("Збережено: anfis_true_vs_pred.png")

if __name__ == "__main__":
    train_demo()