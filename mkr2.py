

# =========================
# Correlation: G vs P
# =========================
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats


os.makedirs("figs", exist_ok=True)
def savefig(fig, name):
    path = os.path.join("figs", name); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return path


# Load data into DataFrame `df`: try to read "data.csv" if it exists, otherwise create example data
if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")
else:
    np.random.seed(0)
    G = np.random.uniform(30, 120, size=200)
    P = 0.5 * G + np.random.normal(0, 10, size=200)
    df = pd.DataFrame({"G": G, "P": P})


mask = df["G"] > 50
Gm, Pm = df.loc[mask, "G"].values, df.loc[mask, "P"].values
n = len(Gm)


# Pearson
r, p = stats.pearsonr(Gm, Pm)
# 95% CI (Fisher z)
z = np.arctanh(r)
se = 1/np.sqrt(n-3)
z_low, z_high = z - 1.96*se, z + 1.96*se
r_low, r_high = np.tanh(z_low), np.tanh(z_high)


# Spearman
rho, p_rho = stats.spearmanr(Gm, Pm)


print(f"Pearson r={r:.3f} 95% CI [{r_low:.3f}, {r_high:.3f}], p={p:.2e}, n={n}")
print(f"Spearman rho={rho:.3f}, p={p_rho:.2e}")


# Scatter + simple linear fit
coef = np.polyfit(Gm, Pm, 1)
xline = np.linspace(Gm.min(), Gm.max(), 200)
yline = np.polyval(coef, xline)


fig = plt.figure(figsize=(4,3))
plt.scatter(Gm, Pm, s=10, alpha=0.6)
plt.plot(xline, yline)
plt.xlabel("G (W/m^2)"); plt.ylabel("P (W)")
plt.title("Correlation: G vs P")
plt.tight_layout()
savefig(fig, "corr_scatter.png")
print("[FIG: corr_scatter.png]")
