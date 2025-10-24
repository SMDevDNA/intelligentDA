

# index.py
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# ---- сумісні метрики (працюють і зі старим sklearn, і без нього)
def rmse_compat(y_true, y_pred):
    try:
        from sklearn.metrics import mean_squared_error
        try:
            return mean_squared_error(y_true, y_pred, squared=False)  # нові версії
        except TypeError:
            return np.sqrt(mean_squared_error(y_true, y_pred))        # старі версії
    except Exception:
        # fallback без sklearn
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


def r2_compat(y_true, y_pred):
    try:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))
    except Exception:
        # простий R^2 без sklearn
        y_true = np.asarray(y_true)
        ss_res = np.sum((y_true - np.asarray(y_pred))**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return float(1 - ss_res/ss_tot) if ss_tot != 0 else float("nan")


def sat_model(Gs, a, b, c):
    return a * (1 - np.exp(-b * Gs)) + c


def _savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fit_p_vs_stabilized_g(df, gamma=0.0045, outdir="figs", maxfev=20000):
    required = {"G", "T_cell", "P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Відсутні колонки у df: {sorted(missing)}")


    G_raw = df["G"].to_numpy(float)
    T_cell = df["T_cell"].to_numpy(float)
    P = df["P"].to_numpy(float)


    # стабілізація
    G_s = G_raw * (1.0 - gamma * (T_cell - 25.0))


    # очищення
    M = np.column_stack([G_s, P])
    mask = np.isfinite(M).all(axis=1)
    if not mask.any():
        raise ValueError("Після очищення від NaN/Inf не лишилося даних.")
    G_s, P = G_s[mask], P[mask]


    eps = 1e-12
    keep = G_s > eps
    if not keep.any():
        raise ValueError("Усі значення G_s ≤ 0 після стабілізації.")
    G_s, P = G_s[keep], P[keep]


    # старт і межі
    p0 = [max(P.max(), eps), 0.005, float(np.percentile(P, 1))]
    bounds = ([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])


    # підбір
    pars, cov = curve_fit(sat_model, G_s, P, p0=p0, bounds=bounds, maxfev=maxfev)
    a, b, c = pars
    perr = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [np.nan, np.nan, np.nan]


    # метрики
    pred = sat_model(G_s, *pars)
    rmse = rmse_compat(P, pred)
    r2 = r2_compat(P, pred)


    # візуалізації
    order = np.argsort(G_s)
    G_sorted, P_sorted, pred_sorted = G_s[order], P[order], pred[order]
    xs = np.linspace(G_sorted.min(), G_sorted.max(), 400)
    ys = sat_model(xs, *pars)


    fig = plt.figure(figsize=(5, 3.6))
    plt.scatter(G_sorted, P_sorted, s=10, alpha=0.5)
    plt.plot(xs, ys, linewidth=1.5)
    plt.xlabel("G_stab"); plt.ylabel("P (W)")
    plt.title("Nonlinear fit: P = a(1 - exp(-b·G_s)) + c")
    plt.tight_layout()
    path_fit = _savefig(fig, outdir, "nonlinear_fit.png")


    res = P_sorted - pred_sorted
    fig = plt.figure(figsize=(5, 3.6))
    plt.scatter(pred_sorted, res, s=12, alpha=0.6)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Fitted P"); plt.ylabel("Residual")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    path_res = _savefig(fig, outdir, "nonlinear_residuals.png")


    return {
        "params": {"a": a, "b": b, "c": c},
        "param_stderr": {"a": perr[0], "b": perr[1], "c": perr[2]},
        "rmse": rmse, "r2": r2,
        "fig_fit": path_fit, "fig_residuals": path_res,
    }


def main():
    ap = argparse.ArgumentParser(description="Fit P vs stabilized G")
    ap.add_argument("csv", help="Шлях до CSV з колонками G,T_cell,P")
    ap.add_argument("--gamma", type=float, default=0.0045)
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()


    df = pd.read_csv(args.csv)
    res = fit_p_vs_stabilized_g(df, gamma=args.gamma, outdir=args.outdir)


    a,b,c = res["params"]["a"], res["params"]["b"], res["params"]["c"]
    sa,sb,sc = res["param_stderr"]["a"], res["param_stderr"]["b"], res["param_stderr"]["c"]
    print(f"Fitted: a={a:.4g} ± {sa:.2g}, b={b:.4g} ± {sb:.2g}, c={c:.4g} ± {sc:.2g}")
    print(f"RMSE={res['rmse']:.4g}, R2={res['r2']:.4f}")
    print(f"[FIG] {res['fig_fit']}")
    print(f"[FIG] {res['fig_residuals']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
        raise

