# -*- coding: utf-8 -*-
"""
Аналіз часового ряду (вітрові турбіни): стаціонарність, (S)ARIMA, адекватність, прогноз 10 кроків.
Залежності: pandas, numpy, matplotlib, statsmodels, scipy (опц. для нормальності)
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ==========================
# Налаштування
# ==========================
CSV_PATH   = "wind_turbine.csv"   # шлях до вашого CSV
DATE_COL   = "timestamp"          # назва стовпця з датою/часом
TARGET_COL = "power_kw"           # досліджувана змінна (наприклад, power_kw або wind_speed)

RESAMPLE_TO_HOURLY = True         # якщо частота < 1 година — агрегуємо до години (mean)

# Якщо захочеш повернутись до повного grid search
MAX_P = 3
MAX_Q = 3
MAX_P_SEAS = 1
MAX_Q_SEAS = 1

N_FORECAST = 10                   # довжина прогнозу

# Збереження графіків
SAVE_FIGS = True
FIG_DIR   = "figs"

def _savefig(name):
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")

# ==========================
# Службові функції
# ==========================

def read_series(csv_path, date_col, target_col, resample_to_hourly=True):
    """Зчитати ряд, встановити DateTimeIndex, прибрати дублі/NaN, за потреби агрегувати до H."""
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Залишаємо тільки цільовий стовпець і прибираємо нечислове
    s = pd.to_numeric(df[target_col], errors="coerce").dropna()

    # Прибрати потенційні дублікати індексу
    s = s[~s.index.duplicated(keep="first")].sort_index()

    # Зафіксувати частоту, якщо можливо (допомагає SARIMAX)
    inf_freq = pd.infer_freq(s.index)
    if inf_freq:
        s = s.asfreq(inf_freq)

    # Якщо частота дрібніша за годину — ресемпл до години
    if resample_to_hourly and len(s) > 1:
        deltas_min = np.diff(s.index.values).astype('timedelta64[m]').astype(int)
        if len(deltas_min) > 0:
            avg_step = int(np.median(deltas_min))
            if avg_step < 60:
                s = s.resample("H").mean()

    return s.dropna()

def guess_seasonal_period(series):
    """Евристика для сезонного періоду s залежно від частоти."""
    freq = pd.infer_freq(series.index)
    if freq is None:
        return 0
    f = freq.upper()
    if f.startswith("H"):  # годинні
        return 24
    if f.startswith("D"):  # денні
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("T"):  # хвилинні
        try:
            step_min = int(f.split("T")[0]) if f != "T" else 1
        except:
            step_min = 1
        per = int(1440 / step_min)
        return per if per <= 24*60 else 0
    return 0

def adf_test(y, autolag='AIC'):
    res = adfuller(y.dropna(), autolag=autolag)
    stat, pval, usedlag, nobs, crit, icbest = res
    return {"stat": stat, "pval": pval, "usedlag": usedlag, "nobs": nobs, "crit_vals": crit, "icbest": icbest}

def kpss_test(y, regression='c', nlags='auto'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, lags, crit = kpss(y.dropna(), regression=regression, nlags=nlags)
    return {"stat": stat, "pval": pval, "lags": lags, "crit_vals": crit}

def choose_d_via_adf(y, max_d=2):
    d = 0
    while d <= max_d:
        test = adf_test(y.diff(d) if d > 0 else y)
        if test["pval"] < 0.05:
            return d, test
        d += 1
    return max_d, adf_test(y.diff(max_d))

def need_seasonal_diff(y, s, threshold=0.3):
    if s <= 1 or len(y) < s + 5:
        return 0
    acf_s = y.autocorr(lag=s)
    return 1 if (acf_s is not None and np.abs(acf_s) >= threshold) else 0

# --- Швидкий підбір без зависань ---
def fast_search_sarima(y, d=0, D=1, s=24):
    """
    Швидкий підбір (S)ARIMA серед невеликого, але практичного набору моделей
    для годинних даних із добовою сезонністю. Повертає (model, order, seasonal_order, best_aic).
    """
    non_seasonal = [(1,d,1), (2,d,1), (1,d,2), (2,d,2), (0,d,1), (1,d,0)]
    seasonal    = [(0,D,1,s), (1,D,0,s), (1,D,1,s)]

    best = (None, None, None, np.inf)
    for order in non_seasonal:
        for seas in seasonal:
            try:
                model = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seas,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                    simple_differencing=True  # пришвидшує
                ).fit(disp=False, method="lbfgs", maxiter=150, concentrate_scale=True)
                aic = model.aic
                if aic < best[3]:
                    best = (model, order, seas, aic)
            except Exception:
                continue

    if best[0] is None:
        raise RuntimeError("Не вдалося підібрати модель серед швидких кандидатів.")
    return best

# --- Діагностика та безпечні графіки ---
def residual_diagnostics(model):
    resid = model.resid.dropna()
    diag = {}
    if len(resid) >= 10:
        lag_for_lb = min(24, max(5, len(resid)//10))
        lb = acorr_ljungbox(resid, lags=[lag_for_lb], return_df=True)
        diag["ljungbox_pvalue"] = float(lb["lb_pvalue"].iloc[-1])
        try:
            sample = resid.sample(min(5000, len(resid)), random_state=42) if len(resid) > 5000 else resid
            sh = stats.shapiro(sample)
            diag["shapiro_pvalue"] = float(sh.pvalue)
        except Exception:
            diag["shapiro_pvalue"] = np.nan
    else:
        diag["ljungbox_pvalue"] = np.nan
        diag["shapiro_pvalue"] = np.nan
    return resid, diag

def safe_plot_series(y, title, fname):
    y = y.dropna()
    if len(y) == 0:
        print(f"[plot] '{title}': немає даних для побудови.")
        return
    plt.figure()
    y.plot()
    plt.title(title)
    plt.xlabel("Час")
    plt.ylabel("Значення")
    plt.tight_layout()
    _savefig(fname)
    plt.close()

def safe_plot_acf_pacf(y, title_prefix, fname_acf, fname_pacf, max_lags=40):
    y = y.dropna()
    n = len(y)
    if n < 5:
        print(f"[plot] '{title_prefix}': занадто мало точок ({n}). Пропускаю ACF/PACF.")
        return
    lags = max(1, min(max_lags, n // 2))

    plt.figure()
    plot_acf(y, lags=lags)
    plt.title(f"{title_prefix}: ACF (lags={lags})")
    plt.tight_layout()
    _savefig(fname_acf)
    plt.close()

    plt.figure()
    plot_pacf(y, lags=lags)
    plt.title(f"{title_prefix}: PACF (lags={lags})")
    plt.tight_layout()
    _savefig(fname_pacf)
    plt.close()

def plot_residuals(resid):
    resid = resid.dropna()
    if len(resid) < 5:
        print(f"[plot] Залишки: занадто мало точок ({len(resid)}). Пропускаю графіки.")
        return
    plt.figure()
    plt.plot(resid.index, resid.values)
    plt.title("Залишки моделі")
    plt.xlabel("Час")
    plt.ylabel("Залишок")
    plt.tight_layout()
    _savefig("residuals_series")
    plt.close()

    lags = max(1, min(40, len(resid)//2))
    plt.figure()
    plot_acf(resid, lags=lags)
    plt.title("ACF залишків")
    plt.tight_layout()
    _savefig("residuals_acf")
    plt.close()

    plt.figure()
    plot_pacf(resid, lags=lags)
    plt.title("PACF залишків")
    plt.tight_layout()
    _savefig("residuals_pacf")
    plt.close()

# ==========================
# Основний сценарій
# ==========================
def main():
    # 1) Зчитування ряду
    y = read_series(CSV_PATH, DATE_COL, TARGET_COL, RESAMPLE_TO_HOURLY)
    print(f"К-сть спостережень після підготовки: {len(y)}")
    print(f"Індексна частота (інференс): {pd.infer_freq(y.index)}")

    safe_plot_series(y, "Початковий ряд", "raw_series")

    # 1) Перевірка стаціонарності (ADF + KPSS)
    adf0 = adf_test(y)
    try:
        kpss0 = kpss_test(y)
    except Exception:
        kpss0 = {"pval": np.nan}
    print("\n=== Стаціонарність (на вихідному ряді) ===")
    print(f"ADF p-value = {adf0['pval']:.4f}  (p<0.05 ⇒ стаціонарний)")
    print(f"KPSS p-value = {kpss0['pval'] if not np.isnan(kpss0['pval']) else 'NaN'}  (p>0.05 ⇒ стаціонарний)")

    # 2) Оцінка диференціювання (d, D)
    s = guess_seasonal_period(y)
    D = need_seasonal_diff(y, s) if s > 1 else 0
    d, _ = choose_d_via_adf(y if D == 0 else y.diff(s), max_d=2)

    print("\n=== Обрані порядки диференціювання ===")
    print(f"d = {d} (несезонне), D = {D} (сезонне), s = {s}")

    # Ряд після різниць
    y_stationary = y.copy()
    if D > 0:
        y_stationary = y_stationary.diff(s)
    if d > 0:
        y_stationary = y_stationary.diff(d)
    y_stationary = y_stationary.dropna()

    safe_plot_series(y_stationary, "Ряд після диференціювання (прагнемо стаціонарності)", "stationary_series")
    safe_plot_acf_pacf(y_stationary, "Стаціонаризований ряд", "stationary_acf", "stationary_pacf", max_lags=40)

    # 3–4) Підбір та оцінювання (S)ARIMA — швидкий підбір
    model, order, seasonal_order, best_aic = fast_search_sarima(y, d=d, D=D, s=s)
    print("\n=== Найкраща модель (швидкий підбір) ===")
    print(f"order = {order}, seasonal_order = {seasonal_order}, AIC = {best_aic:.2f}")
    print(model.summary())

    # 5) Перевірка адекватності
    resid, diag = residual_diagnostics(model)
    print("\n=== Діагностика залишків ===")
    print(f"Ljung–Box p-value = {diag['ljungbox_pvalue']:.4f} (p>0.05 ⇒ залишки ~ білий шум)")
    print(f"Shapiro–Wilk p-value = {diag['shapiro_pvalue'] if not np.isnan(diag['shapiro_pvalue']) else 'NaN'} (p>0.05 ⇒ близько до нормальних)")
    plot_residuals(resid)

    adequate = (not np.isnan(diag["ljungbox_pvalue"])) and (diag["ljungbox_pvalue"] > 0.05)
    print(f"\nАдекватність моделі за критерієм білого шуму залишків: {'ТАК' if adequate else 'НІ'}")

    # 6) Прогноз на 10 періодів уперед
    forecast_res = model.get_forecast(steps=N_FORECAST)
    fc_mean = forecast_res.predicted_mean
    fc_ci = forecast_res.conf_int()

    forecast_df = pd.DataFrame({
        "forecast": fc_mean,
        "lower": fc_ci.iloc[:, 0],
        "upper": fc_ci.iloc[:, 1]
    })
    print("\n=== Прогноз на 10 кроків уперед ===")
    print(forecast_df)

    # Графік прогнозу
    plt.figure()
    y.plot(label="Історія")
    fc_mean.plot(label="Прогноз")
    plt.fill_between(fc_ci.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], alpha=0.2, label="95% ДІ")
    plt.title("Прогноз SARIMA на 10 кроків уперед")
    plt.xlabel("Час")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.tight_layout()
    _savefig("forecast_10_steps")
    plt.close()

    print(f"\nГрафіки збережено у папку: {os.path.abspath(FIG_DIR)}")

if __name__ == "__main__":
    main()
