# ============================================
# Wind Turbine: Time-Series Study & Forecast
# ============================================

from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

# ---- backend для збереження графіків без екрана (важливо для Windows/серверів)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests

# ======== НАЛАШТУВАННЯ КОРИСТУВАЧА ========
CSV_PATH: Optional[str] = None     # напр.: r"D:\data\scada.csv" або None щоб згенерувати
OUTPUT_DIR = ""  # туди покладемо PNG та CSV
FORECAST_HOURS = 24 * 7            # 7 днів
RUN_SARIMAX = False                # True -> повільніше, але з екзогенами
SARIMAX_WINDOW_DAYS = 120          # скільки днів брати у train для SARIMAX
RANDOM_SEED = 7

# ======== УТИЛІТИ ========
def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def safe_rmse(y_true, y_pred) -> float:
    """RMSE без параметра 'squared' (сумісно зі старими sklearn)."""
    # вирівнюємо по індексу, якщо це Series
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        y_pred = y_pred.reindex(y_true.index)
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask])**2)))

def ensure_dt_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if ts_col not in df.columns:
            raise ValueError(f"Очікую колонку часу '{ts_col}' або DatetimeIndex.")
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    df = df.sort_index()
    return df

def simulate_data(
    start="2024-10-01 00:00:00",
    end="2025-09-30 23:00:00",
    freq="h",      # <— ЛІТЕРА 'h' (не 'H') щоб не було FutureWarning
    seed=RANDOM_SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq)
    n = len(idx)

    baseline = 7.0
    daily = 2.0 * np.sin(2*np.pi * np.arange(n) / 24.0)
    annual = 1.2 * np.sin(2*np.pi * np.arange(n) / (24.0 * 365.25))
    weather = rng.normal(0, 1.5, size=n)
    gusts = np.maximum(0, rng.normal(0, 0.7, size=n))
    wind_speed = np.clip(baseline + daily + annual + weather + gusts, 0, None)

    P_rated = 3_000.0  # кВт
    v_rated = 12.0
    v_cut_in = 3.0
    v_cut_out = 25.0

    def power_curve(v):
        v = np.asarray(v)
        P = np.zeros_like(v)
        op = (v >= v_cut_in) & (v < v_rated)
        P[op] = P_rated * ((v[op] - v_cut_in) / (v_rated - v_cut_in))**3
        P[v >= v_rated] = P_rated
        P[v >= v_cut_out] = 0.0
        return P

    raw_power = power_curve(wind_speed)
    downtime = (rng.random(n) < 0.01).astype(float)
    downtime = np.convolve(downtime, np.ones(3), mode="same")
    downtime = np.clip(downtime, 0, 1)
    noise = rng.normal(0, 80, size=n)
    power_kw = np.clip(raw_power * (1 - 0.9*downtime) + noise, 0, None)

    df = pd.DataFrame({"wind_speed_ms": wind_speed, "power_kw": power_kw}, index=idx)
    df.index.name = "timestamp"
    return df

@dataclass
class ForecastResult:
    forecast: pd.Series
    actual: pd.Series
    mae: float
    rmse: float

# ======== АНАЛІТИКА ========
def structural_analysis(df: pd.DataFrame, series: str = "power_kw", seasonal_period=24):
    stl = STL(df[series], period=seasonal_period, robust=True).fit()

    # останні 30 днів
    last_30 = df.iloc[-24*30:] if len(df) >= 24 else df

    if not last_30.empty:
        plt.figure(); plt.plot(last_30.index, last_30[series])
        plt.title("Power (останні 30 днів)"); plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
        savefig(os.path.join(OUTPUT_DIR, "01_power_last30d.png"))

    plt.figure(); plt.plot(stl.trend.index, stl.trend.values)
    plt.title("STL тренд"); plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "02_stl_trend.png"))

    plt.figure(); plt.plot(stl.seasonal.index, stl.seasonal.values)
    plt.title("STL сезонність (24h)"); plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "03_stl_seasonal.png"))

    plt.figure(); plot_acf(df[series], lags=min(72, len(df)-2))
    plt.title("ACF потужності"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "04_acf_power.png"))

    plt.figure(); plot_pacf(df[series], lags=min(72, len(df)//2 - 1), method="ywm")
    plt.title("PACF потужності"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "05_pacf_power.png"))

def fit_ets_and_forecast(df: pd.DataFrame, series="power_kw", seasonal_periods=24, horizon=24*7) -> ForecastResult:
    if len(df) <= horizon + seasonal_periods * 2:
        raise ValueError("Замало даних для навчання ETS з заданим горизонтом.")

    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    model = ExponentialSmoothing(
        train[series], trend="add", seasonal="add",
        seasonal_periods=seasonal_periods, initialization_method="estimated"
    ).fit()

    fc = model.forecast(horizon)

    mae = float(np.mean(np.abs(test[series].values - fc.values)))
    rmse = safe_rmse(test[series], fc)

    plt.figure()
    plt.plot(train.index[-24*7:], train[series].iloc[-24*7:])
    plt.plot(test.index, test[series])
    plt.plot(test.index, fc)
    plt.title(f"ETS (Holt–Winters) прогноз 7 днів — MAE={mae:.1f} кВт, RMSE={rmse:.1f} кВт")
    plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "06_ets_forecast_7d.png"))

    return ForecastResult(fc, test[series], mae, rmse)

def fit_sarimax_exog_and_forecast(df: pd.DataFrame, y_col="power_kw", exog_cols=("wind_speed_ms",),
                                  horizon=24*7, window_days=120) -> ForecastResult:
    tail = df.iloc[-24*window_days:]
    train = tail.iloc[:-horizon]
    test = tail.iloc[-horizon:]

    model = SARIMAX(
        train[y_col], order=(1,1,1), seasonal_order=(1,1,1,24),
        exog=train[list(exog_cols)], enforce_stationarity=False, enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc_obj = res.get_forecast(steps=horizon, exog=test[list(exog_cols)])
    fc = fc_obj.predicted_mean
    ci = fc_obj.conf_int()

    mae = float(np.mean(np.abs(test[y_col].values - fc.values)))
    rmse = safe_rmse(test[y_col], fc)

    plt.figure()
    plt.plot(train.index[-24*7:], train[y_col].iloc[-24*7:])
    plt.plot(test.index, test[y_col])
    plt.plot(test.index, fc)
    plt.fill_between(test.index, ci[f"lower {y_col}"], ci[f"upper {y_col}"], alpha=0.2)
    plt.title(f"SARIMAX+exog прогноз 7 днів — MAE={mae:.1f} кВт, RMSE={rmse:.1f} кВт")
    plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "07_sarimax_forecast_7d.png"))

    return ForecastResult(fc, test[y_col], mae, rmse)

def cross_correlation(ws: pd.Series, pw: pd.Series, max_lag=48) -> Tuple[int, float, pd.DataFrame]:
    # стандартизуємо
    ws = (ws - ws.mean()) / ws.std()
    pw = (pw - pw.mean()) / pw.std()

    lags = np.arange(-max_lag, max_lag + 1)
    vals = []
    for L in lags:
        if L < 0:
            c = np.corrcoef(ws[-L:].values, pw[:len(pw)+L].values)[0, 1]
        elif L > 0:
            c = np.corrcoef(ws[:len(ws)-L].values, pw[L:].values)[0, 1]
        else:
            c = np.corrcoef(ws.values, pw.values)[0, 1]
        vals.append(c)
    ccf = pd.DataFrame({"lag_h": lags, "corr": vals})

    # найкращий лаг за модулем кореляції
    best_idx = int(np.argmax(np.abs(ccf["corr"])))
    best_lag = int(ccf.iloc[best_idx]["lag_h"])
    best_corr = float(ccf.iloc[best_idx]["corr"])

    # СУМІСНИЙ ВИКЛИК stem: працює в різних версіях matplotlib
    plt.figure()
    try:
        plt.stem(ccf["lag_h"], ccf["corr"], use_line_collection=True)
    except TypeError:
        # старі версії не підтримують use_line_collection
        plt.stem(ccf["lag_h"], ccf["corr"])
    plt.title("Крос-кореляція: швидкість вітру ↔ потужність")
    plt.xlabel("Лаг (год), + означає, що вітер випереджає потужність")
    plt.ylabel("Кореляція")
    plt.tight_layout()

    # збереження у файл (припускаю, що у вас вже є OUTPUT_DIR і savefig)
    savefig(os.path.join(OUTPUT_DIR, "08_cross_correlation.png"))

    return best_lag, best_corr, ccf


def simple_granger(df: pd.DataFrame, maxlag=8):
    try:
        gdf = df[["power_kw", "wind_speed_ms"]].resample("3h").mean().dropna()
        out = grangercausalitytests(gdf[["power_kw", "wind_speed_ms"]], maxlag=maxlag, verbose=False)
        pvals = [(lag, res[0]["ssr_ftest"][1]) for lag, res in out.items()]
        return sorted(pvals, key=lambda x: x[1])[0]  # (lag_best, p_best)
    except Exception:
        return None

# ======== MAIN ========
def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0) Дані
    if CSV_PATH and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = ensure_dt_index(df, "timestamp")
        need = {"wind_speed_ms", "power_kw"}
        if not need.issubset(df.columns):
            raise ValueError("CSV має містити колонки: wind_speed_ms, power_kw")
    else:
        df = simulate_data()

    df.to_csv(os.path.join(OUTPUT_DIR, "00_timeseries.csv"))

    # 1) Структура
    structural_analysis(df, "power_kw", seasonal_period=24)

    # 2) ETS (швидко)
    ets = fit_ets_and_forecast(df, "power_kw", seasonal_periods=24, horizon=FORECAST_HOURS)
    print(f"[ETS] MAE={ets.mae:.2f} кВт; RMSE={ets.rmse:.2f} кВт")

    # 3) SARIMAX (за потреби)
    if RUN_SARIMAX:
        sar = fit_sarimax_exog_and_forecast(
            df, y_col="power_kw", exog_cols=("wind_speed_ms",),
            horizon=FORECAST_HOURS, window_days=SARIMAX_WINDOW_DAYS
        )
        print(f"[SARIMAX+exog] MAE={sar.mae:.2f} кВт; RMSE={sar.rmse:.2f} кВт")

    # 4) Причинність
    best_lag, best_corr, ccf = cross_correlation(df["wind_speed_ms"], df["power_kw"], max_lag=48)
    ccf.to_csv(os.path.join(OUTPUT_DIR, "09_cross_correlation_values.csv"), index=False)
    print(f"Максимальна |крос-кореляція| при лагу {best_lag} год: corr={best_corr:.2f}")

    gr = simple_granger(df, maxlag=8)
    if gr:
        lag_g, p_g = gr
        print(f"Granger (3-год крок): найкращий лаг={lag_g}, p-value={p_g:.4f} "
              "(<0.05 — статистично значимо).")
    else:
        print("Granger: не вдалося обчислити (замало даних або чисельні труднощі).")

    # 5) Згладжування
    df["power_ma24"] = df["power_kw"].rolling(window=24, min_periods=1).mean()
    hw_fit = ExponentialSmoothing(df["power_kw"], trend="add", seasonal="add",
                                  seasonal_periods=24, initialization_method="estimated").fit().fittedvalues
    df["power_hw_fit"] = hw_fit

    tail = df.iloc[-24*7:] if len(df) >= 24*7 else df
    plt.figure()
    plt.plot(tail.index, tail["power_kw"])
    plt.plot(tail.index, tail["power_ma24"])
    plt.plot(tail.index, tail["power_hw_fit"])
    plt.title("Згладжування: Raw vs MA(24h) vs Holt–Winters (ост. 7 днів)")
    plt.xlabel("Час"); plt.ylabel("кВт"); plt.tight_layout()
    savefig(os.path.join(OUTPUT_DIR, "10_smoothing_last7d.png"))

    df.to_csv(os.path.join(OUTPUT_DIR, "11_features_with_smoothing.csv"))
    print(f"Готово! PNG та CSV у папці: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
