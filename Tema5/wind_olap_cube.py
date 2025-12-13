#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wind_olap_cube.py
-----------------
Python-реалізація "OLAP-куба" для теми: електрична потужність вітрової турбіни.

Мета: повторити логіку лабораторної з icCube (фактова таблиця + виміри + ієрархії + міри),
але зробити це на Python (pandas + SQLite) і показати OLAP-операції:
- Drill-up / Drill-down
- Slice
- Rotate (pivot)
- Агрегації по рівнях ієрархій Time/Region/Turbine/Wind

Файли:
- wind_fact_hourly.csv      (фактова таблиця в CSV)
- wind_farm_olap.db         (SQLite зі схемою "зірка": dim_* + fact_generation)
- wind_olap_cube.xlsx       (Excel: факт + кілька "зведених" таблиць)
- wind_olap_demo_report.md  (markdown з результатами демо-запитів)

Приклад:
  python wind_olap_cube.py generate --days 120
  python wind_olap_cube.py build_db
  python wind_olap_cube.py demo --corr_threshold 0.75
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------- Power curve (simplified) -----------------------
@dataclass(frozen=True)
class TurbineSpec:
    rated_power_kw: float = 3000.0
    cut_in_mps: float = 3.0
    rated_speed_mps: float = 12.0
    cut_out_mps: float = 25.0


def power_curve_kw(v_mps: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    """Спрощена крива потужності вітротурбіни."""
    v = np.asarray(v_mps, dtype=float)
    p = np.zeros_like(v)
    ci, vr, co = spec.cut_in_mps, spec.rated_speed_mps, spec.cut_out_mps
    pr = spec.rated_power_kw
    mask1 = (v >= ci) & (v < vr)
    if np.any(mask1):
        p[mask1] = pr * ((v[mask1] ** 3 - ci ** 3) / (vr ** 3 - ci ** 3))
    mask2 = (v >= vr) & (v < co)
    p[mask2] = pr
    return p


def _weibull(rng: np.random.Generator, k: float, lam: float, n: int) -> np.ndarray:
    """Вибірка швидкостей вітру ~ Weibull(k, lam)."""
    u = rng.random(n)
    return lam * (-np.log(1 - u)) ** (1 / k)


def generate_fact_csv(
    out_csv: Path,
    start: str = "2025-01-01 00:00:00",
    days: int = 120,
    seed: int = 7,
) -> Path:
    """
    Генерує синтетичну фактову таблицю (погодинні записи) для 3 майданчиків × 3 турбіни.
    Вихід: CSV з полями часу, регіону, турбіни, швидкості вітру, потужності, енергії, тарифу, доходу.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")

    # параметри "регіонів" (майданчиків)
    sites = {
        "Coastal": (2.2, 9.0),    # вітряніше
        "Steppe":  (2.0, 7.5),
        "Mountain":(1.8, 6.5),
    }
    turbines = ["T1", "T2", "T3"]
    models = {"T1": "3MW-A", "T2": "3MW-B", "T3": "3MW-C"}

    spec = TurbineSpec(rated_power_kw=3000.0)

    rows = []
    for site, (k, lam) in sites.items():
        for t in turbines:
            v = _weibull(rng, k, lam, len(idx))
            wind_dir = (rng.normal(loc=210 if site == "Coastal" else 180, scale=60, size=len(idx)) % 360)
            p_kw = power_curve_kw(v, spec)

            # простий "ринковий" тариф: EUR/MWh
            base_tariff = 75 if site == "Coastal" else (70 if site == "Steppe" else 68)
            tariff_eur_mwh = base_tariff + rng.normal(0, 3, len(idx))

            # курс EUR->UAH (коливається)
            eur_uah = 41 + rng.normal(0, 0.7, len(idx))

            # енергія за 1 годину (kWh) = kW * 1h
            energy_kwh = p_kw * 1.0

            # дохід у грн: (kWh -> MWh) * тариф(EUR/MWh) * курс
            revenue_uah = (energy_kwh / 1000.0) * tariff_eur_mwh * eur_uah

            df = pd.DataFrame({
                "timestamp": idx,
                "site": site,
                "turbine_id": f"{site}-{t}",
                "turbine_model": models[t],
                "wind_speed_mps": v,
                "wind_dir_deg": wind_dir,
                "power_electric_kw": p_kw,
                "energy_kwh": energy_kwh,
                "tariff_eur_mwh": tariff_eur_mwh,
                "eur_uah": eur_uah,
                "revenue_uah": revenue_uah,
            })

            rows.append(df)

    data = pd.concat(rows, ignore_index=True)

    # time attributes (Time dimension hierarchy: Year->Quarter->Month->Day->Hour)
    ts = pd.to_datetime(data["timestamp"])
    data["year"] = ts.dt.year
    data["quarter"] = ts.dt.quarter
    data["month"] = ts.dt.month
    data["day"] = ts.dt.date.astype(str)
    data["hour"] = ts.dt.hour

    # wind speed bins (Wind dimension)
    bins = [-np.inf, 3, 5, 7, 9, 12, 15, 20, 25, np.inf]
    labels = ["<3", "3-5", "5-7", "7-9", "9-12", "12-15", "15-20", "20-25", ">=25"]
    data["wind_speed_bin"] = pd.cut(data["wind_speed_mps"], bins=bins, labels=labels, right=False)

    data.to_csv(out_csv, index=False)
    return out_csv


# ----------------------- Build "star schema" in SQLite -----------------------
def build_sqlite_star_schema(csv_path: Path, db_path: Path) -> Path:
    """
    Створює SQLite DB зі схемою "зірка":
      dim_time, dim_region, dim_turbine, dim_wind, fact_generation
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # dim_time (grain = hour)
    dim_time = df[["timestamp", "year", "quarter", "month", "day", "hour"]].drop_duplicates().copy()
    dim_time = dim_time.sort_values("timestamp").reset_index(drop=True)
    dim_time["time_id"] = np.arange(1, len(dim_time) + 1)

    # dim_region
    dim_region = df[["site"]].drop_duplicates().sort_values("site").reset_index(drop=True)
    dim_region["region_id"] = np.arange(1, len(dim_region) + 1)

    # dim_turbine
    dim_turbine = df[["turbine_id", "turbine_model", "site"]].drop_duplicates().sort_values(["site", "turbine_id"]).reset_index(drop=True)
    dim_turbine["turbine_key"] = np.arange(1, len(dim_turbine) + 1)

    # dim_wind
    dim_wind = df[["wind_speed_bin"]].drop_duplicates().sort_values("wind_speed_bin").reset_index(drop=True)
    dim_wind["wind_id"] = np.arange(1, len(dim_wind) + 1)

    # map keys
    df = df.merge(dim_time[["timestamp", "time_id"]], on="timestamp", how="left")
    df = df.merge(dim_region[["site", "region_id"]], on="site", how="left")
    df = df.merge(dim_turbine[["turbine_id", "turbine_key"]], on="turbine_id", how="left")
    df = df.merge(dim_wind[["wind_speed_bin", "wind_id"]], on="wind_speed_bin", how="left")

    fact = df[[
        "time_id", "region_id", "turbine_key", "wind_id",
        "wind_speed_mps", "wind_dir_deg",
        "power_electric_kw", "energy_kwh",
        "tariff_eur_mwh", "eur_uah", "revenue_uah"
    ]].copy()

    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        dim_time.to_sql("dim_time", con, index=False)
        dim_region.to_sql("dim_region", con, index=False)
        dim_turbine.to_sql("dim_turbine", con, index=False)
        dim_wind.to_sql("dim_wind", con, index=False)
        fact.to_sql("fact_generation", con, index=False)

        # indexes for faster joins
        cur = con.cursor()
        cur.executescript("""
        CREATE INDEX idx_fact_time ON fact_generation(time_id);
        CREATE INDEX idx_fact_region ON fact_generation(region_id);
        CREATE INDEX idx_fact_turbine ON fact_generation(turbine_key);
        CREATE INDEX idx_fact_wind ON fact_generation(wind_id);
        """)
        con.commit()
    finally:
        con.close()

    return db_path


# ----------------------- OLAP queries (pandas) -----------------------
def load_star(db_path: Path) -> pd.DataFrame:
    """Завантажує fact + joins (denormalized view) у pandas."""
    con = sqlite3.connect(db_path)
    try:
        fact = pd.read_sql_query("SELECT * FROM fact_generation", con)
        dim_time = pd.read_sql_query("SELECT * FROM dim_time", con)
        dim_region = pd.read_sql_query("SELECT * FROM dim_region", con)
        dim_turbine = pd.read_sql_query("SELECT * FROM dim_turbine", con)
        dim_wind = pd.read_sql_query("SELECT * FROM dim_wind", con)
    finally:
        con.close()

    df = (fact
          .merge(dim_time, on="time_id", how="left")
          .merge(dim_region, on="region_id", how="left")
          .merge(dim_turbine, on="turbine_key", how="left")
          .merge(dim_wind, on="wind_id", how="left"))

    # після merge маємо site_x (з dim_region) та site_y (з dim_turbine)
    if "site_x" in df.columns:
        df = df.rename(columns={"site_x": "site"})
    if "site_y" in df.columns:
        df = df.drop(columns=["site_y"])

    return df


def cube_groupby(
    df: pd.DataFrame,
    dims: List[str],
    measures: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Аналог "кубового" запиту: groupby по dims, агрегації measures.
    measures: dict {col: agg}, напр. {"energy_kwh":"sum","revenue_uah":"sum","power_electric_kw":"mean"}
    """
    if measures is None:
        measures = {"energy_kwh": "sum", "revenue_uah": "sum", "power_electric_kw": "mean"}

    # Якщо dims порожній список — робимо "глобальну" агрегацію по всьому датасету
    if not dims:
        out = df.agg(measures).to_frame().T
        return out.reset_index(drop=True)

    grouped = df.groupby(dims, dropna=False).agg(measures).reset_index()
    return grouped


def pivot_rotate(df: pd.DataFrame, index: List[str], columns: List[str], values: str, aggfunc: str = "sum") -> pd.DataFrame:
    """Rotate: pivot_table (2D візуалізація)."""
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc, fill_value=0.0)



def df_to_markdown_fallback(df: pd.DataFrame, index: bool = False) -> str:
    """
    Безпечне перетворення DataFrame -> Markdown таблиця.
    pandas.DataFrame.to_markdown() потребує optional dependency 'tabulate'.
    Якщо tabulate не встановлено — робимо просту markdown-таблицю вручну.
    """
    try:
        return df.to_markdown(index=index)  # type: ignore[attr-defined]
    except Exception:
        df2 = df.copy()
        if index:
            df2 = df2.reset_index()
        cols = [str(c) for c in df2.columns.tolist()]
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df2.iterrows():
            vals = []
            for v in r.tolist():
                if isinstance(v, float):
                    # компактний запис для float
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

# ----------------------- Demo + report export -----------------------
def demo(db_path: Path, out_md: Path, out_xlsx: Path) -> None:
    df = load_star(db_path)

    # базові агрегати
    base = cube_groupby(df, dims=[], measures={"energy_kwh": "sum", "revenue_uah": "sum", "power_electric_kw": "mean"})
    # drill-up: енергія по Year/Quarter/Site
    drill = cube_groupby(df, dims=["year", "quarter", "site"], measures={"energy_kwh": "sum", "revenue_uah": "sum"}).sort_values(["year","quarter","site"])
    # rotate: quarter x site (energy)
    rot = pivot_rotate(df, index=["year", "quarter"], columns=["site"], values="energy_kwh", aggfunc="sum")
    # slice: тільки Steppe, year=max
    y_max = int(df["year"].max())
    slice_df = df[(df["site"] == "Steppe") & (df["year"] == y_max)]
    slice_daily = (slice_df
                   .assign(date=pd.to_datetime(slice_df["timestamp"]).dt.date.astype(str))
                   .groupby(["date"], as_index=False)[["energy_kwh","revenue_uah"]].sum()
                   .sort_values("date")
                   .head(10))
    # wind bins analysis
    wind_bins = cube_groupby(df, dims=["wind_speed_bin"], measures={"power_electric_kw":"mean","energy_kwh":"sum"}).sort_values("wind_speed_bin")

    # capacity factor (за rated 3000 kW)
    rated_kw = 3000.0
    hours = df["timestamp"].nunique()  # because dim_time unique per hour
    # Actually timestamp repeats per turbine; so use time dimension count
    hours = df[["time_id"]].drop_duplicates().shape[0]
    n_turbines = df[["turbine_id"]].drop_duplicates().shape[0]
    cf = float(base["energy_kwh"].iloc[0] / (rated_kw * hours * n_turbines))

    # export markdown report
    lines = []
    lines.append("# OLAP cube (Python) для вітрових турбін\n")
    lines.append("Цей звіт згенеровано скриптом `wind_olap_cube.py`.\n")
    lines.append("## Базові агрегати (весь датасет)\n")
    lines.append(df_to_markdown_fallback(base, index=False) + "\n")
    lines.append("## Drill-up: по роках/кварталах/майданчиках\n")
    lines.append(df_to_markdown_fallback(drill.head(20), index=False) + "\n")
    lines.append("## Rotate (pivot): quarter × site (energy_kwh)\n")
    lines.append(df_to_markdown_fallback(rot, index=True) + "\n")
    lines.append("## Slice: site='Steppe', year=max (перші 10 днів)\n")
    lines.append(df_to_markdown_fallback(slice_daily, index=False) + "\n")
    lines.append("## Аналіз за бінaми швидкості вітру\n")
    lines.append(df_to_markdown_fallback(wind_bins, index=False) + "\n")
    lines.append(f"## Capacity factor\nCF = {cf:.3f}\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # export Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # raw (small sample to not explode)
        df.sort_values("timestamp").head(5000).to_excel(writer, sheet_name="fact_sample", index=False)
        base.to_excel(writer, sheet_name="agg_base", index=False)
        drill.to_excel(writer, sheet_name="drill_qtr_site", index=False)
        rot.to_excel(writer, sheet_name="pivot_qtr_x_site")
        slice_daily.to_excel(writer, sheet_name="slice_steppe_daily", index=False)
        wind_bins.to_excel(writer, sheet_name="wind_bins", index=False)


# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic hourly fact CSV for wind turbines")
    g.add_argument("--out_csv", type=Path, default=Path("wind_fact_hourly.csv"))
    g.add_argument("--start", type=str, default="2025-01-01 00:00:00")
    g.add_argument("--days", type=int, default=120)
    g.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("build_db", help="Build SQLite star schema from fact CSV")
    b.add_argument("--csv", type=Path, default=Path("wind_fact_hourly.csv"))
    b.add_argument("--db", type=Path, default=Path("wind_farm_olap.db"))

    d = sub.add_parser("demo", help="Run demo OLAP queries + export report.md and xlsx")
    d.add_argument("--db", type=Path, default=Path("wind_farm_olap.db"))
    d.add_argument("--out_md", type=Path, default=Path("wind_olap_demo_report.md"))
    d.add_argument("--out_xlsx", type=Path, default=Path("wind_olap_cube.xlsx"))

    args = ap.parse_args()

    if args.cmd == "generate":
        out = generate_fact_csv(args.out_csv, start=args.start, days=args.days, seed=args.seed)
        print(f"Saved: {out.resolve()}")
    elif args.cmd == "build_db":
        out = build_sqlite_star_schema(args.csv, args.db)
        print(f"Saved DB: {out.resolve()}")
    elif args.cmd == "demo":
        demo(args.db, args.out_md, args.out_xlsx)
        print(f"Saved report: {args.out_md.resolve()}")
        print(f"Saved xlsx:   {args.out_xlsx.resolve()}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
