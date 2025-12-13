from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurbineSpec:
    rotor_diameter_m: float = 100.0
    rated_power_kw: float = 3000.0
    cut_in_mps: float = 3.0
    rated_speed_mps: float = 12.0
    cut_out_mps: float = 25.0
    generator_eff: float = 0.92

    @property
    def swept_area_m2(self) -> float:
        r = self.rotor_diameter_m / 2.0
        return float(np.pi * r * r)


def power_curve_kw(v_mps: np.ndarray, spec: TurbineSpec) -> np.ndarray:
    """
    Спрощена електрична крива потужності:
    - 0 нижче cut-in і вище/дорівнює cut-out
    - кубічне наростання між cut-in і rated_speed
    - константа rated_power до cut-out
    """
    v = np.asarray(v_mps, dtype=float)
    p = np.zeros_like(v)

    ci, vr, co = spec.cut_in_mps, spec.rated_speed_mps, spec.cut_out_mps
    pr = spec.rated_power_kw

    # region 1: ci..vr (cubic ramp)
    mask1 = (v >= ci) & (v < vr)
    if np.any(mask1):
        p[mask1] = pr * ((v[mask1]**3 - ci**3) / (vr**3 - ci**3))

    # region 2: vr..co (rated)
    mask2 = (v >= vr) & (v < co)
    p[mask2] = pr

    # below ci or >= co already 0
    return p


def compute_powers(df: pd.DataFrame, spec: TurbineSpec) -> pd.DataFrame:
    """
    Додає показники:
    - power_electric_kw: електрична потужність по кривій
    - energy_kwh: енергія за 1 годину (якщо дані погодинні)
    - capacity_factor: миттєвий коеф. використання встановленої потужності
    """
    out = df.copy()
    out["power_electric_kw"] = power_curve_kw(out["wind_speed_mps"].to_numpy(), spec)
    # Якщо крок часу не 1 година — можна замінити на реальне dt.
    out["energy_kwh"] = out["power_electric_kw"] * 1.0
    out["capacity_factor"] = out["power_electric_kw"] / spec.rated_power_kw
    return out


def direction_sector(deg: float) -> str:
    # 8 секторів (N, NE, E, SE, S, SW, W, NW)
    if np.isnan(deg):
        return "NA"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg % 360) + 22.5) // 45) % 8
    return dirs[idx]


def build_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає ієрархії вимірів:
    - Time: year/quarter/month/day/hour
    - Wind: wind_speed_bin, direction_sector
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=False)

    out["year"] = ts.dt.year
    out["quarter"] = ts.dt.quarter
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    out["hour"] = ts.dt.hour
    out["date"] = ts.dt.date.astype(str)

    # Бінування швидкості вітру (м/с)
    bins = [0, 3, 5, 7, 9, 12, 15, 20, 25, 60]
    labels = ["<3", "3-5", "5-7", "7-9", "9-12", "12-15", "15-20", "20-25", ">=25"]
    out["wind_speed_bin"] = pd.cut(out["wind_speed_mps"], bins=bins, labels=labels, right=False, include_lowest=True)

    if "wind_dir_deg" in out.columns:
        out["wind_dir_sector"] = out["wind_dir_deg"].apply(direction_sector)
    else:
        out["wind_dir_sector"] = "NA"
    return out


def olap_aggregate(
    df: pd.DataFrame,
    dims: List[str],
    measures: Dict[str, str],
) -> pd.DataFrame:
    """
    Загальна агрегація (OLAP drill-up/down):
      dims: список вимірів, по яких групуємо
      measures: {measure_name: aggfunc} наприклад {"energy_kwh": "sum", "power_electric_kw": "mean"}
    """
    if not dims:
        # агрегати "на весь куб"
        return df.agg(measures).to_frame().T

    grouped = df.groupby(dims, dropna=False).agg(measures).reset_index()
    return grouped


def slice_cube(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Зріз (slice): фіксуємо значення вимірів.
    Приклад: slice_cube(df, site="Steppe", year=2025, month=11)
    """
    out = df
    for k, v in filters.items():
        out = out[out[k] == v]
    return out


def rotate(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Обертання (rotate): міняємо місцями осі для двовимірного подання."""
    return pivot_df.T


def demo(df: pd.DataFrame, spec: TurbineSpec) -> None:
    print("\n=== DEMO: базові агрегати ===")
    total = olap_aggregate(df, dims=[], measures={"energy_kwh": "sum", "power_electric_kw": "mean"})
    print(total)

    print("\n=== Drill-up: енергія (kWh) по місяцях і турбінах ===")
    by_month = olap_aggregate(df, dims=["year", "month", "turbine_id"], measures={"energy_kwh": "sum"})
    print(by_month.head(12))

    print("\n=== Pivot (2D) + Rotate: month x turbine_id ===")
    piv = by_month.pivot_table(index=["year", "month"], columns="turbine_id", values="energy_kwh", aggfunc="sum")
    print(piv.head())
    print("\n--- rotated ---")
    print(rotate(piv).iloc[:, :5])

    print("\n=== Slice: тільки site='Steppe' і year=max ===")
    y = int(df["year"].max())
    sliced = slice_cube(df, site="Steppe", year=y)
    by_day = olap_aggregate(sliced, dims=["date"], measures={"energy_kwh": "sum", "power_electric_kw": "mean"})
    print(by_day.head(10))

    print("\n=== Аналіз за швидкісними бінaми (wind_speed_bin) ===")
    by_bin = olap_aggregate(df, dims=["wind_speed_bin"], measures={"power_electric_kw": "mean", "energy_kwh": "sum"})
    print(by_bin)

    # Загальний CF
    hours = df.shape[0]
    total_energy = float(df["energy_kwh"].sum())
    max_energy = spec.rated_power_kw * hours
    cf = total_energy / max_energy if max_energy > 0 else np.nan
    print(f"\n=== Capacity factor (по всьому датасету) ===\nCF = {cf:.3f}")


def generate_demo_dataset(days: int = 90, seed: int = 7) -> pd.DataFrame:
    """
    Генерує погодинні дані для 3 майданчиків і 3 турбін на кожному.
    Швидкість вітру ~ Weibull (типово для вітру), з різними параметрами по сайтах.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-09-01 00:00:00")
    idx = pd.date_range(start, periods=days * 24, freq="H")

    sites = {
        "Coastal": (2.2, 9.0),   # (k, lambda) - сильніший вітер
        "Steppe":  (2.0, 7.5),
        "Mountain":(1.8, 6.5)    # більш турбулентний/нерівний
    }
    turbines = ["T1", "T2", "T3"]

    rows = []
    for site, (k, lam) in sites.items():
        for t in turbines:
            # Weibull: v = lam * (-ln(1-u))^(1/k)
            u = rng.random(len(idx))
            v = lam * (-np.log(1 - u)) ** (1 / k)

            # вітер по напрямку
            wd = (rng.normal(loc=210 if site=="Coastal" else 180, scale=60, size=len(idx)) % 360)
            df_site = pd.DataFrame({
                "timestamp": idx.astype(str),
                "site": site,
                "turbine_id": f"{site}-{t}",
                "wind_speed_mps": v,
                "wind_dir_deg": wd,
            })
            rows.append(df_site)

    df = pd.concat(rows, ignore_index=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wind turbine OLAP cube in pandas")
    p.add_argument("--input", type=str, default=None, help="CSV з даними (timestamp, site, turbine_id, wind_speed_mps, ...)")
    p.add_argument("--generate", action="store_true", help="Згенерувати демонстраційний датасет")
    p.add_argument("--days", type=int, default=90, help="К-сть днів для генерації (якщо --generate)")
    p.add_argument("--out_csv", type=str, default=None, help="Куди зберегти згенерований CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = TurbineSpec()

    if args.generate:
        df = generate_demo_dataset(days=args.days)
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"Saved demo dataset to: {args.out_csv}")
    else:
        if not args.input:
            raise SystemExit("Потрібно або --generate, або --input <file.csv>")
        df = pd.read_csv(args.input)

    required = {"timestamp", "site", "turbine_id", "wind_speed_mps"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"У CSV відсутні колонки: {sorted(missing)}")

    df = build_dimensions(df)
    df = compute_powers(df, spec)

    demo(df, spec)


if __name__ == "__main__":
    main()
