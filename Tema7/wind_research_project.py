#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wind_research_project.py
========================
Дослідницький проєкт з "Інтелектуального аналізу даних"
Тема: ЕЛЕКТРИЧНА ПОТУЖНІСТЬ ВІТРОВОЇ ТУРБІНИ

Пайплайн:
1) Завантаження даних (CSV)
2) EDA + якість даних
3) Метрики генерації: AEP/CF, сезонність
4) Weibull (k, lambda) для швидкості вітру
5) Емпірична крива потужності (binned power curve) + індикатор performance
6) Економіка: LCOE, NPV, IRR (спрощена модель)
7) Чутливість + Monte-Carlo
8) Експорт: Excel + Markdown + PNG

Вимоги:
  pip install pandas numpy openpyxl matplotlib

Запуск:
  python wind_research_project.py run --csv wind_fact_hourly.csv --out_dir out_wind
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def df_to_markdown_fallback(df: pd.DataFrame, index: bool = False) -> str:
    """Без tabulate: формує markdown-таблицю вручну."""
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
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def parse_ts(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def infer_site(df: pd.DataFrame) -> pd.DataFrame:
    if "site" in df.columns:
        return df
    if "turbine_id" in df.columns:
        df["site"] = df["turbine_id"].astype(str).str.split("-", n=1).str[0]
        return df
    df["site"] = "Unknown"
    return df


# -------------------------- wind analysis ------------------------------
def weibull_mle(samples: np.ndarray) -> Tuple[float, float]:
    """Оцінка параметрів Weibull(k, lambda) через ітеративну MLE."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 10:
        return (np.nan, np.nan)

    k = 2.0
    for _ in range(30):
        xk = x ** k
        xklnx = xk * np.log(x)
        f = (np.sum(xklnx) / np.sum(xk)) - (np.mean(np.log(x))) - (1.0 / k)
        eps = 1e-4
        k2 = k + eps
        xk2 = x ** k2
        xk2lnx = xk2 * np.log(x)
        f2 = (np.sum(xk2lnx) / np.sum(xk2)) - (np.mean(np.log(x))) - (1.0 / k2)
        dfdk = (f2 - f) / eps
        if dfdk == 0:
            break
        k_new = k - f / dfdk
        if not np.isfinite(k_new) or k_new <= 0:
            break
        if abs(k_new - k) < 1e-6:
            k = k_new
            break
        k = k_new

    lam = (np.mean(x ** k)) ** (1.0 / k)
    return (float(k), float(lam))


def binned_power_curve(df: pd.DataFrame, wind_col="wind_speed_mps", power_col="power_electric_kw",
                       bin_width=1.0, v_max=25.0) -> pd.DataFrame:
    """Емпірична крива потужності: середня потужність у бінaх швидкості."""
    d = df[[wind_col, power_col]].copy()
    d = d[np.isfinite(d[wind_col]) & np.isfinite(d[power_col])]
    bins = np.arange(0, v_max + bin_width, bin_width)
    d["bin"] = pd.cut(d[wind_col], bins=bins, right=False, include_lowest=True)
    out = (d.groupby("bin", observed=True)
           .agg(v_mean=(wind_col, "mean"),
                p_mean=(power_col, "mean"),
                p_p10=(power_col, lambda s: np.nanpercentile(s, 10)),
                p_p90=(power_col, lambda s: np.nanpercentile(s, 90)),
                n=("bin", "size"))
           .reset_index())
    return out


# ---------------------------- finance ----------------------------------
def lcoe(discount_rate: float, years: int, capex: float, opex_yearly: float,
         energy_year1_kwh: float, deg: float = 0.0) -> float:
    """LCOE = PV(costs) / PV(energy)."""
    r = discount_rate
    pv_costs = capex
    pv_energy = 0.0
    for t in range(1, years + 1):
        disc = (1 + r) ** t
        pv_costs += opex_yearly / disc
        e_t = energy_year1_kwh * ((1 - deg) ** (t - 1))
        pv_energy += e_t / disc
    return float(pv_costs / pv_energy) if pv_energy > 0 else float("nan")


def npv_irr(discount_rate: float, years: int, capex: float, opex_yearly: float,
            energy_year1_kwh: float, price_uah_per_kwh: float, deg: float = 0.0) -> Tuple[float, float]:
    """Грошовий потік: -capex + (energy*price - opex)."""
    r = discount_rate
    cash = [-capex]
    for t in range(1, years + 1):
        e_t = energy_year1_kwh * ((1 - deg) ** (t - 1))
        cash.append(e_t * price_uah_per_kwh - opex_yearly)

    # NPV
    npv = 0.0
    for t, cf in enumerate(cash):
        npv += cf / ((1 + r) ** t)

    # IRR (bisection)
    def npv_at(rate: float) -> float:
        s = 0.0
        for t, cf in enumerate(cash):
            s += cf / ((1 + rate) ** t)
        return s

    lo, hi = -0.9, 1.5
    f_lo, f_hi = npv_at(lo), npv_at(hi)
    irr = float("nan")
    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
        for _ in range(80):
            mid = (lo + hi) / 2
            f_mid = npv_at(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        irr = (lo + hi) / 2
    return float(npv), float(irr)


@dataclass
class RunConfig:
    rated_kw: float = 3000.0
    bin_width: float = 1.0
    v_max: float = 25.0
    lifetime_years: int = 20
    discount_rate: float = 0.08
    capex_uah_per_kw: float = 55000.0
    opex_share: float = 0.03
    deg: float = 0.005
    price_uah_per_kwh: float = 6.0


def run_pipeline(csv: Path, out_dir: Path, cfg: RunConfig) -> Dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(csv)
    df = infer_site(parse_ts(df))

    required = {"timestamp", "turbine_id", "wind_speed_mps", "power_electric_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "energy_kwh" not in df.columns:
        # припускаємо 1h інтервал: energy_kwh = power_kW * 1h
        df["energy_kwh"] = df["power_electric_kw"].astype(float)

    q = pd.DataFrame([{
        "n_rows": len(df),
        "n_turbines": df["turbine_id"].nunique(),
        "ts_min": str(df["timestamp"].min()),
        "ts_max": str(df["timestamp"].max()),
        "missing_wind_speed": int(df["wind_speed_mps"].isna().sum()),
        "missing_power": int(df["power_electric_kw"].isna().sum()),
    }])

    per_turbine = (df.groupby(["site", "turbine_id"], as_index=False)
                   .agg(energy_kwh=("energy_kwh", "sum"),
                        power_kw_mean=("power_electric_kw", "mean"),
                        wind_mps_mean=("wind_speed_mps", "mean"),
                        n=("turbine_id", "size")))
    per_turbine["capacity_factor"] = per_turbine["power_kw_mean"] / cfg.rated_kw

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    monthly = (df.groupby(["year", "month", "site"], as_index=False)
               .agg(energy_kwh=("energy_kwh", "sum"),
                    power_kw_mean=("power_electric_kw", "mean"),
                    wind_mps_mean=("wind_speed_mps", "mean")))

    wb_rows = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        k, lam = weibull_mle(g["wind_speed_mps"].to_numpy())
        wb_rows.append({"site": site, "turbine_id": tid, "weibull_k": k, "weibull_lambda": lam})
    weibull_tbl = pd.DataFrame(wb_rows)

    pc_all = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        pc = binned_power_curve(g, bin_width=cfg.bin_width, v_max=cfg.v_max)
        pc["site"] = site
        pc["turbine_id"] = tid
        pc_all.append(pc)
    power_curve_tbl = pd.concat(pc_all, ignore_index=True) if pc_all else pd.DataFrame()

    perf = []
    for (site, tid), g in df.groupby(["site", "turbine_id"]):
        pc = binned_power_curve(g, bin_width=1.0, v_max=25.0)
        pc = pc[np.isfinite(pc["v_mean"])]
        window = pc[(pc["v_mean"] >= 9) & (pc["v_mean"] < 12)]
        p_9_12 = float(window["p_mean"].mean()) if len(window) else float("nan")
        perf.append({"site": site, "turbine_id": tid, "p_mean_9_12": p_9_12,
                     "perf_index_9_12": p_9_12 / cfg.rated_kw if np.isfinite(p_9_12) else np.nan})
    perf_tbl = pd.DataFrame(perf)

    # scaling to annual energy
    hours_per_turb = (df[["timestamp", "turbine_id"]].drop_duplicates()
                      .groupby("turbine_id")["timestamp"].nunique().mean())
    scale = (8760.0 / hours_per_turb) if hours_per_turb and hours_per_turb > 0 else 1.0

    fin_rows = []
    for _, r in per_turbine.iterrows():
        energy_year1 = float(r["energy_kwh"]) * scale
        capex = cfg.capex_uah_per_kw * cfg.rated_kw
        opex = capex * cfg.opex_share
        l = lcoe(cfg.discount_rate, cfg.lifetime_years, capex, opex, energy_year1, cfg.deg)
        npv, irr = npv_irr(cfg.discount_rate, cfg.lifetime_years, capex, opex, energy_year1, cfg.price_uah_per_kwh, cfg.deg)
        fin_rows.append({
            "site": r["site"],
            "turbine_id": r["turbine_id"],
            "energy_year1_kwh_est": energy_year1,
            "capex_uah": capex,
            "opex_yearly_uah": opex,
            "price_uah_per_kwh": cfg.price_uah_per_kwh,
            "LCOE_uah_per_kwh": l,
            "NPV_uah": npv,
            "IRR": irr
        })
    finance_tbl = pd.DataFrame(fin_rows)

    # farm-level sensitivity and MC
    farm_energy_year1 = float(df.groupby(["timestamp"], as_index=False)["energy_kwh"].sum()["energy_kwh"].sum()) * scale
    n_t = df["turbine_id"].nunique()
    base_capex = cfg.capex_uah_per_kw * cfg.rated_kw * n_t
    base_opex = base_capex * cfg.opex_share

    capex_mult = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    price_mult = np.array([0.7, 0.85, 1.0, 1.15, 1.3])

    grid = []
    for cm in capex_mult:
        for pm in price_mult:
            capex = base_capex * cm
            opex = base_opex * cm
            price = cfg.price_uah_per_kwh * pm
            npv, irr = npv_irr(cfg.discount_rate, cfg.lifetime_years, capex, opex, farm_energy_year1, price, cfg.deg)
            grid.append({"capex_mult": cm, "price_mult": pm, "NPV_uah": npv, "IRR": irr})
    sens_tbl = pd.DataFrame(grid)

    rng = np.random.default_rng(42)
    n_mc = 2000
    capex_samples = base_capex * rng.normal(1.0, 0.12, n_mc)
    price_samples = cfg.price_uah_per_kwh * rng.normal(1.0, 0.10, n_mc)
    wind_scale = rng.normal(1.0, 0.08, n_mc)

    npv_list, irr_list = [], []
    for i in range(n_mc):
        capex = max(0.0, capex_samples[i])
        opex = capex * cfg.opex_share
        price = max(0.0, price_samples[i])
        e1 = farm_energy_year1 * max(0.0, wind_scale[i])
        npv, irr = npv_irr(cfg.discount_rate, cfg.lifetime_years, capex, opex, e1, price, cfg.deg)
        npv_list.append(npv)
        irr_list.append(irr)
    mc_tbl = pd.DataFrame({"NPV_uah": npv_list, "IRR": irr_list})
    mc_summary = mc_tbl.describe(percentiles=[0.05, 0.5, 0.95]).reset_index()

    # ------------------ plots ------------------
    plt.figure()
    for site, g in monthly.groupby("site"):
        x = g["month"].astype(int) + (g["year"].astype(int) - g["year"].min()) * 12
        plt.plot(x, g["energy_kwh"], marker="o", label=str(site))
    plt.xlabel("Month index")
    plt.ylabel("Energy (kWh)")
    plt.title("Monthly energy by site")
    plt.legend()
    fig1 = out_dir / "monthly_energy_by_site.png"
    plt.tight_layout()
    plt.savefig(fig1, dpi=150)
    plt.close()

    first_tid = df["turbine_id"].iloc[0]
    g = df[df["turbine_id"] == first_tid]
    pc = binned_power_curve(g, bin_width=1.0, v_max=25.0)
    plt.figure()
    plt.plot(pc["v_mean"], pc["p_mean"], marker="o")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Mean electric power (kW)")
    plt.title(f"Binned power curve: {first_tid}")
    fig2 = out_dir / "power_curve_example.png"
    plt.tight_layout()
    plt.savefig(fig2, dpi=150)
    plt.close()

    plt.figure()
    vals = mc_tbl["NPV_uah"].to_numpy()
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=40)
    plt.xlabel("NPV (UAH)")
    plt.ylabel("Count")
    plt.title("Monte-Carlo: NPV distribution (farm)")
    fig3 = out_dir / "mc_npv_hist.png"
    plt.tight_layout()
    plt.savefig(fig3, dpi=150)
    plt.close()

    # ------------------ exports ------------------
    xlsx = out_dir / "wind_research_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        q.to_excel(w, sheet_name="quality", index=False)
        per_turbine.to_excel(w, sheet_name="per_turbine", index=False)
        perf_tbl.to_excel(w, sheet_name="performance", index=False)
        monthly.to_excel(w, sheet_name="monthly_site", index=False)
        weibull_tbl.to_excel(w, sheet_name="weibull", index=False)
        power_curve_tbl.to_excel(w, sheet_name="power_curve_bins", index=False)
        finance_tbl.to_excel(w, sheet_name="finance", index=False)
        sens_tbl.to_excel(w, sheet_name="sensitivity", index=False)
        mc_summary.to_excel(w, sheet_name="montecarlo_summary", index=False)

    md = out_dir / "wind_research_report.md"
    lines: List[str] = []
    lines.append("# Дослідницький проєкт: електрична потужність вітрової турбіни\n")
    lines.append("## 1) Дані та якість\n")
    lines.append(df_to_markdown_fallback(q, index=False) + "\n")
    lines.append("## 2) Підсумки по турбінах\n")
    lines.append(df_to_markdown_fallback(per_turbine.sort_values(['site','turbine_id']), index=False) + "\n")
    lines.append("## 3) Weibull (вітровий ресурс)\n")
    lines.append(df_to_markdown_fallback(weibull_tbl.sort_values(['site','turbine_id']), index=False) + "\n")
    lines.append("## 4) Performance (індикатор 9–12 м/с)\n")
    lines.append(df_to_markdown_fallback(perf_tbl.sort_values(['site','turbine_id']), index=False) + "\n")
    lines.append("## 5) Економіка (LCOE/NPV/IRR, спрощено)\n")
    lines.append(df_to_markdown_fallback(finance_tbl.sort_values(['site','turbine_id']), index=False) + "\n")
    lines.append("## 6) Sensitivity (CAPEX×price) — приклад 10 рядків\n")
    lines.append(df_to_markdown_fallback(sens_tbl.head(10), index=False) + "\n")
    lines.append("## 7) Monte-Carlo (summary)\n")
    lines.append(df_to_markdown_fallback(mc_summary, index=False) + "\n")
    lines.append("\n### Графіки\n")
    lines.append(f"- {fig1.name}\n- {fig2.name}\n- {fig3.name}\n")
    md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "xlsx": xlsx,
        "md_report": md,
        "plot_monthly": fig1,
        "plot_power_curve": fig2,
        "plot_mc": fig3,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run full research pipeline")
    r.add_argument("--csv", type=Path, required=True)
    r.add_argument("--out_dir", type=Path, default=Path("out_wind"))

    r.add_argument("--rated_kw", type=float, default=3000.0)
    r.add_argument("--capex_uah_per_kw", type=float, default=55000.0)
    r.add_argument("--opex_share", type=float, default=0.03)
    r.add_argument("--price_uah_per_kwh", type=float, default=6.0)
    r.add_argument("--discount_rate", type=float, default=0.08)
    r.add_argument("--lifetime_years", type=int, default=20)
    r.add_argument("--deg", type=float, default=0.005)

    args = ap.parse_args()

    if args.cmd == "run":
        cfg = RunConfig(
            rated_kw=args.rated_kw,
            capex_uah_per_kw=args.capex_uah_per_kw,
            opex_share=args.opex_share,
            price_uah_per_kwh=args.price_uah_per_kwh,
            discount_rate=args.discount_rate,
            lifetime_years=args.lifetime_years,
            deg=args.deg,
        )
        outputs = run_pipeline(args.csv, args.out_dir, cfg)
        print("Saved outputs:")
        for k, v in outputs.items():
            print(f"  {k}: {v.resolve()}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
