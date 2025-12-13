#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind turbine electric power research project (Tema 7) — Python pipeline.

Features:
- Demo dataset generator (hourly SCADA-like data)
- Data validation & cleaning
- EDA summaries
- Power curve analysis (wind_speed -> power)
- Capacity factor & energy KPIs
- Daily-series correlation between turbines + NetworkX similarity graph
- Traversal demo (BFS/DFS)
- Outputs: Markdown report, Excel workbook, PNG charts

Requirements:
  pip install pandas numpy matplotlib networkx openpyxl

Usage:
  python wind_project_pipeline.py generate --out_csv wind_fact_hourly.csv --days 90
  python wind_project_pipeline.py run --csv wind_fact_hourly.csv --out_dir out_wind --corr_threshold 0.85
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# optional but recommended
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception as e:
    nx = None


# ---------------------------
# Domain model / helpers
# ---------------------------

@dataclass
class TurbineSpec:
    rated_power_kw: float = 3000.0
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0

def power_curve_kw(v_ms: float, spec: TurbineSpec) -> float:
    """Simple piecewise power curve approximation."""
    if np.isnan(v_ms):
        return np.nan
    if v_ms < spec.cut_in_ms:
        return 0.0
    if v_ms >= spec.cut_out_ms:
        return 0.0
    if v_ms >= spec.rated_ms:
        return spec.rated_power_kw
    # cubic ramp between cut-in and rated
    x = (v_ms - spec.cut_in_ms) / (spec.rated_ms - spec.cut_in_ms)
    return spec.rated_power_kw * (x ** 3)

def ensure_non_empty_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(
            f"CSV is empty (0 bytes): {path}. "
            f"Re-generate: `python wind_project_pipeline.py generate --out_csv {path.name}`"
        )

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Demo data generation
# ---------------------------

def generate_demo(days: int, out_csv: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-09-01 00:00:00")
    idx = pd.date_range(start, periods=days * 24, freq="h")

    sites = {
        "Coastal": dict(mu=9.0, sigma=2.2, turbulence=0.12),
        "Mountain": dict(mu=6.5, sigma=2.0, turbulence=0.18),
        "Steppe": dict(mu=7.5, sigma=2.4, turbulence=0.15),
    }
    turbines_per_site = 3
    spec = TurbineSpec()

    rows = []
    for site, p in sites.items():
        for t in range(1, turbines_per_site + 1):
            turbine_id = f"{site}-T{t}"
            # slightly different behavior per turbine
            bias = rng.normal(0, 0.25)
            avail = np.clip(rng.normal(0.96, 0.015), 0.85, 0.995)

            # wind speed (truncate to >=0)
            v = rng.normal(p["mu"] + bias, p["sigma"], size=len(idx))
            v = np.clip(v, 0, None)

            # turbulence-like noise
            v = v * (1.0 + rng.normal(0, p["turbulence"], size=len(idx)))

            # occasional high gusts
            gust_mask = rng.random(len(idx)) < 0.002
            v[gust_mask] = v[gust_mask] + rng.uniform(5, 12, size=gust_mask.sum())

            # power from curve + measurement noise
            power = np.array([power_curve_kw(x, spec) for x in v])
            power = power * avail
            power = power + rng.normal(0, 60, size=len(power))  # metering noise
            power = np.clip(power, 0, spec.rated_power_kw)

            # downtime (set power to 0)
            down_mask = rng.random(len(idx)) < 0.01
            power[down_mask] = 0.0

            energy_kwh = power  # since 1h resolution: kW * 1h = kWh

            df_t = pd.DataFrame(
                {
                    "timestamp": idx,
                    "site": site,
                    "turbine_id": turbine_id,
                    "wind_speed_ms": v,
                    "power_electric_kw": power,
                    "energy_kwh": energy_kwh,
                    "rated_power_kw": spec.rated_power_kw,
                }
            )
            rows.append(df_t)

    df = pd.concat(rows, ignore_index=True)

    # inject a few missing values
    miss_mask = rng.random(len(df)) < 0.002
    df.loc[miss_mask, "wind_speed_ms"] = np.nan

    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False)
    return out_csv


# ---------------------------
# Pipeline steps
# ---------------------------

REQUIRED_COLS = [
    "timestamp",
    "site",
    "turbine_id",
    "wind_speed_ms",
    "power_electric_kw",
    "energy_kwh",
    "rated_power_kw",
]

def load_and_clean(csv: Path) -> pd.DataFrame:
    ensure_non_empty_csv(csv)
    df = pd.read_csv(csv)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # basic sanity rules
    df.loc[df["wind_speed_ms"] < 0, "wind_speed_ms"] = np.nan
    df.loc[df["power_electric_kw"] < 0, "power_electric_kw"] = 0.0
    df.loc[df["energy_kwh"] < 0, "energy_kwh"] = 0.0

    # fill rated power if missing
    df["rated_power_kw"] = df["rated_power_kw"].fillna(df["power_electric_kw"].max())

    # wind speed bins
    bins = [-np.inf, 3, 5, 7, 9, 12, 15, 20, 25, np.inf]
    labels = ["<3", "3-5", "5-7", "7-9", "9-12", "12-15", "15-20", "20-25", ">=25"]
    df["wind_speed_bin"] = pd.cut(df["wind_speed_ms"], bins=bins, labels=labels)

    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    return df

def kpi_capacity_factor(df: pd.DataFrame) -> float:
    # CF = total_energy / (rated_power * total_hours)
    total_energy = df["energy_kwh"].sum()
    rated = df["rated_power_kw"].max()
    hours = df["timestamp"].nunique()  # hourly index count
    return float(total_energy / (rated * hours)) if hours > 0 else float("nan")

def aggregates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # DataFrame.agg() output shape differs across pandas versions; use explicit scalars.
    base = pd.DataFrame([
        {
            "energy_kwh": float(df["energy_kwh"].sum()),
            "power_electric_kw": float(df["power_electric_kw"].mean()),
            "wind_speed_ms": float(df["wind_speed_ms"].mean()),
        }
    ])

    by_site_month = (
        df.groupby(["year", "month", "site"], dropna=False)
        .agg(
            energy_kwh=("energy_kwh", "sum"),
            avg_power_kw=("power_electric_kw", "mean"),
            avg_wind_ms=("wind_speed_ms", "mean"),
        )
        .reset_index()
        .sort_values(["year", "month", "site"])
    )

    by_turbine_month = (
        df.groupby(["year", "month", "turbine_id"], dropna=False)
        .agg(
            energy_kwh=("energy_kwh", "sum"),
            avg_power_kw=("power_electric_kw", "mean"),
            avg_wind_ms=("wind_speed_ms", "mean"),
        )
        .reset_index()
        .sort_values(["year", "month", "turbine_id"])
    )

    wind_bins = (
        df.groupby(["wind_speed_bin"], dropna=False)
        .agg(
            power_electric_kw=("power_electric_kw", "mean"),
            energy_kwh=("energy_kwh", "sum"),
            n=("power_electric_kw", "size"),
        )
        .reset_index()
    )

    cf_by_turbine = (
        df.groupby(["turbine_id"], dropna=False)
        .apply(lambda g: (g["energy_kwh"].sum() / (g["rated_power_kw"].max() * g["timestamp"].nunique())))
        .rename("capacity_factor")
        .reset_index()
        .sort_values("capacity_factor", ascending=False)
    )

    return dict(
        base=base,
        by_site_month=by_site_month,
        by_turbine_month=by_turbine_month,
        wind_bins=wind_bins,
        cf_by_turbine=cf_by_turbine,
    )

def daily_matrix(df: pd.DataFrame, metric: str = "energy_kwh") -> pd.DataFrame:
    pivot = (
        df.groupby(["date", "turbine_id"])[metric]
        .sum()
        .reset_index()
        .pivot(index="date", columns="turbine_id", values=metric)
        .sort_index()
    )
    return pivot

def turbine_similarity_graph(daily: pd.DataFrame, corr_threshold: float) -> Tuple[object, pd.DataFrame]:
    corr = daily.corr(min_periods=max(3, int(0.4 * len(daily))))
    if nx is None:
        return None, corr
    G = nx.Graph()
    for t in corr.columns:
        G.add_node(t)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1 :]:
            c = corr.loc[a, b]
            if pd.notna(c) and c >= corr_threshold:
                G.add_edge(a, b, weight=float(c))
    return G, corr

def graph_stats(G) -> Dict[str, object]:
    if nx is None or G is None:
        return {"note": "NetworkX not available"}
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], dtype=float) if n_nodes else np.array([])
    comps = list(nx.connected_components(G)) if n_nodes else []
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": float(deg.mean()) if len(deg) else 0.0,
        "deg_p50": float(np.percentile(deg, 50)) if len(deg) else 0.0,
        "deg_p90": float(np.percentile(deg, 90)) if len(deg) else 0.0,
        "deg_max": int(deg.max()) if len(deg) else 0,
        "n_components": len(comps),
    }

def traversal_demo(G, source: str) -> Dict[str, object]:
    if nx is None or G is None:
        return {"note": "NetworkX not available"}
    if source not in G:
        source = next(iter(G.nodes)) if G.nodes else source
    bfs_edges = list(nx.bfs_edges(G, source))
    dfs_edges = list(nx.dfs_edges(G, source))
    bfs_tree = nx.bfs_tree(G, source)
    layers = dict(enumerate(nx.bfs_layers(G, source))) if G.number_of_edges() else {0: [source]}
    return {
        "source": source,
        "bfs_edges_count": len(bfs_edges),
        "dfs_edges_count": len(dfs_edges),
        "bfs_tree_nodes": bfs_tree.number_of_nodes(),
        "dfs_tree_nodes": len({source} | {v for _, v in dfs_edges}),
        "bfs_layer_sizes": [len(v) for _, v in layers.items()],
        "edge_bfs_first": bfs_edges[:10],
        "edge_dfs_first": dfs_edges[:10],
    }


# ---------------------------
# Reporting (Markdown/Excel/Charts)
# ---------------------------

def plot_monthly_energy_by_site(by_site_month: pd.DataFrame, out_png: Path) -> None:
    # pivot: rows month, cols site
    by_site_month = by_site_month.copy()
    by_site_month["ym"] = by_site_month["year"].astype(str) + "-" + by_site_month["month"].astype(str).str.zfill(2)
    p = by_site_month.pivot(index="ym", columns="site", values="energy_kwh").sort_index()
    ax = p.plot(kind="bar")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Energy, kWh")
    ax.set_title("Monthly energy by site")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_power_curve(df: pd.DataFrame, out_png: Path) -> None:
    sample = df.sample(n=min(20000, len(df)), random_state=7)
    plt.figure()
    plt.scatter(sample["wind_speed_ms"], sample["power_electric_kw"], s=5, alpha=0.25)
    plt.xlabel("Wind speed, m/s")
    plt.ylabel("Electric power, kW")
    plt.title("Power curve (scatter sample)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_graph(G, out_png: Path) -> None:
    if nx is None or G is None:
        return
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=7) if G.number_of_edges() else nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=9)
    if G.number_of_edges():
        nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.axis("off")
    plt.title("Wind turbine similarity graph (daily correlation)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def write_excel(out_xlsx: Path, tables: Dict[str, pd.DataFrame], corr: pd.DataFrame, gstats: Dict[str, object]) -> None:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        corr.reset_index().to_excel(xw, sheet_name="corr_matrix", index=False)
        pd.DataFrame([gstats]).to_excel(xw, sheet_name="graph_stats", index=False)

def write_markdown(out_md: Path, cfg: Dict[str, object], tables: Dict[str, pd.DataFrame], cf: float,
                   gstats: Dict[str, object], trav: Dict[str, object], artifacts: Dict[str, str]) -> None:
    lines = []
    lines.append("# Дослідницький проєкт (Тема 7): електрична потужність вітрової турбіни\n")
    lines.append("## 1. Мета та питання дослідження\n")
    lines.append("- Як змінюється електрична потужність залежно від швидкості вітру (power curve)?\n")
    lines.append("- Який коефіцієнт використання встановленої потужності (Capacity Factor) для турбін/площадки?\n")
    lines.append("- Чи є групи турбін зі схожою динамікою виробітку (кореляційний граф + BFS/DFS traversal)?\n\n")

    lines.append("## 2. Вхідні дані\n")
    lines.append("Очікувана структура CSV (SCADA-подібні дані): timestamp, site, turbine_id, wind_speed_ms, power_electric_kw, energy_kwh, rated_power_kw.\n\n")
    lines.append("## 3. Налаштування запуску\n")
    for k, v in cfg.items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\n")

    lines.append("## 4. Базові агрегати\n")
    lines.append(tables["base"].to_markdown(index=False) + "\n\n")

    lines.append("## 5. Агрегації по місяцях (приклад)\n")
    lines.append(tables["by_site_month"].head(12).to_markdown(index=False) + "\n\n")

    lines.append("## 6. Аналіз power curve та бінів швидкості\n")
    lines.append(tables["wind_bins"].to_markdown(index=False) + "\n\n")

    lines.append("## 7. Capacity Factor\n")
    lines.append(f"- **CF (весь датасет)** = {cf:.3f}\n\n")

    lines.append("## 8. Граф схожості турбін (кореляція денних рядів)\n")
    lines.append(pd.DataFrame([gstats]).to_markdown(index=False) + "\n\n")
    lines.append("### Traversal (BFS/DFS)\n")
    lines.append(pd.DataFrame([trav]).to_markdown(index=False) + "\n\n")

    lines.append("## 9. Артефакти\n")
    for k, v in artifacts.items():
        lines.append(f"- {k}: `{v}`\n")

    out_md.write_text("".join(lines), encoding="utf-8")

def run_pipeline(csv: Path, out_dir: Path, corr_threshold: float) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    safe_mkdir(out_dir)

    df = load_and_clean(Path(csv))
    tables = aggregates(df)
    cf = kpi_capacity_factor(df)

    # plots
    png_month = out_dir / "monthly_energy_by_site.png"
    png_curve = out_dir / "power_curve_scatter.png"
    plot_monthly_energy_by_site(tables["by_site_month"], png_month)
    plot_power_curve(df, png_curve)

    # graph
    daily = daily_matrix(df, metric="energy_kwh")
    G, corr = turbine_similarity_graph(daily, corr_threshold=corr_threshold)
    gstats = graph_stats(G)
    trav = traversal_demo(G, source=str(df["turbine_id"].iloc[0]) if len(df) else "T1")

    png_graph = out_dir / "turbine_similarity_graph.png"
    plot_graph(G, png_graph)

    out_xlsx = out_dir / "wind_research_results.xlsx"
    write_excel(out_xlsx, tables, corr=corr, gstats=gstats)

    out_md = out_dir / "wind_research_report.md"
    cfg = {"corr_threshold": corr_threshold, "rows": len(df), "turbines": df["turbine_id"].nunique()}
    artifacts = {
        "monthly_energy_by_site.png": png_month.name,
        "power_curve_scatter.png": png_curve.name,
        "turbine_similarity_graph.png": png_graph.name,
        "wind_research_results.xlsx": out_xlsx.name,
    }
    write_markdown(out_md, cfg, tables, cf, gstats, trav, artifacts)

    return {"report_md": out_md, "results_xlsx": out_xlsx, "graph_png": png_graph, "curve_png": png_curve, "month_png": png_month}


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_g = sub.add_parser("generate", help="Generate demo wind dataset (hourly CSV)")
    ap_g.add_argument("--out_csv", required=True, type=str)
    ap_g.add_argument("--days", default=90, type=int)
    ap_g.add_argument("--seed", default=7, type=int)

    ap_r = sub.add_parser("run", help="Run full research pipeline")
    ap_r.add_argument("--csv", required=True, type=str)
    ap_r.add_argument("--out_dir", required=True, type=str)
    ap_r.add_argument("--corr_threshold", default=0.85, type=float)

    args = ap.parse_args()

    if args.cmd == "generate":
        out = generate_demo(days=args.days, out_csv=Path(args.out_csv), seed=args.seed)
        print(f"Saved demo dataset to: {out}")
    elif args.cmd == "run":
        outs = run_pipeline(Path(args.csv), Path(args.out_dir), corr_threshold=args.corr_threshold)
        print("Saved outputs:")
        for k, v in outs.items():
            print(f"- {k}: {v}")
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
