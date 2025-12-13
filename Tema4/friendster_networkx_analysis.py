#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friendster social network + ground-truth communities (SNAP) — NetworkX analysis
=============================================================================
Ціль:
  1) Дослідити набір даних com-Friendster (ребра) та ground-truth communities.
  2) Показати "грунтовний" аналіз, з акцентом на алгоритми обходу (traversal) NetworkX:
     bfs_edges/bfs_tree/bfs_layers/descendants_at_distance,
     dfs_edges/dfs_tree,
     edge_bfs/edge_dfs,
     generic_bfs_edges, bfs_beam_edges.
  3) Додати застосування тих самих ідей до теми:
     "Електрична потужність вітрової турбіни" — побудова графа турбін за схожістю виробітку.

ВАЖЛИВО ПРО РОЗМІР:
  Оригінальний Friendster з SNAP — ~65.6 млн вузлів і ~1.806 млрд ребер.
  NetworkX (pure-Python) НЕ підходить для завантаження всього графа в RAM.
  Тому цей скрипт працює з підвибірками (sampling) і/або індукованими підграфами по спільнотах.

Джерело:
  https://snap.stanford.edu/data/com-Friendster.html

Файли SNAP:
  - com-friendster.ungraph.txt.gz (ребра)
  - com-friendster.top5000.cmty.txt.gz (топ-5000 спільнот)
  - com-friendster.all.cmty.txt.gz (всі спільноти)

Формат (типово для SNAP):
  Текст, рядки-коментарі починаються з '#', далі пари "u v" (через пробіл/таб).

Запуск (приклади):

  # 0) Підготовка: вказати директорію для файлів SNAP
  python friendster_networkx_analysis.py friendster --data_dir ./data

  # 1) Завантажити файли (опційно, великі!):
  python friendster_networkx_analysis.py download --data_dir ./data --which edges top5000

  # 2) Побудувати граф з reservoir sampling по ребрах (напр. 2 млн ребер):
  python friendster_networkx_analysis.py friendster --data_dir ./data --mode edge_sample --k_edges 2000000

  # 3) Аналіз однієї спільноти (індукований підграф):
  python friendster_networkx_analysis.py friendster --data_dir ./data --mode community --community_idx 0 --expand_hops 1

  # 4) Застосування до теми "вітротурбіна": граф турбін за кореляцією потужності/енергії
  python friendster_networkx_analysis.py wind --wind_csv ./sample_wind_turbine_data.csv --metric energy_kwh --freq D --corr_threshold 0.85

Результати:
  Скрипт створює папку ./out (або --out_dir), де зберігає:
    - report.md
    - degree_hist.png
    - bfs_layers.png
    - community_sizes.png (якщо є спільноти)
    - wind_turbine_graph.png (для wind режиму)

"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ---------- Download helpers ----------

SNAP_BASE = "https://snap.stanford.edu/data/bigdata/communities"
FILES = {
    "edges": "com-friendster.ungraph.txt.gz",
    "all": "com-friendster.all.cmty.txt.gz",
    "top5000": "com-friendster.top5000.cmty.txt.gz",
}


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream-download a file (no external progress bar dependencies)."""
    import requests  # local import to keep base deps minimal
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    downloaded += len(part)
                    if total:
                        pct = 100.0 * downloaded / total
                        sys.stdout.write(f"\rDownloading {dest.name}: {pct:5.1f}%")
                        sys.stdout.flush()
    sys.stdout.write("\n")


# ---------- Parsing SNAP text.gz ----------

def iter_edges_from_snap_gz(path: Path) -> Iterator[Tuple[int, int]]:
    """
    Yields edges (u, v) from SNAP .txt.gz file.
    Skips comment lines that start with '#'.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            yield int(parts[0]), int(parts[1])


def read_communities_gz(path: Path, max_lines: Optional[int] = None) -> List[List[int]]:
    """
    Reads communities from SNAP cmty file.
    Each line is a community: space-separated node ids.
    Returns list of communities (each is list[int]).
    """
    comms: List[List[int]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if not line or line.startswith("#"):
                continue
            nodes = [int(x) for x in line.strip().split() if x]
            if nodes:
                comms.append(nodes)
    return comms


# ---------- Sampling strategies ----------

def reservoir_sample_edges(edge_iter: Iterable[Tuple[int, int]], k: int, seed: int = 7) -> List[Tuple[int, int]]:
    """
    Reservoir sampling over stream of edges.
    Complexity: O(E) time, O(k) memory.
    """
    rng = np.random.default_rng(seed)
    sample: List[Tuple[int, int]] = []
    for i, e in enumerate(edge_iter):
        if i < k:
            sample.append(e)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                sample[j] = e
    return sample


def induced_subgraph_from_node_set(edge_path: Path, nodes: Set[int], max_edges: Optional[int] = None) -> nx.Graph:
    """
    Builds induced subgraph on 'nodes' by scanning edge file and keeping edges with both ends in nodes.
    Note: requires streaming the whole edge file (can be long).
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    kept = 0
    for u, v in iter_edges_from_snap_gz(edge_path):
        if u in nodes and v in nodes:
            G.add_edge(u, v)
            kept += 1
            if max_edges is not None and kept >= max_edges:
                break
    return G


def expand_nodes_by_hops(G: nx.Graph, seed_nodes: Set[int], hops: int = 1, cap: int = 200000) -> Set[int]:
    """
    Expands a node set within an already built graph (sample graph) by BFS hops.
    cap prevents blow-up.
    """
    out = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(hops):
        nxt = set()
        for n in list(frontier):
            nxt.update(G.neighbors(n))
            if len(out) + len(nxt) > cap:
                break
        out.update(nxt)
        frontier = nxt
        if len(out) >= cap:
            break
    return out


# ---------- Traversal analysis ----------

def traversal_summary(G: nx.Graph, source: Optional[int] = None, depth_limit: Optional[int] = 3) -> Dict[str, object]:
    """
    Runs a set of traversal algorithms and returns compact stats for reporting.
    """
    if G.number_of_nodes() == 0:
        return {"error": "empty graph"}

    if source is None:
        source = next(iter(G.nodes))

    # BFS (tree edges)
    bfs_edges = list(nx.bfs_edges(G, source=source, depth_limit=depth_limit))
    bfs_tree = nx.bfs_tree(G, source=source, depth_limit=depth_limit)
    layers = [list(layer) for layer in nx.bfs_layers(G, sources=[source])]
    layers = layers[: (depth_limit + 1) if depth_limit is not None else len(layers)]

    # DFS (tree edges)
    dfs_edges = list(nx.dfs_edges(G, source=source, depth_limit=depth_limit))
    dfs_tree = nx.dfs_tree(G, source=source, depth_limit=depth_limit)

    # Edge traversals (may include "back" edges)
    edge_bfs = list(nx.edge_bfs(G, source=source))[: 1000]
    edge_dfs = list(nx.edge_dfs(G, source=source))[: 1000]

    # descendants at exact distance
    dist2 = list(nx.descendants_at_distance(G, source=source, distance=2)) if G.number_of_nodes() > 1 else []

    return {
        "source": source,
        "bfs_edges_count": len(bfs_edges),
        "dfs_edges_count": len(dfs_edges),
        "bfs_tree_nodes": bfs_tree.number_of_nodes(),
        "dfs_tree_nodes": dfs_tree.number_of_nodes(),
        "bfs_layer_sizes": [len(x) for x in layers],
        "desc_at_dist2": len(dist2),
        "edge_bfs_first": edge_bfs[:10],
        "edge_dfs_first": edge_dfs[:10],
    }


def graph_basic_stats(G: nx.Graph) -> Dict[str, object]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], dtype=float) if n else np.array([])
    stats = {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": float(deg.mean()) if n else 0.0,
        "deg_p50": float(np.percentile(deg, 50)) if n else 0.0,
        "deg_p90": float(np.percentile(deg, 90)) if n else 0.0,
        "deg_max": int(deg.max()) if n else 0,
        "n_components": nx.number_connected_components(G) if n else 0,
    }
    # clustering може бути дорогим — беремо вибірково
    if n > 0:
        sample_nodes = list(G.nodes)[: min(5000, n)]
        stats["avg_clustering_sample"] = float(nx.average_clustering(G, nodes=sample_nodes))
    return stats


# ---------- Plotting ----------

def plot_degree_hist(G: nx.Graph, out_path: Path) -> None:
    degs = [d for _, d in G.degree()]
    plt.figure(figsize=(7.2, 3.8))
    plt.hist(degs, bins=60)
    plt.title("Розподіл степенів вузлів (degree histogram)")
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bfs_layers(layer_sizes: Sequence[int], out_path: Path) -> None:
    plt.figure(figsize=(7.2, 3.8))
    plt.plot(range(len(layer_sizes)), layer_sizes, marker="o")
    plt.title("BFS layers: кількість вузлів на відстані k")
    plt.xlabel("k (distance from source)")
    plt.ylabel("nodes in layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_community_sizes(comms: List[List[int]], out_path: Path) -> Dict[str, float]:
    sizes = np.array([len(c) for c in comms], dtype=float)
    plt.figure(figsize=(7.2, 3.8))
    plt.hist(sizes, bins=60)
    plt.title("Розподіл розмірів ground-truth спільнот")
    plt.xlabel("community size")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {
        "n_communities": float(len(comms)),
        "size_mean": float(sizes.mean()) if len(sizes) else 0.0,
        "size_p50": float(np.percentile(sizes, 50)) if len(sizes) else 0.0,
        "size_p90": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
        "size_max": float(sizes.max()) if len(sizes) else 0.0,
    }


# ---------- Wind turbine theme: turbine similarity graph ----------

def build_turbine_similarity_graph(wind_csv: Path, metric: str = "energy_kwh", freq: str = "D", corr_threshold: float = 0.85) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Будує граф турбін за кореляцією часових рядів метрики (наприклад, energy_kwh).
    Вузли: turbine_id
    Ребра: corr >= threshold (вага = corr)
    """
    df = pd.read_csv(wind_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # агрегуємо у часі (за день/годину/тиждень)
    agg = (df.set_index("timestamp")
             .groupby("turbine_id")[metric]
             .resample(freq)
             .sum()
             .reset_index())

    pivot = agg.pivot_table(index="timestamp", columns="turbine_id", values=metric, aggfunc="sum").fillna(0.0)
    corr = pivot.corr()

    G = nx.Graph()
    turbines = list(corr.columns)
    G.add_nodes_from(turbines)
    for i in range(len(turbines)):
        for j in range(i + 1, len(turbines)):
            c = float(corr.iloc[i, j])
            if c >= corr_threshold:
                G.add_edge(turbines[i], turbines[j], weight=c)
    return G, corr


def plot_turbine_graph(G: nx.Graph, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    if G.number_of_nodes() == 0:
        plt.title("Граф турбін порожній (збільшіть дані або зменште поріг corr_threshold)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    pos = nx.spring_layout(G, seed=7)
    nx.draw(G, pos, with_labels=True, node_size=700, font_size=8)
    plt.title("Граф турбін за кореляцією виробітку")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------- Reporting ----------

def write_report_md(out_dir: Path, title: str, sections: List[Tuple[str, str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines.append(f"## {h}")
        lines.append(body.rstrip())
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download SNAP Friendster files")
    d.add_argument("--data_dir", type=str, required=True)
    d.add_argument("--which", nargs="+", choices=["edges", "all", "top5000"], default=["edges", "top5000"])

    f = sub.add_parser("friendster", help="Analyze Friendster with sampling / communities")
    f.add_argument("--data_dir", type=str, required=True)
    f.add_argument("--out_dir", type=str, default="out_friendster")
    f.add_argument("--mode", choices=["edge_sample", "community"], default="edge_sample")
    f.add_argument("--k_edges", type=int, default=500000, help="edges to sample (edge_sample mode)")
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--depth_limit", type=int, default=4)
    f.add_argument("--community_idx", type=int, default=0, help="index in top5000 list (community mode)")
    f.add_argument("--expand_hops", type=int, default=0, help="extra hops expansion using sample graph (community mode)")
    f.add_argument("--max_edges_in_induced", type=int, default=2000000, help="safety cap when building induced subgraph")

    w = sub.add_parser("wind", help="Wind turbine theme: build turbine similarity graph")
    w.add_argument("--wind_csv", type=str, required=True)
    w.add_argument("--out_dir", type=str, default="out_wind")
    w.add_argument("--metric", type=str, default="energy_kwh", choices=["energy_kwh", "power_electric_kw"])
    w.add_argument("--freq", type=str, default="D", help="resample freq (D, H, W, ...)")
    w.add_argument("--corr_threshold", type=float, default=0.85)
    w.add_argument("--depth_limit", type=int, default=4)

    return p.parse_args()


def cmd_download(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for key in args.which:
        name = FILES[key]
        url = f"{SNAP_BASE}/{name}"
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Exists: {dest}")
            continue
        print(f"Downloading from {url}")
        download_file(url, dest)
    print("Done.")


def cmd_friendster(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_path = data_dir / FILES["edges"]
    comm_path = data_dir / FILES["top5000"]

    if not edge_path.exists():
        raise SystemExit(f"Missing edges file: {edge_path}. Run: python ... download --which edges")
    if not comm_path.exists():
        print(f"Warning: Missing community file: {comm_path}. Some analyses will be skipped.")

    sections: List[Tuple[str, str]] = []
    sections.append(("Дані та обмеження",
                     "Оригінальний Friendster дуже великий, тому нижче аналіз виконано на підвибірці/підграфі.\n"
                     "Порада: для більшої точності збільшуйте --k_edges або використовуйте --mode community."))

    # ---- Build analysis graph ----
    if args.mode == "edge_sample":
        print(f"Sampling {args.k_edges} edges via reservoir sampling...")
        edges = reservoir_sample_edges(iter_edges_from_snap_gz(edge_path), k=args.k_edges, seed=args.seed)
        G = nx.Graph()
        G.add_edges_from(edges)
        build_note = f"Побудовано підграф методом reservoir sampling: k_edges={args.k_edges}."
    else:
        if not comm_path.exists():
            raise SystemExit("Community mode requires top5000 cmty file. Run download --which top5000")
        comms = read_communities_gz(comm_path)
        if args.community_idx < 0 or args.community_idx >= len(comms):
            raise SystemExit(f"community_idx out of range: 0..{len(comms)-1}")
        base_nodes = set(comms[args.community_idx])
        print(f"Building induced subgraph for community_idx={args.community_idx}, |C|={len(base_nodes)} ...")
        G_ind = induced_subgraph_from_node_set(edge_path, base_nodes, max_edges=args.max_edges_in_induced)
        G = G_ind
        build_note = f"Індукований підграф за ground-truth спільнотою idx={args.community_idx}, |nodes|={len(base_nodes)}."
        # optionally expand by hops if user already has a sample graph (not available here)
        # Here expansion is not used because we don't have adjacency outside induced graph.

    sections.append(("Побудова підграфа", build_note))

    # ---- Basic stats ----
    stats = graph_basic_stats(G)
    stats_md = "\n".join([f"- **{k}**: {v}" for k, v in stats.items()])
    sections.append(("Базові статистики підграфа", stats_md))

    # ---- Traversal ----
    # choose high-degree node as a "hub" seed if possible
    source = None
    if G.number_of_nodes() > 0:
        source = max(G.degree, key=lambda x: x[1])[0]
    trav = traversal_summary(G, source=source, depth_limit=args.depth_limit)
    trav_md = "\n".join([f"- **{k}**: {v}" for k, v in trav.items() if k != "error"])
    sections.append(("Traversal (BFS/DFS/edge_bfs/edge_dfs)", trav_md))

    # ---- Plots ----
    deg_png = out_dir / "degree_hist.png"
    plot_degree_hist(G, deg_png)

    bfs_png = out_dir / "bfs_layers.png"
    if "bfs_layer_sizes" in trav:
        plot_bfs_layers(trav["bfs_layer_sizes"], bfs_png)

    # ---- Communities overview ----
    if comm_path.exists():
        comms = read_communities_gz(comm_path, max_lines=5000)
        comm_png = out_dir / "community_sizes.png"
        comm_stats = plot_community_sizes(comms, comm_png)
        comm_md = "\n".join([f"- **{k}**: {v}" for k, v in comm_stats.items()])
        sections.append(("Ground-truth communities (top5000): розміри", comm_md))
        sections.append(("Файли (побудовані графіки)",
                         f"- degree_hist.png\n- bfs_layers.png\n- community_sizes.png"))

    report_path = write_report_md(out_dir, "Friendster + NetworkX traversal analysis", sections)
    print(f"\nSaved report: {report_path}")
    print(f"Saved plots: {deg_png}, {bfs_png}")


def cmd_wind(args: argparse.Namespace) -> None:
    wind_csv = Path(args.wind_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wind_csv.exists():
        raise SystemExit(f"Missing wind csv: {wind_csv}")

    G, corr = build_turbine_similarity_graph(
        wind_csv, metric=args.metric, freq=args.freq, corr_threshold=args.corr_threshold
    )

    stats = graph_basic_stats(G)
    source = None
    if G.number_of_nodes() > 0:
        source = next(iter(G.nodes))
    trav = traversal_summary(G, source=source, depth_limit=args.depth_limit)

    # plots
    graph_png = out_dir / "wind_turbine_graph.png"
    plot_turbine_graph(G, graph_png)

    # report
    sections = []
    sections.append(("Ідея",
                     "Вузли = турбіни. Ребра = висока кореляція часових рядів виробітку/потужності.\n"
                     "Traversal (BFS/DFS) дозволяє швидко обходити 'кластер' турбін зі схожою поведінкою."))
    sections.append(("Параметри",
                     f"- metric: {args.metric}\n- freq: {args.freq}\n- corr_threshold: {args.corr_threshold}"))
    sections.append(("Базові статистики графа турбін", "\n".join([f"- **{k}**: {v}" for k, v in stats.items()])))
    sections.append(("Traversal", "\n".join([f"- **{k}**: {v}" for k, v in trav.items() if k != "error"])))
    sections.append(("Візуалізація", "- wind_turbine_graph.png"))

    report_path = write_report_md(out_dir, "Wind turbine power: turbine similarity graph (NetworkX)", sections)
    print(f"Saved report: {report_path}")
    print(f"Saved plot: {graph_png}")


def main() -> None:
    args = parse_args()
    if args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "friendster":
        cmd_friendster(args)
    elif args.cmd == "wind":
        cmd_wind(args)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
