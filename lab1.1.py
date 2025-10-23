# python firstsr.py --show

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------- Налаштування виводу --------------------------------
BOX_H = "═" * 72
BOX_S = "─" * 72

# --------------------------- Допоміжні функції ----------------------------------

def strength_ua(r: float) -> str:
    """Лаконічний словесний опис сили зв'язку за |r|.
    Межі: 0–0.2 (дуже слабкий), 0.2–0.4 (слабкий), 0.4–0.6 (помірний),
           0.6–0.8 (сильний), 0.8–1.0 (дуже сильний). Додаємо знак.
    """
    a = abs(r)
    if a < 0.20:
        base = "дуже слабкий"
    elif a < 0.40:
        base = "слабкий"
    elif a < 0.60:
        base = "помірний"
    elif a < 0.80:
        base = "сильний"
    else:
        base = "дуже сильний"
    return f"{base} {'позитивний' if r >= 0 else 'негативний'}"


def normality_ok(x: np.ndarray) -> Tuple[bool, float]:
    """Перевірка нормальності: тест Д'Агостіно K^2 (менш схожий на 'класичне' рішення).
    Повертає (is_normal, p_value)."""
    x = pd.Series(x).dropna().values
    # Якщо розмір вибірки дуже малий, тест слабкий — повертаємо True за замовчуванням
    if x.size < 8:
        return True, 1.0
    k2, p = stats.normaltest(x)
    return p > 0.05, p


def choose_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[str, float, float, float, float]:
    """Обираємо коефіцієнт: якщо обидві змінні ~нормальні — Пірсона, інакше — Спірмена."""
    x = pd.Series(x).astype(float).values
    y = pd.Series(y).astype(float).values
    ok_x, p_x = normality_ok(x)
    ok_y, p_y = normality_ok(y)
    if ok_x and ok_y:
        r, p = stats.pearsonr(x, y)
        method = "Пірсона"
    else:
        r, p = stats.spearmanr(x, y)
        method = "Спірмена"
    return method, float(r), float(p), float(p_x), float(p_y)


def print_summary(title: str, method: str, r: float, p: float) -> None:
    print(f"\n{BOX_H}\n{title}\n{BOX_S}")
    arrow = "↑" if r >= 0 else "↓"
    print(f"Метод кореляції: {method}")
    print(f"Коефіцієнт r: {r:.3f} {arrow}")
    print(f"p-значення: {p:.4g}")
    print(f"Інтерпретація: {strength_ua(r)}")
    print("Висновок: " + ("зв’язок статистично значущий (відхиляємо H0: ρ=0)." if p < 0.05 else "зв’язок НЕ доведений (не відхиляємо H0: ρ=0)."))
    print(BOX_H)


def ensure_out_dir(path: str = "out") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def scatter_and_save(x, y, title: str, fname: str, show: bool) -> None:
    outdir = ensure_out_dir()
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=18, alpha=0.8)
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Графік збережено: {path}")


# --------------------------- Приклади -------------------------------------------

def scene_linear(show: bool = False) -> None:
    print("# Сцена 1: майже лінійна залежність")
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 220)
    y = 0.7 * x + rng.normal(0, 1, 220)
    method, r, p, _, _ = choose_correlation(x, y)
    print_summary("Майже лінійний кейс", method, r, p)
    scatter_and_save(x, y, "Сцена 1 — лінійність", "scene1.png", show)


def scene_monotone(show: bool = False) -> None:
    print("# Сцена 2: монотонний, але нелінійний зв’язок")
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.5, 2.5, 260)
    y = np.exp(0.9 * x) + rng.normal(0, 0.5, x.size)
    method, r, p, _, _ = choose_correlation(x, y)
    print_summary("Нелінійно-монотонний кейс", method, r, p)
    scatter_and_save(x, y, "Сцена 2 — нелінійність", "scene2.png", show)


def scene_kendall(show: bool = False) -> None:
    print("# Сцена 3: Kendall tau-b для рангових даних")
    x = np.array([1, 1, 2, 3, 3, 4, 5, 5, 6])
    y = np.array([2, 3, 1, 2, 4, 5, 4, 6, 7])
    tau, p = stats.kendalltau(x, y)
    print_summary("Kendall tau-b (ранги)", "Кендалла tau-b", float(tau), float(p))
    scatter_and_save(x, y, "Сцена 3 — ранги", "scene3.png", show)


def scene_csv(csv_path: str | None, show: bool = False) -> None:
    print("# Сцена 4: ваш CSV або синтетика")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("CSV повинен містити колонки 'x' та 'y'.")
        x, y = df["x"].values, df["y"].values
        title = "Ваші дані (CSV)"
    else:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 140)
        y = 0.45 * x + rng.normal(0, 1, 140)
        title = "Синтетичні дані (fallback)"
    method, r, p, _, _ = choose_correlation(x, y)
    print_summary(title, method, r, p)
    scatter_and_save(x, y, "Сцена 4 — CSV/синтетика", "scene4.png", show)


def scene_matrix(show: bool = False) -> None:
    print("# Сцена 5: матриця кореляцій (Spearman)")
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 320)
    b = 0.55 * a + rng.normal(0, 1, 320)
    c = np.exp(0.6 * a) + rng.normal(0, 0.55, 320)
    d = rng.normal(0, 1, 320)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    corr = df.corr(method="spearman")
    print(BOX_S)
    print("Оціночна матриця Spearman (|r| виділяє порядки зв'язку):")
    print(corr.round(3))
    print(BOX_S)

    # Теплова карта
    outdir = ensure_out_dir()
    plt.figure(figsize=(5.2, 4.6))
    im = plt.imshow(corr.values, origin="upper")
    plt.xticks(range(corr.shape[1]), corr.columns)
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title("Spearman correlation")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # Підписи значень
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    path = os.path.join(outdir, "scene5_heatmap.png")
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Графік збережено: {path}")


# --------------------------- Точка входу ----------------------------------------

@dataclass
class Args:
    csv: str | None
    show: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Набір прикладів для кореляційного аналізу")
    p.add_argument("--csv", type=str, default=None, help="Шлях до CSV з колонками x,y")
    p.add_argument("--show", action="store_true", help="Показувати графіки на екрані")
    a = p.parse_args()
    return Args(csv=a.csv, show=a.show)


def main() -> None:
    args = parse_args()
    scene_linear(show=args.show)
    scene_monotone(show=args.show)
    scene_kendall(show=args.show)
    scene_csv(csv_path=args.csv, show=args.show)
    scene_matrix(show=args.show)


if __name__ == "__main__":
    main()