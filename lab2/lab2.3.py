import numpy as np
import matplotlib.pyplot as plt

# =========================
# Налаштування експерименту
# =========================
SEED = 42
np.random.seed(SEED)

N_BITS = 64           # довжина хромосоми
POP_SIZE = 80         # розмір популяції
GENERATIONS = 150     # кількість поколінь
RUNS = 20             # скільки незалежних прогонів усереднювати

PX = 0.9              # ймовірність кросоверу (1-point)
PM = 1.0 / N_BITS     # імовірність мутації біта

TOUR_SIZE = 3         # розмір турніру
RANK_S = 1.7          # параметр лінійного рангового відбору (1<s<=2)
THRESH_Q = 0.5        # поріг за квантилем (наприклад, верхні 50%)

# =========================
# Задача та утиліти
# =========================
def init_pop(pop_size, n_bits):
    return np.random.randint(0, 2, size=(pop_size, n_bits), dtype=np.int8)

def fitness_onemax(pop):
    # сума «1» по кожній хромосомі
    return pop.sum(axis=1).astype(np.float64)

def one_point_crossover(p1, p2):
    if np.random.rand() > PX or len(p1) <= 1:
        return p1.copy(), p2.copy()
    cut = np.random.randint(1, len(p1))
    c1 = np.concatenate([p1[:cut], p2[cut:]])
    c2 = np.concatenate([p2[:cut], p1[cut:]])
    return c1, c2

def bit_mutation(child):
    mask = np.random.rand(child.size) < PM
    child[mask] ^= 1
    return child

def hamming_diversity(pop):
    # середня попарна відстань Геммінга, нормована на N_BITS
    # обчислюємо ефективно через частоти «1» по кожному біту
    p = pop.mean(axis=0)                # частка «1» у кожному локусі
    per_locus = 2 * p * (1 - p)         # імовірність різниці в 2 випадкових інд.
    return per_locus.mean()             # у [0, 0.5] для бінарного коду

# =========================
# Оператори відбору
# =========================
def select_proportional(pop, fit):
    # рулетка: ймовірність ~ fitness (додаємо епсилон, щоб уникнути нулів)
    probs = fit - fit.min() + 1e-12
    s = probs.sum()
    if s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    idx = np.random.choice(len(pop), size=len(pop), p=probs, replace=True)
    return pop[idx]

def select_rank(pop, fit, s=RANK_S):
    # лінійний ранговий (Baker): p_i = (2 - s)/N + 2(i-1)(s-1)/(N(N-1))
    order = np.argsort(fit)              # зростання; i=1 — найгірший
    N = len(pop)
    i = np.arange(1, N+1)
    p = (2 - s)/N + (2*(i-1)*(s-1)) / (N*(N-1) + 1e-12)
    p = p / p.sum()
    idx = np.random.choice(N, size=N, p=p, replace=True)
    return pop[order][idx]

def select_tournament(pop, fit, k=TOUR_SIZE):
    N = len(pop)
    winners = []
    for _ in range(N):
        cand = np.random.randint(0, N, size=k)
        best = cand[np.argmax(fit[cand])]
        winners.append(pop[best])
    return np.array(winners, dtype=np.int8)

def select_threshold(pop, fit, q=THRESH_Q):
    # елітний поріг: беремо тих, у кого fitness >= квантиль q, вибираємо з них рівноймовірно
    thr = np.quantile(fit, q)
    elite_idx = np.where(fit >= thr)[0]
    if elite_idx.size == 0:  # на випадок виродження
        elite_idx = np.arange(len(pop))
    idx = np.random.choice(elite_idx, size=len(pop), replace=True)
    return pop[idx]

# мапа імен → функцій відбору
SELECTIONS = {
    "Proportional": select_proportional,
    "Rank":         lambda pop, fit: select_rank(pop, fit, s=RANK_S),
    "Tournament":   lambda pop, fit: select_tournament(pop, fit, k=TOUR_SIZE),
    "Threshold":    lambda pop, fit: select_threshold(pop, fit, q=THRESH_Q),
}

# =========================
# Основний цикл GA (фіксуємо однакові кросовер/мутацію)
# =========================
def run_ga(selection_fn):
    best_hist = []
    mean_hist = []
    div_hist  = []

    pop = init_pop(POP_SIZE, N_BITS)

    for gen in range(GENERATIONS):
        fit = fitness_onemax(pop)
        best_hist.append(fit.max())
        mean_hist.append(fit.mean())
        div_hist.append(hamming_diversity(pop))

        # відбір батьків
        parents = selection_fn(pop, fit)

        # створення нащадків
        children = []
        for i in range(0, POP_SIZE, 2):
            p1 = parents[i]
            p2 = parents[i+1 if i+1 < POP_SIZE else 0]
            c1, c2 = one_point_crossover(p1, p2)
            c1 = bit_mutation(c1)
            c2 = bit_mutation(c2)
            children.extend([c1, c2])
        pop = np.array(children[:POP_SIZE], dtype=np.int8)

    # фінальна оцінка
    fit = fitness_onemax(pop)
    return (np.array(best_hist), np.array(mean_hist), np.array(div_hist),
            fit.max(), fit.mean(), hamming_diversity(pop))

def aggregate_many_runs(selection_fn, runs=RUNS):
    bests, means, divs = [], [], []
    finals = []
    for _ in range(runs):
        bh, mh, dh, fbest, fmean, fdiv = run_ga(selection_fn)
        bests.append(bh); means.append(mh); divs.append(dh)
        finals.append((fbest, fmean, fdiv))
    return (np.mean(bests, axis=0),
            np.mean(means, axis=0),
            np.mean(divs, axis=0),
            np.mean(finals, axis=0))

# =========================
# Запуск експерименту
# =========================
def main():
    curves = {}
    summary = {}

    for name, sel in SELECTIONS.items():
        best_curve, mean_curve, div_curve, finals = aggregate_many_runs(sel, runs=RUNS)
        curves[name] = (best_curve, mean_curve, div_curve)
        summary[name] = {
            "final_best": float(finals[0]),
            "final_mean": float(finals[1]),
            "final_div":  float(finals[2]),
        }
        print(f"[{name:11s}] final best={summary[name]['final_best']:.2f}  "
              f"final mean={summary[name]['final_mean']:.2f}  "
              f"diversity={summary[name]['final_div']:.3f}")

    # --- графік: найкраща придатність ---
    plt.figure(figsize=(7.5,5))
    for name, (best_curve, _, _) in curves.items():
        plt.plot(best_curve, label=name)
    plt.xlabel("Покоління")
    plt.ylabel("Найкраща придатність (avg за runs)")
    plt.title(f"OneMax (n_bits={N_BITS}, pop={POP_SIZE}) — порівняння відборів")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sel_best_curve.png", dpi=150)

    # --- графік: різноманіття ---
    plt.figure(figsize=(7.5,5))
    for name, (_, _, div_curve) in curves.items():
        plt.plot(div_curve, label=name)
    plt.xlabel("Покоління")
    plt.ylabel("Середня Hamming-різноманітність")
    plt.title("Динаміка різноманіття популяції")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sel_diversity_curve.png", dpi=150)

    print("\nЗбережено графіки: sel_best_curve.png, sel_diversity_curve.png")

if __name__ == "__main__":
    main()