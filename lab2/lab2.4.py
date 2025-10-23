import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
random.seed(42)
np.random.seed(42)

# -----------------------------
# Утиліти TSP
# -----------------------------
def gen_tsp(n=50, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, 0.0)
    return pts, D

def tour_length(tour, D):
    return float(np.sum(D[tour, np.roll(tour, -1)]))

def nearest_neighbor(D, start=0):
    n = D.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return np.array(tour, dtype=int)

def two_opt_once(tour, D):
    n = len(tour)
    best_delta = 0.0
    best_i, best_k = None, None
    for i in range(n - 1):
        for k in range(i + 2, n - (0 if i > 0 else 1)):
            a, b = tour[i], tour[(i + 1) % n]
            c, d = tour[k], tour[(k + 1) % n]
            delta = (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])
            if delta < best_delta:
                best_delta = delta
                best_i, best_k = i, k
    if best_delta < 0:
        new_tour = tour.copy()
        new_tour[best_i + 1:best_k + 1] = new_tour[best_i + 1:best_k + 1][::-1]
        return new_tour, True
    return tour, False

def two_opt_local_search(tour, D, max_iters=1000):
    for _ in range(max_iters):
        tour, improved = two_opt_once(tour, D)
        if not improved:
            break
    return tour

# -----------------------------
# 1) ACO (Ant Colony System, стислий)
# -----------------------------
def aco_acs(D, iters=200, ants=None, alpha=1.0, beta=5.0, rho=0.1, q0=0.9, tau0=None):
    n = D.shape[0]
    if ants is None:
        ants = n
    eta = 1.0 / (D + 1e-12)  # евристика
    if tau0 is None:
        # стартовий феромон на базі NN-туру; невеликий мінімум для стабільності
        nn = nearest_neighbor(D, 0)
        tau0 = max(1e-6, ants / tour_length(nn, D))
    tau = np.full((n, n), tau0, dtype=float)
    np.fill_diagonal(tau, 0.0)

    best_tour = None
    best_len = math.inf
    hist = []
    t0 = time.time()

    for it in range(iters):
        tours = []
        lengths = []

        for ant in range(ants):
            start = np.random.randint(0, n)
            tour = [start]
            unvisited = set(range(n))
            unvisited.remove(start)
            cur = start

            while unvisited:
                # ---- ФІКС: фіксуємо множину кандидатів та працюємо лише з нею
                candidates = np.array(list(unvisited), dtype=int)
                desir = (tau[cur, candidates] ** alpha) * (eta[cur, candidates] ** beta)

                # якщо desir "поганий", обираємо рівноймовірно
                if (not np.isfinite(desir).all()) or desir.sum() <= 0:
                    j = np.random.choice(candidates)
                else:
                    if np.random.rand() < q0:
                        j = candidates[np.argmax(desir)]
                    else:
                        p = desir.astype(np.float64)
                        s = p.sum()
                        if s <= 0:
                            j = np.random.choice(candidates)
                        else:
                            p /= s
                            # додаткова нормалізація від чисельних артефактів
                            p /= p.sum()
                            j = np.random.choice(candidates, p=p)

                # локальне оновлення феромону (ACS)
                tau[cur, j] = (1 - rho) * tau[cur, j] + rho * tau0
                tau[j, cur] = tau[cur, j]

                tour.append(j)
                unvisited.remove(j)
                cur = j

            tour = np.array(tour, dtype=int)
            # легкий 2-opt для стабільності
            tour = two_opt_local_search(tour, D, max_iters=5)
            L = tour_length(tour, D)
            tours.append(tour)
            lengths.append(L)

        # оновлення кращого
        idx = int(np.argmin(lengths))
        if lengths[idx] < best_len:
            best_len = lengths[idx]
            best_tour = tours[idx].copy()

        # глобальне оновлення феромону (тільки для найкращого)
        tau *= (1 - rho)
        tl = best_len
        # захист від ділення на 0
        add = rho * (1.0 / max(tl, 1e-12))
        for i in range(n):
            a = best_tour[i]; b = best_tour[(i + 1) % n]
            tau[a, b] += add
            tau[b, a] = tau[a, b]

        hist.append(best_len)

    elapsed = time.time() - t0
    return best_tour, best_len, np.array(hist), elapsed

# -----------------------------
# 2) Генетичний алгоритм (OX + swap mutation + еліта)
# -----------------------------
def ox_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    fill = [g for g in p2 if g not in child]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]; j += 1
    return np.array(child, dtype=int)

def swap_mutation(t, p=0.2):
    if np.random.rand() < p:
        i, j = sorted(np.random.choice(len(t), 2, replace=False))
        t = t.copy()
        t[i], t[j] = t[j], t[i]
    return t

def ga_tsp(D, pop_size=100, iters=200, cx_rate=0.9, mut_rate=0.2, elite=2):
    n = D.shape[0]
    # ініціалізація
    pop = [np.random.permutation(n) for _ in range(pop_size)]
    # додамо один nn-тур
    pop[0] = nearest_neighbor(D, 0)
    t0 = time.time()
    best_hist = []
    best_tour = None
    best_len = math.inf

    for it in range(iters):
        fits = np.array([tour_length(t, D) for t in pop])
        idx = np.argsort(fits)
        pop = [pop[i] for i in idx]
        fits = fits[idx]
        # еліта
        new_pop = pop[:elite]
        if fits[0] < best_len:
            best_len = fits[0]; best_tour = pop[0].copy()
        best_hist.append(best_len)

        # турнірний відбір + кросовер
        while len(new_pop) < pop_size:
            # відбір батьків
            cand = np.random.choice(pop_size//2, size=4, replace=False)
            p1 = pop[cand[np.argmin(fits[cand])]]
            cand = np.random.choice(pop_size//2, size=4, replace=False)
            p2 = pop[cand[np.argmin(fits[cand])]]
            # кросовер
            if np.random.rand() < cx_rate:
                c1 = ox_crossover(p1, p2)
                c2 = ox_crossover(p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            # мутація
            c1 = swap_mutation(c1, mut_rate)
            c2 = swap_mutation(c2, mut_rate)
            # легкий 2-opt для стабільності
            c1 = two_opt_local_search(c1, D, max_iters=2)
            c2 = two_opt_local_search(c2, D, max_iters=2)
            new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]

    elapsed = time.time() - t0
    return best_tour, best_len, np.array(best_hist), elapsed

# -----------------------------
# 3) Simulated Annealing (2-opt сусіди)
# -----------------------------
def sa_tsp(D, iters=2000, T0=1.0, cooling=0.999):
    n = D.shape[0]
    cur = nearest_neighbor(D, 0)
    cur = two_opt_local_search(cur, D, max_iters=50)
    curL = tour_length(cur, D)
    best = cur.copy(); bestL = curL
    hist = []
    t0 = time.time()
    T = T0
    for it in range(iters):
        # випадковий 2-opt крок
        i, j = sorted(np.random.choice(n, 2, replace=False))
        if j == i + 1:
            continue
        cand = cur.copy()
        cand[i+1:j+1] = cand[i+1:j+1][::-1]
        candL = tour_length(cand, D)
        dE = candL - curL
        if dE < 0 or np.random.rand() < math.exp(-dE / max(T, 1e-12)):
            cur, curL = cand, candL
            if curL < bestL:
                best, bestL = cur.copy(), curL
        hist.append(bestL)
        T *= cooling
    elapsed = time.time() - t0
    return best, bestL, np.array(hist), elapsed

# -----------------------------
# 4) 2-opt hill climbing (baseline)
# -----------------------------
def two_opt_baseline(D):
    n = D.shape[0]
    t0 = time.time()
    cur = nearest_neighbor(D, 0)
    cur = two_opt_local_search(cur, D, max_iters=5000)
    L = tour_length(cur, D)
    elapsed = time.time() - t0
    hist = np.array([L])
    return cur, L, hist, elapsed

# -----------------------------
# Експеримент
# -----------------------------
def main():
    pts, D = gen_tsp(n=50, seed=42)

    methods = []

    # ACO
    aco_tour, aco_len, aco_hist, aco_t = aco_acs(D, iters=200, ants=50, alpha=1.0, beta=5.0, rho=0.1, q0=0.9)
    methods.append(("ACO(ACS)", aco_len, aco_t, aco_hist))

    # GA
    ga_tour, ga_len, ga_hist, ga_t = ga_tsp(D, pop_size=120, iters=200, cx_rate=0.9, mut_rate=0.2, elite=2)
    methods.append(("Genetic", ga_len, ga_t, ga_hist))

    # SA
    sa_tour, sa_len, sa_hist, sa_t = sa_tsp(D, iters=4000, T0=1.0, cooling=0.9993)
    methods.append(("SimAnneal", sa_len, sa_t, sa_hist))

    # 2-opt baseline
    b_tour, b_len, b_hist, b_t = two_opt_baseline(D)
    methods.append(("2-opt", b_len, b_t, b_hist))

    # Підсумкова таблиця
    print(f"{'Method':12s} {'Best length':>12s} {'Time(s)':>9s}")
    print("-"*36)
    for name, L, t, _ in methods:
        print(f"{name:12s} {L:12.3f} {t:9.3f}")

    # Графік збіжності
    plt.figure(figsize=(8,5))
    for name, _, _, h in methods:
        x = np.arange(1, len(h)+1)
        plt.plot(x, h, label=name)
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраща довжина туру")
    plt.title("Порівняння ACO vs інші методи на TSP (n=50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tsp_methods_convergence.png", dpi=150)
    print("\nЗбережено графік: tsp_methods_convergence.png")

if __name__ == "__main__":
    main()