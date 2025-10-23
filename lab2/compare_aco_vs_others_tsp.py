import numpy as np

# --- Генерація відстаней ---
n_cities = 8
coords = np.random.rand(n_cities, 2)
dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2))

def tour_length(tour):
    tour = np.array(tour, dtype=int)  # <---- гарантовано індекси int
    return sum(dist[tour[i], tour[(i + 1) % n_cities]] for i in range(n_cities))

# --- 2-opt локальний пошук ---
def two_opt(tour):
    tour = np.array(tour, dtype=int)  # <---- теж примусово цілі
    best = tour.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, n_cities - 2):
            for j in range(i + 1, n_cities):
                if j - i == 1:
                    continue
                new = best.copy()
                new[i:j] = best[j - 1:i - 1:-1]
                if tour_length(new) < tour_length(best):
                    best, improved = new, True
        tour = best
    return best

# --- Простий GA для TSP ---
pop_size, n_gen, pc, pm = 40, 200, 0.9, 0.2
pop = [np.random.permutation(n_cities) for _ in range(pop_size)]

for gen in range(n_gen):
    fitness = np.array([tour_length(p) for p in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    if gen % 50 == 0 or gen == n_gen - 1:
        print(f"Gen {gen:3d} | best L = {fitness[best_idx]:.3f}")

    # Турнірний відбір
    parents = [pop[i] if fitness[i] < fitness[j] else pop[j]
               for i, j in np.random.randint(0, pop_size, (pop_size, 2))]

    # --- OX кросовер ---
    children = []
    for i in range(0, pop_size, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.rand() < pc:
            a, b = sorted(np.random.randint(0, n_cities, 2))
            hole = [x for x in p2 if x not in p1[a:b]]
            c1 = np.concatenate([hole[:a], p1[a:b], hole[a:]])
            hole = [x for x in p1 if x not in p2[a:b]]
            c2 = np.concatenate([hole[:a], p2[a:b], hole[a:]])
        else:
            c1, c2 = p1.copy(), p2.copy()

        # 🔹 ключова правка — одразу приводимо тип
        children += [np.array(c1, dtype=int), np.array(c2, dtype=int)]

    # --- Swap-мутація ---
    for c in children:
        if np.random.rand() < pm:
            i, j = np.random.randint(0, n_cities, 2)
            c[i], c[j] = c[j], c[i]

    # --- Гібридизація з локальним пошуком (2-opt) ---
    pop = [two_opt(np.array(c, dtype=int)) for c in children]

print("\nНайкращий тур:", best)
print("Довжина:", tour_length(best))
