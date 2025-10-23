import numpy as np


# --- Цільова функція (мінімізація) ---
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


# --- Параметри ---
n_dim = 5  # розмірність задачі
pop_size = 40  # розмір популяції
n_gen = 200  # кількість поколінь
pc, pm = 0.9, 0.2  # ймовірності кросоверу і мутації
bounds = (-5.12, 5.12)

# --- Ініціалізація ---
pop = np.random.uniform(bounds[0], bounds[1], (pop_size, n_dim))

# --- Основний цикл ---
for gen in range(n_gen):
    # Оцінка пристосованості
    fitness = np.array([rastrigin(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    if gen % 20 == 0 or gen == n_gen - 1:
        print(f"Gen {gen:3d} | best f = {fitness[best_idx]:.4f}")

    # Відбір (турнір)
    parents = []
    for _ in range(pop_size):
        i, j = np.random.randint(0, pop_size, 2)
        parents.append(pop[i] if fitness[i] < fitness[j] else pop[j])
    parents = np.array(parents)

    # Кросовер (однорідний)
    children = []
    for i in range(0, pop_size, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.rand() < pc:
            mask = np.random.rand(n_dim) < 0.5
            c1, c2 = np.where(mask, p1, p2), np.where(mask, p2, p1)
        else:
            c1, c2 = p1.copy(), p2.copy()
        children.extend([c1, c2])

    # Мутація (додавання випадкового шуму)
    for child in children:
        if np.random.rand() < pm:
            idx = np.random.randint(0, n_dim)
            child[idx] += np.random.normal(0, 0.5)
            # обмежимо в межах допустимих значень
            child[idx] = np.clip(child[idx], bounds[0], bounds[1])

    pop = np.array(children)

print("\nНайкраще знайдене рішення:")
print("x* =", best)
print("f(x*) =", rastrigin(best))