import numpy as np

def fitness(ind):  # максимізує кількість одиниць
    return np.sum(ind)

n_bits, pop_size, n_gen = 50, 40, 100
pc, pm = 0.8, 0.02

pop = np.random.randint(0, 2, (pop_size, n_bits))

for gen in range(n_gen):
    fit = np.array([fitness(ind) for ind in pop])
    best = pop[np.argmax(fit)]
    if gen % 10 == 0 or gen == n_gen - 1:
        print(f"Gen {gen:3d} | best = {fit.max()} | mean = {fit.mean():.2f}")

    # Турнірний відбір
    parents = [pop[i] if fit[i] > fit[j] else pop[j]
               for i, j in np.random.randint(0, pop_size, (pop_size, 2))]

    # Одноточковий кросовер
    children = []
    for i in range(0, pop_size, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.rand() < pc:
            cx = np.random.randint(1, n_bits - 1)
            c1 = np.concatenate([p1[:cx], p2[cx:]])
            c2 = np.concatenate([p2[:cx], p1[cx:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        children += [c1, c2]

    # Мутація — фліп бітів
    for c in children:
        for k in range(n_bits):
            if np.random.rand() < pm:
                c[k] ^= 1

    pop = np.array(children)
