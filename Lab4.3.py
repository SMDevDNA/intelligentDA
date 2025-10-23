import numpy as np

# Генеруємо 3 кластери у 2D-просторі
rng = np.random.default_rng(7)
centers = np.array([[1, 1], [6, 4], [-4, 5]])

X = np.vstack([
    rng.normal(centers[0], 0.7, size=(150, 2)),
    rng.normal(centers[1], 1.0, size=(200, 2)),
    rng.normal(centers[2], 0.8, size=(180, 2)),
])

def init_kmeans_pp(X, k, rng):
    n = len(X)
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(n)]
    for i in range(1, k):
        d2 = np.min(np.sum((X[:, None] - C[None, :i])**2, axis=2), axis=1)
        p = d2 / np.sum(d2)
        C[i] = X[rng.choice(n, p=p)]
    return C

def kmeans(X, k=3, max_iter=100, tol=1e-4, rng=None):
    rng = rng or np.random.default_rng()
    C = init_kmeans_pp(X, k, rng)
    for _ in range(max_iter):
        # Призначення до найближчого центроїда
        d2 = np.sum((X[:, None] - C[None, :])**2, axis=2)
        labels = np.argmin(d2, axis=1)
        # Оновлення центроїдів
        newC = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        shift = np.linalg.norm(newC - C)
        C = newC
        if shift < tol:
            break
    inertia = np.sum((X - C[labels])**2)
    return C, labels, inertia

C, labels, inertia = kmeans(X, k=3, rng=rng)
print("Центроїди кластерів:\n", C)
print("Сума квадратів відстаней:", inertia)


class MiniSOM:
    def __init__(self, m, n, dim, lr=0.4, sigma=None, rng=None):
        self.m, self.n, self.dim = m, n, dim
        self.lr0 = lr
        self.sigma0 = sigma or max(m, n) / 2
        self.rng = rng or np.random.default_rng()
        self.W = self.rng.normal(0, 1, size=(m, n, dim))
        y, x = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
        self.grid = np.stack([y, x], axis=-1)

    def _bmu(self, x):
        d2 = np.sum((self.W - x)**2, axis=2)
        return np.unravel_index(np.argmin(d2), (self.m, self.n))

    def fit(self, X, epochs=15):
        total = epochs * len(X)
        t = 0
        for epoch in range(epochs):
            self.rng.shuffle(X)
            for x in X:
                lr = self.lr0 * np.exp(-t / total)
                sigma = self.sigma0 * np.exp(-t / total)
                iy, ix = self._bmu(x)
                dist2 = np.sum((self.grid - [iy, ix])**2, axis=-1)
                h = np.exp(-dist2 / (2 * sigma**2))
                self.W += lr * h[..., None] * (x - self.W)
                t += 1

    def transform(self, X):
        coords = np.array([self._bmu(x) for x in X])
        return coords

# Дані для навчання SOM
rng = np.random.default_rng(0)
centers = np.array([[1, 1], [6, 4], [-4, 5]])
X = np.vstack([
    rng.normal(centers[0], 0.7, size=(150, 2)),
    rng.normal(centers[1], 1.0, size=(200, 2)),
    rng.normal(centers[2], 0.8, size=(180, 2)),
])

som = MiniSOM(8, 8, dim=2, lr=0.5, rng=rng)
som.fit(X, epochs=12)
coords = som.transform(X)

heat = np.zeros((som.m, som.n), dtype=int)
for y, x in coords:
    heat[y, x] += 1

print("Топ-5 найактивніших нейронів:")
for i, (y, x) in enumerate(np.dstack(np.unravel_index(np.argsort(heat, axis=None)[::-1], heat.shape))[0][:5]):
    print(f"{i+1}) ({y}, {x}) -> {heat[y, x]} точок")

rng = np.random.default_rng(101)

n_cities = 12
coords = rng.random((n_cities, 2)) * 100

# Відстані між містами
D = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(axis=2)) + 1e-9
eta = 1 / D

# Параметри
alpha = 1
beta = 4
rho = 0.4
Q = 80
n_ants = n_cities
n_iters = 100

tau = np.ones((n_cities, n_cities))
best_len = np.inf
best_tour = None

def tour_length(tour):
    return np.sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

for it in range(n_iters):
    tours, lengths = [], []
    for _ in range(n_ants):
        start = rng.integers(n_cities)
        visited = {start}
        tour = [start]
        while len(visited) < n_cities:
            unvisited = list(set(range(n_cities)) - visited)
            prob = (tau[tour[-1], unvisited]**alpha) * (eta[tour[-1], unvisited]**beta)
            prob /= prob.sum()
            nxt = rng.choice(unvisited, p=prob)
            tour.append(nxt)
            visited.add(nxt)
        L = tour_length(tour)
        tours.append(tour)
        lengths.append(L)
    tau *= (1 - rho)
    best_idx = np.argmin(lengths)
    best_iter_tour, best_iter_len = tours[best_idx], lengths[best_idx]
    delta_tau = Q / best_iter_len
    for i in range(n_cities):
        a, b = best_iter_tour[i], best_iter_tour[(i + 1) % n_cities]
        tau[a, b] += delta_tau
        tau[b, a] += delta_tau
    if best_iter_len < best_len:
        best_len = best_iter_len
        best_tour = best_iter_tour

print("Найкраща довжина:", best_len)
print("Найкращий тур:", best_tour)