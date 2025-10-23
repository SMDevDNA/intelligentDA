import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

def rosenbrock(x):
    """Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    """Gradient of Rosenbrock function."""
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

def conjugate_gradient(method_name, x0, max_iters=2000, tol=1e-6):
    x = x0.copy()
    g = grad_rosenbrock(x)
    d = -g
    values = [rosenbrock(x)]

    for k in range(max_iters):
        alpha = 1e-3
        while rosenbrock(x + alpha * d) > rosenbrock(x) - 1e-4 * alpha * np.dot(g, d):
            alpha *= 0.5
            if alpha < 1e-10:
                break

        x_new = x + alpha * d
        g_new = grad_rosenbrock(x_new)
        values.append(rosenbrock(x_new))

        if norm(g_new) < tol:
            break

        if method_name == "Fletcher–Reeves":
            beta = np.dot(g_new, g_new) / np.dot(g, g)
        elif method_name == "Polak–Ribiere":
            beta = np.dot(g_new, g_new - g) / np.dot(g, g)
        elif method_name == "Hestenes–Stiefel":
            beta = np.dot(g_new, g_new - g) / np.dot(d, g_new - g)
        elif method_name == "Dai–Yuan":
            beta = np.dot(g_new, g_new) / np.dot(d, g_new - g)
        else:
            raise ValueError("Unknown method")

        if beta < 0 or beta > 10:
            beta = 0

        d = -g_new + beta * d
        x, g = x_new, g_new

    return x, rosenbrock(x), norm(g), k + 1, np.array(values)

methods = ["Fletcher–Reeves", "Polak–Ribiere", "Hestenes–Stiefel", "Dai–Yuan"]
results = []
x0 = np.array([-1.2, 1.0])

plt.figure(figsize=(7, 4))

for method in methods:
    x_opt, fx, gnorm, iters, values = conjugate_gradient(method, x0)
    results.append((method, fx, gnorm, iters))
    plt.plot(values, label=method)

plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Conjugate Gradient Methods Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('conjugate_gradient_comparison.png', dpi=200)

print(f"{'Method':<20} {'f(x)':>12} {'||grad||':>12} {'Iters':>8}")
print('-'*54)
for method, fx, gnorm, iters in results:
    print(f"{method:<20} {fx:12.3e} {gnorm:12.3e} {iters:8d}")

print("\nЗбережено: conjugate_gradient_comparison.png")
