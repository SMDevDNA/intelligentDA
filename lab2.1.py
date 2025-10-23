import sys
import importlib
import numpy as np

REQS = ["numpy", "pandas", "matplotlib", "scikit-learn", "statsmodels", "scipy"]


def verify_deps():
    """Перевірка наявності потрібних бібліотек у середовищі."""
    missing = []
    for pkg in REQS:
        modname = "sklearn" if pkg == "scikit-learn" else pkg
        try:
            importlib.import_module(modname)
        except Exception:
            missing.append(pkg)
    if missing:
        print("[ENV] Не знайдені пакети:", ", ".join(missing))
        print("Встановіть командою:\n  pip install " + " ".join(missing))
    else:
        print("[ENV] Усе, що потрібно, вже встановлено.")


# -----------------------------
# Q1. Ознаки часових рядів + стаціонарність
# -----------------------------
def ts_stationarity_demo(show_plot: bool = False):
    """Генерація тренд+сезонність+шум і ADF-тест."""
    try:
        from statsmodels.tsa.stattools import adfuller
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[Q1] Потрібні statsmodels та matplotlib:", e)
        return

    rng = np.random.default_rng(0)
    n = 300
    trend = np.linspace(0.0, 5.0, n)
    season = 2.0 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 1.0, n)
    series = trend + season + noise

    adf_stat, pval, *_ = adfuller(series)
    print(f"[Q1] ADF статистика = {adf_stat:.3f}, p-значення = {pval:.4f} (p<0.05 ⇒ стаціонарність)")

    if show_plot:
        plt.figure()
        plt.plot(series)
        plt.title("Ряд із трендом та сезонністю")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Q2. Коваріаційна матриця
# -----------------------------
def covariance_example():
    """Обчислення коваріаційної матриці та перевірка симетричності."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 4.0, 5.0],
            [4.0, 3.0, 2.0],
        ]
    )
    S = np.cov(X, rowvar=False, bias=False)
    print("[Q2] Коваріаційна матриця S:\n", np.round(S, 3))
    print("[Q2] Симетричність:", bool(np.allclose(S, S.T)))


# -----------------------------
# Q3. Лінійний дискримінантний аналіз (LDA)
# -----------------------------
def lda_basic_demo():
    """Базовий LDA на синтетичних даних із трьома класами."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        print("[Q3] Потрібен scikit-learn:", e)
        return

    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    model = LinearDiscriminantAnalysis()
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    print("[Q3] Підсумки класифікації (LDA):\n", classification_report(yte, yp))
    print("[Q3] Матриця невідповідностей:\n", confusion_matrix(yte, yp))


# -----------------------------
# Q4. Порівняння LDA vs QDA
# -----------------------------
def lda_qda_contrast(show_plot: bool = False):
    """Порівняння точності LDA і QDA на даних із різними дисперсіями класів."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[Q4] Потрібен scikit-learn (та matplotlib для графіків):", e)
        return

    X, y = make_blobs(
        n_samples=500,
        centers=[[0, 0], [3, 1]],
        cluster_std=[1.0, 2.0],
        random_state=7,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, stratify=y, random_state=7)

    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis().fit(Xtr, ytr)

    lda_acc = accuracy_score(yte, lda.predict(Xte))
    qda_acc = accuracy_score(yte, qda.predict(Xte))
    print(f"[Q4] LDA точність = {lda_acc:.3f}")
    print(f"[Q4] QDA точність = {qda_acc:.3f}")

    if show_plot:
        import numpy as _np  # локальний псевдонім, щоб не плутати
        xx, yy = _np.meshgrid(
            _np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            _np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        grid = _np.c_[xx.ravel(), yy.ravel()]
        Z_lda = lda.predict(grid).reshape(xx.shape)
        Z_qda = qda.predict(grid).reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z_lda, alpha=0.25)
        for cls in _np.unique(y):
            plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"class {cls}")
        plt.title("LDA: області рішень")
        plt.legend()

        plt.figure()
        plt.contourf(xx, yy, Z_qda, alpha=0.25)
        for cls in _np.unique(y):
            plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"class {cls}")
        plt.title("QDA: області рішень")
        plt.legend()

        plt.show()


# -----------------------------
# Q5. Регуляризація та CV для LDA
# -----------------------------
def lda_cv_with_shrinkage():
    """LDA зі зменшенням розмірності коваріації (shrinkage) та крос-валідацією."""
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.datasets import make_classification
    except Exception as e:
        print("[Q5] Потрібен scikit-learn:", e)
        return

    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        class_sep=1.2,
        random_state=0,
    )
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(f"[Q5] Середня точність CV = {np.mean(scores):.3f}, σ = {np.std(scores):.3f}")


# -----------------------------
# Q6. Аналоги можливостей Statistics Toolbox у Python
# -----------------------------
def stats_toolbox_like_suite():
    """Підбір розподілу, OLS, ANOVA, KMeans, PCA, ARIMA — короткі приклади."""
    try:
        import pandas as pd
        from scipy import stats
        import statsmodels.formula.api as smf
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.datasets import make_blobs
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print("[Q6] Потрібні scipy, statsmodels, scikit-learn, pandas:", e)
        return

    rng = np.random.default_rng(123)

    # 6.1 Описова статистика
    x = rng.normal(loc=5, scale=2, size=200)
    print("[Q6] Описова статистика: mean =", round(np.mean(x), 3), "; std =", round(np.std(x, ddof=1), 3))

    # 6.2 Підбір нормального розподілу і KS-тест
    mu, sigma = stats.norm.fit(x)
    ks_stat, ks_p = stats.kstest(x, "norm", args=(mu, sigma))
    print(f"[Q6] Norm fit: μ={mu:.3f}, σ={sigma:.3f}; p(KS)={ks_p:.3f}")

    # 6.3 Лінійна регресія (OLS)
    df = pd.DataFrame(
        {
            "y": x + rng.normal(0, 1.0, size=len(x)),
            "x1": x,
            "x2": rng.uniform(size=len(x)),
        }
    )
    ols = smf.ols("y ~ x1 + x2", data=df).fit()
    print("[Q6] OLS: R^2 =", round(ols.rsquared, 3))

    # 6.4 Однофакторна ANOVA
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0.5, 1, 30)
    g3 = rng.normal(1.0, 1, 30)
    f_stat, p_val = stats.f_oneway(g1, g2, g3)
    print(f"[Q6] ANOVA: F={f_stat:.3f}, p={p_val:.4f}")

    # 6.5 Кластеризація (k-means)
    Xc, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xc)
    print("[Q6] KMeans: inertia =", round(km.inertia_, 2))

    # 6.6 Зниження розмірності (PCA)
    Xp = rng.normal(size=(200, 5))
    pca = PCA(n_components=2).fit(Xp)
    print("[Q6] PCA: частки дисперсії =", np.round(pca.explained_variance_ratio_, 3))

    # 6.7 Часові ряди (ARIMA)
    ts = np.cumsum(rng.normal(size=300))  # інтегрований шум
    arima = ARIMA(ts, order=(1, 1, 1)).fit()
    print("[Q6] ARIMA(1,1,1): AIC =", round(arima.aic, 2))


# -----------------------------
# Q7. Аналоги gscatter та classify у Python
# -----------------------------
def grouped_scatter_and_lda(show_plot: bool = False):
    """Групований scatter + класифікація точок через LDA."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[Q7] Потрібні scikit-learn та matplotlib:", e)
        return

    rng = np.random.default_rng(1)
    X0 = rng.normal(loc=[0, 0], scale=1.0, size=(60, 2))
    X1 = rng.normal(loc=[3, 2], scale=1.2, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 60 + [1] * 60)

    if show_plot:
        for cls in np.unique(y):
            pts = X[y == cls]
            plt.scatter(pts[:, 0], pts[:, 1], label=f"class {cls}")
        plt.legend()
        plt.title("Групована діаграма (аналог gscatter)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()
        plt.show()

    clf = LinearDiscriminantAnalysis().fit(X, y)
    preds = clf.predict([[1, 1], [4, 3]])
    print("[Q7] Прогнози LDA для точок [1,1] та [4,3]:", preds)


# -----------------------------
# Узагальнений запуск
# -----------------------------
def run_everything():
    print("=== Перевірка залежностей ===")
    verify_deps()
    print("\n=== Q1 ===")
    ts_stationarity_demo(show_plot=False)
    print("\n=== Q2 ===")
    covariance_example()
    print("\n=== Q3 ===")
    lda_basic_demo()
    print("\n=== Q4 ===")
    lda_qda_contrast(show_plot=False)
    print("\n=== Q5 ===")
    lda_cv_with_shrinkage()
    print("\n=== Q6 ===")
    stats_toolbox_like_suite()
    print("\n=== Q7 ===")
    grouped_scatter_and_lda(show_plot=False)


# Відповідність старих імен функцій для сумісності з CLI-параметрами
_ALIAS_MAP = {
    "q1_timeseries_demo": ts_stationarity_demo,
    "q2_covariance_demo": covariance_example,
    "q3_lda_demo": lda_basic_demo,
    "q4_compare_lda_qda": lda_qda_contrast,
    "q5_cv_shrinkage_lda": lda_cv_with_shrinkage,
    "q6_stats_toolbox_equivalents": stats_toolbox_like_suite,
    "q7_gscatter_and_classify_analogs": grouped_scatter_and_lda,
    "run_all": run_everything,
}


if __name__ == "__main__":
    # Якщо аргументів немає — проганяємо всі розділи.
    if len(sys.argv) == 1:
        run_everything()
    else:
        # Можна викликати нові імена або старі (через _ALIAS_MAP).
        name = sys.argv[1]
        fn = globals().get(name) or _ALIAS_MAP.get(name)
        if callable(fn):
            fn()
        else:
            print(f"[CLI] Невідома команда '{name}'. Доступні ключі:")
            print("  ", ", ".join(sorted(list(_ALIAS_MAP.keys()) + [k for k, v in globals().items() if callable(v) and k in {  # виділяємо лише наші
                'ts_stationarity_demo', 'covariance_example', 'lda_basic_demo', 'lda_qda_contrast',
                'lda_cv_with_shrinkage', 'stats_toolbox_like_suite', 'grouped_scatter_and_lda', 'run_everything'
            }])))