import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1) Дані: використаймо вбудований набір Iris
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names

    # 2) Train/test спліт (стратифікований)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3) Базова модель дерева рішень
    base_clf = DecisionTreeClassifier(random_state=42, criterion="gini")
    base_clf.fit(X_train, y_train)

    # 4) Оцінка базової моделі
    y_pred_base = base_clf.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)

    print("=== Базова модель (без тюнінгу) ===")
    print(f"Точність (accuracy): {acc_base:.3f}")
    print("Матриця неточностей (confusion matrix):")
    print(confusion_matrix(y_test, y_pred_base))
    print("\nЗвіт класифікації:")
    print(classification_report(y_test, y_pred_base, target_names=target_names))

    # 5) Підбір гіперпараметрів (невелика сітка)
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 2, 3, 4, 5],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== Найкраща модель після GridSearchCV ===")
    print("Найкращі параметри:", grid.best_params_)
    print(f"Точність (test accuracy): {acc:.3f}")
    print("Матриця неточностей (confusion matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nЗвіт класифікації:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 6) Дерево у вигляді правил (текст)
    rules = export_text(
        best_clf,
        feature_names=list(X.columns),
        show_weights=True
    )
    print("=== Дерево рішень (правила) ===")
    print(rules)

    # 7) Важливості ознак
    importances = pd.Series(best_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("=== Важливості ознак ===")
    print(importances)

    # 8) Візуалізація дерева та збереження в PNG
    plt.figure(figsize=(14, 10))
    plot_tree(
        best_clf,
        feature_names=X.columns,
        class_names=list(target_names),
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree (Iris)")
    plt.tight_layout()
    plt.savefig("decision_tree_iris.png", dpi=150)
    print("\nВізуалізацію дерева збережено у файлі: decision_tree_iris.png")

    # Додаткова візуалізація важливостей ознак (барчарт)
    plt.figure(figsize=(6,4))
    importances.plot(kind="bar")
    plt.ylabel("Importance")
    plt.title("Feature Importances (Decision Tree)")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    print("Діаграму важливостей ознак збережено у файлі: feature_importances.png")

if __name__ == "__main__":
    main()