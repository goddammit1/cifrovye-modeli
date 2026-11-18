"""
Лабораторная работа № 3 (Домашнее задание)
Аппроксимация экспериментальных данных различными функциями

Сравнение 6 аппроксимирующих моделей по критериям SSE и RMSE:
  1) Линейная            y = a*x + b
  2) Квадратичная        y = a*x^2 + b*x + c
  3) Логарифмическая     y = a*ln(x + 1) + b
  4) Степенная           y = a*(x + 1)^b
  5) Экспоненциальная    y = a*e^(b*x)
  6) Тригонометрическая  y = a*sin(x) + b*cos(x) + c
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# 1. Исходные экспериментальные данные
X_DATA = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
Y_DATA = np.array([1.10, 1.48, 2.20, 3.05, 4.10, 5.75, 7.90, 10.60], dtype=float)


@dataclass
class FitResult:
    name: str
    short_name: str
    formula: str
    params: np.ndarray
    param_labels: tuple[str, ...]
    eval_fn: Callable[[np.ndarray], np.ndarray]
    y_hat: np.ndarray
    sse: float
    rmse: float

    def format_params(self) -> str:
        return ", ".join(f"{lbl} = {val:.4f}" for lbl, val in zip(self.param_labels, self.params))


def fit_linear_ols(design_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Находит вектор параметров линейной по коэффициентам модели методом МНК."""
    theta, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
    return theta


def fit_gauss_newton(model_fn: Callable, jacobian_fn: Callable,
                     x: np.ndarray, y: np.ndarray, theta_init: list[float],
                     tol: float = 1e-11, max_iter: int = 100) -> np.ndarray:
    """
    Нелинейный метод наименьших квадратов (алгоритм Гаусса–Ньютона)
    для прямой минимизации суммы квадратов ошибок SSE = sum((y_i - y_hat_i)^2).
    """
    theta = np.array(theta_init, dtype=float)
    for _ in range(max_iter):
        residual = y - model_fn(x, *theta)
        jac = jacobian_fn(x, *theta)
        delta, *_ = np.linalg.lstsq(jac, residual, rcond=None)
        theta = theta + delta
        if np.linalg.norm(delta) < tol:
            break
    return theta


def compute_errors(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    """Вычисляет сумму квадратов ошибок SSE и среднеквадратическую ошибку RMSE."""
    sse = float(np.sum((y_true - y_hat) ** 2))
    rmse = float(np.sqrt(sse / len(y_true)))
    return sse, rmse


def build_all_models(x: np.ndarray, y: np.ndarray) -> list[FitResult]:
    """Оценивает коэффициенты и метрики точности всех шести моделей."""
    ones = np.ones_like(x)
    models: list[FitResult] = []

    # 1. Линейная модель: y = a*x + b
    p_lin = fit_linear_ols(np.column_stack([x, ones]), y)
    fn_lin = lambda t, p=p_lin: p[0] * t + p[1]
    yh_lin = fn_lin(x)
    models.append(FitResult(
        "Линейная функция", "Линейная", "y = ax + b",
        p_lin, ("a", "b"), fn_lin, yh_lin, *compute_errors(y, yh_lin)
    ))

    # 2. Квадратичная модель: y = a*x^2 + b*x + c
    p_quad = fit_linear_ols(np.column_stack([x ** 2, x, ones]), y)
    fn_quad = lambda t, p=p_quad: p[0] * t ** 2 + p[1] * t + p[2]
    yh_quad = fn_quad(x)
    models.append(FitResult(
        "Квадратичная функция", "Квадратичная", "y = ax² + bx + c",
        p_quad, ("a", "b", "c"), fn_quad, yh_quad, *compute_errors(y, yh_quad)
    ))

    # 3. Логарифмическая модель: y = a*ln(x + 1) + b
    p_log = fit_linear_ols(np.column_stack([np.log(x + 1.0), ones]), y)
    fn_log = lambda t, p=p_log: p[0] * np.log(t + 1.0) + p[1]
    yh_log = fn_log(x)
    models.append(FitResult(
        "Логарифмическая функция", "Логарифм.", "y = a ln(x + 1) + b",
        p_log, ("a", "b"), fn_log, yh_log, *compute_errors(y, yh_log)
    ))

    # 4. Степенная модель: y = a*(x + 1)^b (старт из логарифмической линеаризации)
    u_pow = np.log(x + 1.0)
    b0_pow, lna0_pow = fit_linear_ols(np.column_stack([u_pow, ones]), np.log(y))
    p_pow = fit_gauss_newton(
        model_fn=lambda t, a, b: a * (t + 1.0) ** b,
        jacobian_fn=lambda t, a, b: np.column_stack([
            (t + 1.0) ** b,
            a * ((t + 1.0) ** b) * np.log(t + 1.0),
        ]),
        x=x, y=y, theta_init=[float(np.exp(lna0_pow)), float(b0_pow)]
    )
    fn_pow = lambda t, p=p_pow: p[0] * (t + 1.0) ** p[1]
    yh_pow = fn_pow(x)
    models.append(FitResult(
        "Степенная функция", "Степенная", "y = a(x + 1)^b",
        p_pow, ("a", "b"), fn_pow, yh_pow, *compute_errors(y, yh_pow)
    ))

    # 5. Экспоненциальная модель: y = a*e^(b*x) (старт из логарифмической линеаризации)
    b0_exp, lna0_exp = fit_linear_ols(np.column_stack([x, ones]), np.log(y))
    p_exp = fit_gauss_newton(
        model_fn=lambda t, a, b: a * np.exp(b * t),
        jacobian_fn=lambda t, a, b: np.column_stack([
            np.exp(b * t),
            a * t * np.exp(b * t),
        ]),
        x=x, y=y, theta_init=[float(np.exp(lna0_exp)), float(b0_exp)]
    )
    fn_exp = lambda t, p=p_exp: p[0] * np.exp(p[1] * t)
    yh_exp = fn_exp(x)
    models.append(FitResult(
        "Экспоненциальная функция", "Экспоненц.", "y = ae^(bx)",
        p_exp, ("a", "b"), fn_exp, yh_exp, *compute_errors(y, yh_exp)
    ))

    # 6. Тригонометрическая модель: y = a*sin(x) + b*cos(x) + c
    p_trig = fit_linear_ols(np.column_stack([np.sin(x), np.cos(x), ones]), y)
    fn_trig = lambda t, p=p_trig: p[0] * np.sin(t) + p[1] * np.cos(t) + p[2]
    yh_trig = fn_trig(x)
    models.append(FitResult(
        "Тригонометрическая функция", "Тригоном.", "y = a sin(x) + b cos(x) + c",
        p_trig, ("a", "b", "c"), fn_trig, yh_trig, *compute_errors(y, yh_trig)
    ))

    return models


def print_results(x: np.ndarray, y: np.ndarray, models: list[FitResult]) -> None:
    """Выводит таблицу приближённых значений y_hat и итоговую таблицу коэффициентов и ошибок."""
    print("=" * 96)
    print("ТАБЛИЦА 1. ПРИБЛИЖЁННЫЕ ЗНАЧЕНИЯ y_hat В ИСХОДНЫХ ТОЧКАХ ДЛЯ ВСЕХ МОДЕЛЕЙ")
    print("=" * 96)
    cols = " | ".join(f"{m.short_name:>11}" for m in models)
    hdr1 = f"  {'x':>4} | {'y эксп.':>8} | {cols}"
    print(hdr1)
    print("  " + "-" * (len(hdr1) - 2))
    for i in range(len(x)):
        vals = " | ".join(f"{m.y_hat[i]:>11.4f}" for m in models)
        print(f"  {x[i]:>4.1f} | {y[i]:>8.2f} | {vals}")

    print("\n" + "=" * 106)
    print("ТАБЛИЦА 2. ИТОГОВАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ И ОШИБОК АППРОКСИМАЦИИ")
    print("=" * 106)
    hdr2 = f"  {'Модель':<27} | {'Вид функции':<27} | {'Коэффициенты':<32} | {'SSE':>8} | {'RMSE':>7}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for m in models:
        print(f"  {m.name:<27} | {m.formula:<27} | {m.format_params():<32} | "
              f"{m.sse:>8.4f} | {m.rmse:>7.4f}")

    best = min(models, key=lambda item: item.sse)
    ranked = sorted(models, key=lambda item: item.sse)
    print("\nРейтинг моделей по возрастанию ошибки SSE:")
    for rank, m in enumerate(ranked, start=1):
        print(f"  {rank}. {m.name:<27} — SSE = {m.sse:.4f}, RMSE = {m.rmse:.4f}")
    print(f"\nВывод: наилучшее качество аппроксимации даёт {best.name.lower()} "
          f"({best.formula}, {best.format_params()}) с минимальными ошибками "
          f"SSE = {best.sse:.4f} и RMSE = {best.rmse:.4f}.")


def plot_models(x: np.ndarray, y: np.ndarray, models: list[FitResult]) -> None:
    """Строит общий график всех моделей и диаграмму сравнения ошибок SSE / RMSE."""
    x_dense = np.linspace(float(x.min()), float(x.max()), 350)
    styles = [
        ("#64748B", "--", 1.7),   # Линейная
        ("#0284C7", "-.", 2.0),   # Квадратичная
        ("#D97706", ":",  1.9),   # Логарифмическая
        ("#7C3AED", "--", 1.8),   # Степенная
        ("#DC2626", "-",  2.5),   # Экспоненциальная (лучшая — сплошная яркая линия)
        ("#0D9488", "-.", 1.7),   # Тригонометрическая
    ]

    # 1. Общий график исходных точек и шести аппроксимирующих кривых
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=130)
    for m, (color, ls, lw) in zip(models, styles):
        ax.plot(x_dense, m.eval_fn(x_dense), color=color, linestyle=ls, linewidth=lw,
                label=f"{m.name} ($SSE = {m.sse:.4f}$)")

    ax.scatter(x, y, color="#0F172A", s=62, zorder=6, edgecolors="white", linewidth=0.9,
               label="Исходные экспериментальные точки")

    ax.set_title("Аппроксимация экспериментальных данных различными видами функций",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Значения аргумента x", fontsize=10.5)
    ax.set_ylabel("Значения функции y", fontsize=10.5)
    ax.set_xlim(-0.12, 3.62)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9.1)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr3_all_models.png", dpi=220)

    # 2. Сравнительная диаграмма ошибок RMSE и SSE (по возрастанию ошибки)
    ranked = sorted(models, key=lambda item: item.sse)
    names = [m.short_name for m in ranked]
    sse_vals = [m.sse for m in ranked]
    rmse_vals = [m.rmse for m in ranked]

    fig2, (ax_sse, ax_rmse) = plt.subplots(1, 2, figsize=(9.6, 4.1), dpi=130)
    bar_colors = ["#16A34A", "#0284C7", "#7C3AED", "#0D9488", "#64748B", "#D97706"]

    bars1 = ax_sse.barh(names, sse_vals, color=bar_colors, height=0.58)
    ax_sse.invert_yaxis()
    ax_sse.set_title("Сумма квадратов ошибок (SSE)", fontsize=10.5, fontweight="bold")
    ax_sse.set_xlabel("SSE (меньше — лучше)", fontsize=9.5)
    ax_sse.set_xlim(0, max(sse_vals) * 1.22)
    ax_sse.grid(True, axis="x", linestyle="--", alpha=0.38)
    for b, v in zip(bars1, sse_vals):
        ax_sse.text(v + 0.25, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                    va="center", fontsize=8.6)

    bars2 = ax_rmse.barh(names, rmse_vals, color=bar_colors, height=0.58)
    ax_rmse.invert_yaxis()
    ax_rmse.set_title("Среднеквадратическая ошибка (RMSE)", fontsize=10.5, fontweight="bold")
    ax_rmse.set_xlabel("RMSE (меньше — лучше)", fontsize=9.5)
    ax_rmse.set_xlim(0, max(rmse_vals) * 1.22)
    ax_rmse.grid(True, axis="x", linestyle="--", alpha=0.38)
    for b, v in zip(bars2, rmse_vals):
        ax_rmse.text(v + 0.025, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                    va="center", fontsize=8.6)

    fig2.suptitle("Ранжирование аппроксимирующих моделей по критериям точности",
                  fontsize=11.3, fontweight="bold", y=1.01)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "lr3_errors_bar.png", dpi=220, bbox_inches="tight")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    models = build_all_models(X_DATA, Y_DATA)
    print_results(X_DATA, Y_DATA, models)
    plot_models(X_DATA, Y_DATA, models)
    print(f"\nГрафики сохранены в папку: {PLOTS_DIR}")

    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
