"""
Лабораторная работа № 2
Метод Эйлера, расширенный метод Эйлера и метод Рунге–Кутты 4-го порядка

Задача Коши: y' = x + y,  y(0) = 1,  0 <= x <= 1,  h = 0.2
Точное решение: y(x) = 2 * exp(x) - x - 1
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

X_START = 0.0
X_END = 1.0
STEP = 0.2
Y_INIT = 1.0

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def ode_rhs(x: float, y: float) -> float:
    """Правая часть дифференциального уравнения y' = x + y."""
    return x + y


def exact_solution(x: np.ndarray | float) -> np.ndarray | float:
    """Точное аналитическое решение: y(x) = 2*e^x - x - 1."""
    return 2.0 * np.exp(x) - x - 1.0


def solve_euler(rhs, x_grid: np.ndarray, y0: float) -> np.ndarray:
    """Классический явный метод Эйлера 1-го порядка."""
    h = float(x_grid[1] - x_grid[0])
    y = np.empty_like(x_grid, dtype=float)
    y[0] = y0
    for n in range(len(x_grid) - 1):
        y[n + 1] = y[n] + h * rhs(x_grid[n], y[n])
    return y


def solve_extended_euler(rhs, x_grid: np.ndarray, y0: float) -> np.ndarray:
    """Расширенный метод Эйлера (схема предиктор–корректор / метод Хойна, 2-й порядок)."""
    h = float(x_grid[1] - x_grid[0])
    y = np.empty_like(x_grid, dtype=float)
    y[0] = y0
    for n in range(len(x_grid) - 1):
        slope_left = rhs(x_grid[n], y[n])
        y_pred = y[n] + h * slope_left
        slope_right = rhs(x_grid[n + 1], y_pred)
        y[n + 1] = y[n] + 0.5 * h * (slope_left + slope_right)
    return y


def solve_rk4(rhs, x_grid: np.ndarray, y0: float) -> np.ndarray:
    """Классический метод Рунге–Кутты 4-го порядка точности."""
    h = float(x_grid[1] - x_grid[0])
    y = np.empty_like(x_grid, dtype=float)
    y[0] = y0
    for n in range(len(x_grid) - 1):
        xn, yn = x_grid[n], y[n]
        k1 = h * rhs(xn, yn)
        k2 = h * rhs(xn + 0.5 * h, yn + 0.5 * k1)
        k3 = h * rhs(xn + 0.5 * h, yn + 0.5 * k2)
        k4 = h * rhs(xn + h, yn + k3)
        y[n + 1] = yn + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


def print_tables(x_grid: np.ndarray, y_true: np.ndarray,
                 y_euler: np.ndarray, y_ext: np.ndarray, y_rk4: np.ndarray,
                 err_euler: np.ndarray, err_ext: np.ndarray, err_rk4: np.ndarray) -> None:
    """Форматированный вывод таблицы значений и таблицы абсолютных погрешностей."""
    print("=" * 88)
    print("ТАБЛИЦА 1. ЗНАЧЕНИЯ ТОЧНОГО И ЧИСЛЕННЫХ РЕШЕНИЙ В УЗЛАХ СЕТКИ")
    print("=" * 88)
    hdr1 = (f"  {'(x)':>5} | {'Точное значение':>16} | {'Метод Эйлера':>14} | "
            f"{'Расшир. м. Эйлера':>18} | {'Метод Рунге–Кутты':>18}")
    print(hdr1)
    print("  " + "-" * (len(hdr1) - 2))
    for i in range(len(x_grid)):
        print(f"  {x_grid[i]:>5.1f} | {y_true[i]:>16.6f} | {y_euler[i]:>14.6f} | "
              f"{y_ext[i]:>18.6f} | {y_rk4[i]:>18.6f}")

    print("\n" + "=" * 88)
    print("ТАБЛИЦА 2. АБСОЛЮТНЫЕ ПОГРЕШНОСТИ ЧИСЛЕННЫХ МЕТОДОВ Δ = |y_точн - y_прибл|")
    print("=" * 88)
    hdr2 = (f"  {'(x)':>5} | {'Погрешность м. Эйлера':>23} | "
            f"{'Погрешность расшир. Эйлера':>27} | {'Погрешность м. Рунге–Кутты':>26}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for i in range(len(x_grid)):
        print(f"  {x_grid[i]:>5.1f} | {err_euler[i]:>23.6f} | "
              f"{err_ext[i]:>27.6f} | {err_rk4[i]:>26.6f}")


def build_plots(x_grid: np.ndarray, y_true: np.ndarray,
                y_euler: np.ndarray, y_ext: np.ndarray, y_rk4: np.ndarray,
                err_euler: np.ndarray, err_ext: np.ndarray, err_rk4: np.ndarray) -> None:
    """Строит общий график решений (с врезкой масштаба) и логарифмический график ошибок."""
    x_fine = np.linspace(X_START, X_END, 300)
    y_fine = exact_solution(x_fine)

    # 1. Основной график всех четырёх решений на одной координатной плоскости
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=130)

    ax.plot(x_fine, y_fine, color="#0F172A", linewidth=2.4,
            label="1. Точное решение $y = 2e^x - x - 1$")
    ax.plot(x_grid, y_euler, color="#DC2626", linestyle="--", marker="o",
            markersize=6.0, linewidth=1.7, label="2. Метод Эйлера ($p = 1$)")
    ax.plot(x_grid, y_ext, color="#0284C7", linestyle="-.", marker="s",
            markersize=5.8, linewidth=1.7, label="3. Расширенный метод Эйлера ($p = 2$)")
    ax.plot(x_grid, y_rk4, color="#16A34A", linestyle=":", marker="^",
            markersize=6.5, linewidth=1.9, label="4. Метод Рунге–Кутты 4-го порядка ($p = 4$)")

    # Врезка у правой границы x = 1.0 для различимости РК4 и расширенного Эйлера
    ax_ins = ax.inset_axes([0.08, 0.36, 0.35, 0.30])
    mask_fine = x_fine >= 0.94
    ax_ins.plot(x_fine[mask_fine], y_fine[mask_fine], color="#0F172A", linewidth=2.0)
    ax_ins.plot(x_grid[-2:], y_ext[-2:], color="#0284C7", linestyle="-.", marker="s", markersize=5.5)
    ax_ins.plot(x_grid[-2:], y_rk4[-2:], color="#16A34A", linestyle=":", marker="^", markersize=6.2)
    ax_ins.scatter([1.0], [y_true[-1]], color="#0F172A", s=26, zorder=5)
    ax_ins.set_xlim(0.96, 1.008)
    ax_ins.set_ylim(3.395, 3.448)
    ax_ins.set_title("Увеличение вблизи x = 1", fontsize=8.3, fontweight="bold")
    ax_ins.grid(True, linestyle="--", alpha=0.35)
    ax_ins.tick_params(labelsize=7.8)

    ax.set_title("Сравнение численных методов решения задачи Коши y' = x + y, y(0) = 1 (h = 0.2)",
                 fontsize=11.2, fontweight="bold", pad=10)
    ax.set_xlabel("Аргумент x", fontsize=10.5)
    ax.set_ylabel("Значение функции y(x)", fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr2_solutions.png", dpi=220)

    # 2. Дополнительный график погрешностей в логарифмическом масштабе (для x >= 0.2)
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.5), dpi=130)
    xk = x_grid[1:]
    ax2.semilogy(xk, err_euler[1:], color="#DC2626", marker="o", linewidth=1.8,
                 label=f"Метод Эйлера ($\\Delta(1) = {err_euler[-1]:.6f}$)")
    ax2.semilogy(xk, err_ext[1:], color="#0284C7", marker="s", linewidth=1.8,
                 label=f"Расширенный метод Эйлера ($\\Delta(1) = {err_ext[-1]:.6f}$)")
    ax2.semilogy(xk, err_rk4[1:], color="#16A34A", marker="^", linewidth=1.8,
                 label=f"Метод Рунге–Кутты 4-го порядка ($\\Delta(1) = {err_rk4[-1]:.6f}$)")

    ax2.set_title("Абсолютные погрешности методов в узлах сетки (логарифмическая шкала)",
                  fontsize=11.0, fontweight="bold", pad=10)
    ax2.set_xlabel("Узлы сетки x", fontsize=10.2)
    ax2.set_ylabel("Абсолютная погрешность Δ(x) (лог. шкала)", fontsize=10.2)
    ax2.set_xticks(xk)
    ax2.grid(True, which="both", linestyle="--", alpha=0.38)
    ax2.legend(loc="center right", frameon=True, fontsize=9.2)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "lr2_errors_log.png", dpi=220)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_nodes = int(round((X_END - X_START) / STEP)) + 1
    x_grid = np.linspace(X_START, X_END, n_nodes)

    y_true = exact_solution(x_grid)
    y_euler = solve_euler(ode_rhs, x_grid, Y_INIT)
    y_ext = solve_extended_euler(ode_rhs, x_grid, Y_INIT)
    y_rk4 = solve_rk4(ode_rhs, x_grid, Y_INIT)

    err_euler = np.abs(y_true - y_euler)
    err_ext = np.abs(y_true - y_ext)
    err_rk4 = np.abs(y_true - y_rk4)

    print_tables(x_grid, y_true, y_euler, y_ext, y_rk4, err_euler, err_ext, err_rk4)
    build_plots(x_grid, y_true, y_euler, y_ext, y_rk4, err_euler, err_ext, err_rk4)

    print("\nВыводы:")
    print(f"  1. Самый точный метод: Рунге–Кутты 4-го порядка (Δ(1) = {err_rk4[-1]:.6f}).")
    print(f"  2. Наибольшая погрешность: явный метод Эйлера (Δ(1) = {err_euler[-1]:.6f}).")
    print("  3. РК4 учитывает 4 пробных наклона по правилу Симпсона, обеспечивая порядок точности O(h^4).")
    print("  4. При уменьшении h в k раз глобальная ошибка падает как k^1 (Эйлер), k^2 (Хойн) и k^4 (РК4).")
    print(f"\nГрафики сохранены в папку: {PLOTS_DIR}")

    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
