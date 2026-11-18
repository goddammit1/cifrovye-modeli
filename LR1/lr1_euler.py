"""
Лабораторная работа № 1
Экспоненциальный рост: реализация метода Эйлера и анализ точности

Модель: y' = 0.25 * y,  y(0) = 80,  0 <= t <= 8
Точное решение: y(t) = 80 * exp(0.25 * t)
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

# Параметры задачи Коши
RATE = 0.25
T_START = 0.0
T_END = 8.0
Y_INIT = 80.0

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def growth_rhs(t: float, y: float) -> float:
    """Правая часть ОДУ y' = f(t, y) = 0.25 * y."""
    return RATE * y


def exact_solution(t: np.ndarray | float) -> np.ndarray | float:
    """Аналитическое решение задачи Коши: y(t) = 80 * exp(0.25 * t)."""
    return Y_INIT * np.exp(RATE * t)


def solve_euler(rhs, t_span: tuple[float, float], y0: float, step: float):
    """
    Явный метод Эйлера для задачи Коши y' = rhs(t, y), y(t0) = y0.
    Возвращает кортеж (t_grid, y_num, n_steps).
    """
    t0, t_final = t_span
    n_steps = int(round((t_final - t0) / step))
    t_grid = np.linspace(t0, t_final, n_steps + 1)
    y_num = np.empty(n_steps + 1, dtype=float)
    y_num[0] = y0

    for n in range(n_steps):
        y_num[n + 1] = y_num[n] + step * rhs(t_grid[n], y_num[n])

    return t_grid, y_num, n_steps


def print_slice_table(title: str, indices: range, t_grid: np.ndarray,
                      y_num: np.ndarray, y_true: np.ndarray) -> None:
    """Печатает фрагмент расчётной таблицы в консоль."""
    print(f"\n  {title}:")
    header = f"  {'n':>4} | {'t_n':>6} | {'y_n (Эйлер)':>13} | {'y(t_n) точн.':>13} | {'Δ_n':>11}"
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)
    for idx in indices:
        err = abs(y_num[idx] - y_true[idx])
        print(f"  {idx:>4d} | {t_grid[idx]:>6.2f} | {y_num[idx]:>13.4f} | "
              f"{y_true[idx]:>13.4f} | {err:>11.4f}")


def run_task_1(steps: list[float]) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Задача 1: расчёт методом Эйлера, вывод таблиц и отдельные графики."""
    print("=" * 74)
    print("ЗАДАЧА 1. МЕТОД ЭЙЛЕРА И СРАВНЕНИЕ С ТОЧНЫМ РЕШЕНИЕМ")
    print("=" * 74)

    t_smooth = np.linspace(T_START, T_END, 400)
    y_smooth = exact_solution(t_smooth)
    y_exact_end = float(exact_solution(T_END))

    palette = {1.0: "#D9480F", 0.5: "#0B7285", 0.1: "#5F3DC4"}
    computed = {}

    for h in steps:
        t_grid, y_num, n_steps = solve_euler(growth_rhs, (T_START, T_END), Y_INIT, h)
        y_true = exact_solution(t_grid)
        err_arr = np.abs(y_num - y_true)
        computed[h] = (t_grid, y_num, err_arr)

        end_err = err_arr[-1]
        rel_err = end_err / y_exact_end * 100.0

        print(f"\n>>> Расчёт для шага h = {h}")
        print(f"  Количество шагов N               : {n_steps}")
        print(f"  Численное значение y(T)          : {y_num[-1]:.4f}")
        print(f"  Точное значение y(8)             : {y_exact_end:.4f}")
        print(f"  Абсолютная ошибка в конечной т.  : {end_err:.4f} ({rel_err:.2f}%)")

        first_idx = range(0, min(5, n_steps + 1))
        last_idx = range(max(0, n_steps - 4), n_steps + 1)
        print_slice_table("Первые 5 значений таблицы", first_idx, t_grid, y_num, y_true)
        print_slice_table("Последние 5 значений таблицы", last_idx, t_grid, y_num, y_true)

        # Отдельный график для текущего шага h
        fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=130)
        color = palette.get(h, "#D9480F")
        ms = 6.0 if n_steps <= 20 else 3.6

        ax.plot(t_smooth, y_smooth, color="#1E293B", linewidth=2.2,
                label="Точное решение $y(t) = 80e^{0.25t}$")
        ax.plot(t_grid, y_num, color=color, linestyle="-", marker="o",
                markersize=ms, linewidth=1.6,
                label=f"Метод Эйлера ($h = {h}$, $N = {n_steps}$)")
        ax.fill_between(t_grid, y_num, y_true, color=color, alpha=0.10,
                        label=f"Накопленная ошибка ($\\Delta(8) = {end_err:.2f}$)")

        ax.set_title(f"Задача 1. Точное и численное решение при шаге h = {h}",
                     fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Время t", fontsize=10.5)
        ax.set_ylabel("Значение y(t)", fontsize=10.5)
        ax.set_xlim(T_START - 0.15, T_END + 0.15)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)
        fig.tight_layout()

        tag = str(h).replace(".", "_")
        fig.savefig(PLOTS_DIR / f"lr1_task1_h_{tag}.png", dpi=220)

    return computed


def run_task_2(computed: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Задача 2: исследование абсолютной погрешности во времени."""
    print("\n" + "=" * 74)
    print("ЗАДАЧА 2. ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ВО ВРЕМЕНИ")
    print("=" * 74)

    palette = {1.0: "#D9480F", 0.5: "#0B7285", 0.1: "#5F3DC4"}
    steps = list(computed.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=130)
    flat_axes = axes.ravel()

    for idx, h in enumerate(steps):
        t_grid, _, err_arr = computed[h]
        ax = flat_axes[idx]
        color = palette[h]
        ms = 5.5 if len(t_grid) <= 25 else 3.2

        print(f"  Шаг h = {h:<4}: max Δ_n = {err_arr.max():8.4f} (в точке t = {t_grid[err_arr.argmax()]:.1f})")

        ax.plot(t_grid, err_arr, color=color, marker="o", markersize=ms, linewidth=1.7)
        ax.fill_between(t_grid, 0.0, err_arr, color=color, alpha=0.10)
        ax.set_title(f"Погрешность во времени при h = {h}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Время $t_n$", fontsize=9.8)
        ax.set_ylabel("Абсолютная ошибка $\\delta_n$", fontsize=9.8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Четвёртая панель — сводное сравнение скоростей роста ошибки
    ax_cmp = flat_axes[3]
    for h in steps:
        t_grid, _, err_arr = computed[h]
        ms = 5.0 if len(t_grid) <= 25 else 2.8
        ax_cmp.plot(t_grid, err_arr, color=palette[h], marker="o", markersize=ms,
                    linewidth=1.7, label=f"$h = {h}$ ($\\Delta(8) = {err_arr[-1]:.2f}$)")
    ax_cmp.set_title("Сравнение роста ошибки для трёх шагов", fontsize=10.5, fontweight="bold")
    ax_cmp.set_xlabel("Время $t_n$", fontsize=9.8)
    ax_cmp.set_ylabel("Абсолютная ошибка $\\delta_n$", fontsize=9.8)
    ax_cmp.grid(True, linestyle="--", alpha=0.4)
    ax_cmp.legend(loc="upper left", fontsize=8.8, frameon=True)

    fig.suptitle("Задача 2. Динамика абсолютной погрешности Δ_n = |y_n − y(t_n)|",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr1_task2_errors.png", dpi=220)

    print("\n  Выводы по Задаче 2:")
    print("  1. Ошибка монотонно и ускоренно (экспоненциально) растёт со временем на [0; 8].")
    print("  2. Быстрее всего ошибка возрастает при самом крупном шаге h = 1 (до 114.2873).")
    print("  3. На конечном отрезке [0; 8] метод численно устойчив (нет осцилляций, при h -> 0 сходится).")


def run_task_3() -> None:
    """Задача 3: зависимость ошибки в конечной точке T от шага интегрирования h."""
    print("\n" + "=" * 74)
    print("ЗАДАЧА 3. ВЛИЯНИЕ ВЕЛИЧИНЫ ШАГА НА ТОЧНОСТЬ И ЧИСЛО ИТЕРАЦИЙ")
    print("=" * 74)

    steps_task3 = [1.0, 0.5, 0.25, 0.1, 0.05]
    y_exact_end = float(exact_solution(T_END))

    n_list = []
    err_list = []

    print(f"  {'Шаг h':>7} | {'Число шагов N':>14} | {'y_N(8)':>11} | "
          f"{'Ошибка Δ(8)':>12} | {'Отн. ошибка':>11}")
    print("  " + "-" * 67)

    for h in steps_task3:
        _, y_num, n_steps = solve_euler(growth_rhs, (T_START, T_END), Y_INIT, h)
        err_end = abs(y_num[-1] - y_exact_end)
        rel_end = err_end / y_exact_end * 100.0
        n_list.append(n_steps)
        err_list.append(err_end)
        print(f"  {h:>7.2f} | {n_steps:>14d} | {y_num[-1]:>11.4f} | "
              f"{err_end:>12.4f} | {rel_end:>10.2f}%")

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=130)

    # Асимптотическая прямая главного члена ошибки (0.5 * lambda^2 * T * y(T) * h)
    h_line = np.linspace(0.0, 1.05, 200)
    lead_slope = 0.5 * (RATE ** 2) * T_END * y_exact_end  # ~147.78 * h
    ax.plot(h_line, lead_slope * h_line, linestyle=":", color="#64748B", linewidth=1.4,
            label="Линейная асимптотика $O(h)$ при $h \\to 0$")

    ax.plot(steps_task3, err_list, color="#0F766E", marker="D", markersize=6.5,
            linewidth=2.0, label="Фактическая ошибка $\\Delta(T)$ метода Эйлера")

    offsets = {
        1.0: (-18, -27),
        0.5: (8, -16),
        0.25: (8, -16),
        0.1: (-30, 12),
        0.05: (8, -14),
    }
    for h_val, err_val in zip(steps_task3, err_list):
        ax.annotate(f"h={h_val}\n({err_val:.1f})",
                    xy=(h_val, err_val),
                    xytext=offsets[h_val],
                    textcoords="offset points",
                    fontsize=8.3, color="#0F172A")

    ax.set_title("Задача 3. Зависимость ошибки в конечной точке T = 8 от шага h",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Шаг интегрирования h", fontsize=10.5)
    ax.set_ylabel("Абсолютная ошибка в точке T = 8", fontsize=10.5)
    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(0.0, max(err_list) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lr1_task3_step_error.png", dpi=220)

    print("\n  Анализ по Задаче 3:")
    print("  1. При уменьшении шага h в k раз ошибка убывает примерно в k раз (первый порядок O(h)).")
    print("  2. Число шагов N = 8 / h растёт обратно пропорционально шагу (с 8 до 160 — в 20 раз).")
    print("  3. Разумный компромисс: h = 0.1 (N = 80, отн. ошибка 2.43%) или h = 0.05 (N = 160, 1.23%).")
    print("  4. Слишком малый шаг резко растит время счёта и приводит к накоплению машинных ошибок округления.")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    computed = run_task_1([1.0, 0.5, 0.1])
    run_task_2(computed)
    run_task_3()
    print(f"\nГрафики сохранены в папку: {PLOTS_DIR}")
    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
