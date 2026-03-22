#!/usr/bin/env python3
"""
PharmaKinetics: главный скрипт запуска всех экспериментов.

Использование:
    python run_all.py          # Запустить всё
    python run_all.py --quick  # Пропустить 3D и объяснимость (быстрее)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step(description, module_name):
    print(f"\n{'*'*70}")
    print(f"* ШАГ: {description}")
    print(f"{'*'*70}\n")
    start = time.time()

    import importlib
    mod = importlib.import_module(module_name)
    mod.main()

    elapsed = time.time() - start
    print(f"\n  [{description}] завершён за {elapsed:.1f}с")
    return elapsed


def main():
    quick = "--quick" in sys.argv

    total_start = time.time()
    timings = {}

    # Шаг 0: загрузка датасетов
    print("\n" + "=" * 70)
    print("ШАГ 0: Загрузка датасетов")
    print("=" * 70)
    from src.data.loader import load_dataset, load_all_datasets
    if quick:
        df = load_dataset("tox21")
        print(f"Датасет Tox21: {len(df)} молекул")
    else:
        datasets = load_all_datasets()
        for name, df in datasets.items():
            print(f"  {name}: {len(df)} молекул")

    # Шаг 1: E1 — бейзлайн vs графовая модель
    t = run_step("E1: Бейзлайн vs Графовая модель",
                 "src.experiments.run_baseline_vs_graph")
    timings["E1"] = t

    if not quick:
        # Шаг 2: E2 — граф vs 3D
        t = run_step("E2: Граф vs Pseudo-3D",
                     "src.experiments.run_graph_vs_3d")
        timings["E2"] = t

        # Шаг 3: объяснимость
        t = run_step("Объяснимость (AOPC + IoU)",
                     "src.experiments.run_explainability")
        timings["XAI"] = t
    else:
        print("\n[--quick] Пропущены: E2 (3D) и эксперименты по объяснимости")

    # Шаг 4: итоговое сравнение
    t = run_step("Итоги экспериментов",
                 "src.experiments.compare_results")
    timings["Итоги"] = t

    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ за {total:.1f}с")
    print(f"{'='*70}")
    print("Время по шагам:")
    for step, t in timings.items():
        print(f"  {step}: {t:.1f}с")
    print(f"\nАртефакты: {PROJECT_ROOT / 'artifacts'}")
    print(f"Отчёт:     {PROJECT_ROOT / 'docs' / 'experiments.md'}")


if __name__ == "__main__":
    main()
