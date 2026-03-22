"""
Эксперимент E3: Сравнение from-scratch GIN vs Pretrained GIN vs Pretrained GIN + XAI-loss.

Все модели обучаются в multi-task режиме на 12 задачах Tox21.
Scaffold split, метрика — среднее ROC-AUC по задачам.
"""

import json
import sys
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader_multitask import load_tox21_multitask, TOX21_TASKS
from src.data.splitter import scaffold_split
from src.data.featurizer_gin import build_gin_dataset
from src.models.gin_pretrained import train_gin, download_pretrained
from src.models.xai_loss import combined_xai_loss

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"


def run_experiment(seed: int = 42):
    """Запуск полного сравнительного эксперимента."""
    set_seed(seed)
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E3: from-scratch vs Pretrained GIN vs Pretrained+XAI")
    print("=" * 70)

    # ── Загрузка данных ──────────────────────────────────────────────
    print("\n[1/6] Загрузка Tox21 (multi-task)...")
    df = load_tox21_multitask()
    print(f"  Молекул: {len(df)}, Задач: {len(TOX21_TASKS)}")

    # ── Scaffold split ───────────────────────────────────────────────
    print("\n[2/6] Scaffold split (70/15/15)...")
    train_df, val_df, test_df = scaffold_split(df, seed=seed)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Фичеризация ──────────────────────────────────────────────────
    print("\n[3/6] Построение GIN-совместимых графов...")
    train_data = build_gin_dataset(train_df, TOX21_TASKS)
    val_data = build_gin_dataset(val_df, TOX21_TASKS)
    test_data = build_gin_dataset(test_df, TOX21_TASKS)
    print(f"  Графов: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    all_results = {}

    # ── Модель 1: GIN from scratch ───────────────────────────────────
    print("\n[4/6] Обучение GIN from scratch...")
    set_seed(seed)
    results_scratch, _ = train_gin(
        train_data, val_data, test_data,
        num_tasks=len(TOX21_TASKS),
        task_names=TOX21_TASKS,
        pretrained=None,
        lr=1e-3,
        epochs=100,
        patience=15,
        batch_size=64,
        save_dir=str(MODELS_DIR),
        model_name="gin_scratch",
    )
    all_results["gin_scratch"] = results_scratch

    # ── Модель 2: Pretrained GIN ─────────────────────────────────────
    print("\n[5/6] Обучение Pretrained GIN (supervised_contextpred)...")
    download_pretrained("supervised_contextpred")
    set_seed(seed)
    results_pretrained, _ = train_gin(
        train_data, val_data, test_data,
        num_tasks=len(TOX21_TASKS),
        task_names=TOX21_TASKS,
        pretrained="supervised_contextpred",
        lr=1e-3,
        epochs=100,
        patience=15,
        batch_size=64,
        save_dir=str(MODELS_DIR),
        model_name="gin_pretrained",
    )
    all_results["gin_pretrained"] = results_pretrained

    # ── Модель 3: Pretrained GIN + XAI loss ──────────────────────────
    print("\n[6/6] Обучение Pretrained GIN + Explainability-Aware Loss...")
    xai_fn = partial(combined_xai_loss, faith_weight=0.5, stab_weight=0.5)
    set_seed(seed)
    results_xai, _ = train_gin(
        train_data, val_data, test_data,
        num_tasks=len(TOX21_TASKS),
        task_names=TOX21_TASKS,
        pretrained="supervised_contextpred",
        xai_loss_fn=xai_fn,
        xai_lambda=0.1,
        lr=1e-3,
        epochs=100,
        patience=15,
        batch_size=64,
        save_dir=str(MODELS_DIR),
        model_name="gin_pretrained_xai",
    )
    all_results["gin_pretrained_xai"] = results_xai

    # ── Сводная таблица ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("СВОДКА РЕЗУЛЬТАТОВ (scaffold split)")
    print("=" * 70)
    print(f"{'Модель':<30} {'Val ROC-AUC':>12} {'Test ROC-AUC':>13} {'Test PR-AUC':>12}")
    print("-" * 70)
    for name, res in all_results.items():
        val_auc = res["val"]["mean_roc_auc"]
        test_auc = res["test"]["mean_roc_auc"]
        test_pr = res["test"]["mean_pr_auc"]
        print(f"{name:<30} {val_auc:>12.4f} {test_auc:>13.4f} {test_pr:>12.4f}")

    # ── Сохранение ───────────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e3_sota_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nРезультаты сохранены: {METRICS_DIR / 'e3_sota_comparison.json'}")

    return all_results


if __name__ == "__main__":
    run_experiment()
