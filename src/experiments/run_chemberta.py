"""
Эксперимент E6: Fine-tuning ChemBERTa-2 на Tox21 (multi-seed).

Обучает ChemBERTa-77M-MLM на 12 задач Tox21 со scaffold split.
Сравнение с GIN-моделями из E4.

Запуск:
  python -m src.experiments.run_chemberta
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader_multitask import load_tox21_multitask, TOX21_TASKS
from src.data.splitter import scaffold_split
from src.models.chemberta import train_chemberta

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

SEEDS = [42, 0, 1]


def run_single_seed(seed, df, device=None):
    """Обучение ChemBERTa для одного seed."""
    set_seed(seed)
    train_df, val_df, test_df = scaffold_split(df, seed=seed)

    print(f"\n--- Seed {seed} ---")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_smiles = train_df["smiles"].tolist()
    val_smiles = val_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()

    train_labels = train_df[TOX21_TASKS].values.astype(float)
    val_labels = val_df[TOX21_TASKS].values.astype(float)
    test_labels = test_df[TOX21_TASKS].values.astype(float)

    results, model = train_chemberta(
        train_smiles, train_labels,
        val_smiles, val_labels,
        test_smiles, test_labels,
        num_tasks=12,
        task_names=TOX21_TASKS,
        lr=2e-5,
        epochs=20,
        patience=5,
        batch_size=32,
        device=device,
        save_dir=str(MODELS_DIR) if seed == 42 else None,
        model_name=f"chemberta_tox21_s{seed}",
    )

    return results


def run_experiment(device=None):
    """Запуск ChemBERTa на Tox21 с 3 seeds."""
    df = load_tox21_multitask()

    all_results = []
    for seed in SEEDS:
        res = run_single_seed(seed, df, device)
        all_results.append(res)

    test_aucs = [r["test"]["mean_roc_auc"] for r in all_results]
    val_aucs = [r["val"]["mean_roc_auc"] for r in all_results]

    summary = {
        "model": "ChemBERTa-77M-MLM",
        "dataset": "Tox21",
        "split": "scaffold",
        "seeds": SEEDS,
        "test_roc_auc_mean": float(np.mean(test_aucs)),
        "test_roc_auc_std": float(np.std(test_aucs)),
        "val_roc_auc_mean": float(np.mean(val_aucs)),
        "val_roc_auc_std": float(np.std(val_aucs)),
        "per_seed_results": all_results,
    }

    print(f"\n{'='*60}")
    print(f"ChemBERTa-2 на Tox21 (scaffold, {len(SEEDS)} seeds)")
    print(f"Test ROC-AUC: {summary['test_roc_auc_mean']:.4f} ± {summary['test_roc_auc_std']:.4f}")
    print(f"Val  ROC-AUC: {summary['val_roc_auc_mean']:.4f} ± {summary['val_roc_auc_std']:.4f}")
    print(f"{'='*60}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "e6_chemberta_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Результаты: {out_path}")

    return summary


if __name__ == "__main__":
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(device=dev)
