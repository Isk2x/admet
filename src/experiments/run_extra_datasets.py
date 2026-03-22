"""
Эксперимент E5: Лучшая модель (GIN+VN) на ClinTox и BBBP.

Подтверждает обобщаемость подхода на других MoleculeNet датасетах.
Multi-seed (3 seeds), scaffold split.
"""

import json
import sys
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader import load_dataset, DATASETS
from src.data.splitter import scaffold_split
from src.data.featurizer_gin import build_gin_dataset
from src.models.gin_pretrained import train_gin

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

SEEDS = [42, 0, 1]

EXTRA_DATASETS = {
    "clintox": {
        "task_columns": ["FDA_APPROVED", "CT_TOX"],
    },
    "bbbp": {
        "task_columns": ["p_np"],
    },
}


def load_multitask_dataset(name: str) -> tuple:
    """Загрузка датасета с multi-task метками."""
    config = DATASETS[name]
    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / config["filename"]
    from src.data.loader import RAW_DIR, _download

    if not raw_path.exists():
        _download(config["url"], raw_path, config["compressed"])

    df = pd.read_csv(raw_path)
    task_cols = EXTRA_DATASETS[name]["task_columns"]

    canonical = []
    task_rows = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[config["smiles_col"]]))
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)

        tasks = row[task_cols].values.astype(float)
        if np.all(np.isnan(tasks)):
            continue

        canonical.append(smi)
        task_rows.append(tasks)

    result = pd.DataFrame({"smiles": canonical})
    task_df = pd.DataFrame(task_rows, columns=task_cols)
    result = pd.concat([result, task_df], axis=1)
    result = result.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    print(f"[{name}] {len(result)} молекул, задач: {len(task_cols)}")
    for tc in task_cols:
        valid = result[tc].dropna()
        print(f"  {tc}: {len(valid)} аннотаций, pos_rate={valid.mean():.3f}")

    return result, task_cols


def run_dataset(name: str, seeds: list):
    """Запуск GIN+VN на одном датасете с несколькими seeds."""
    df, task_cols = load_multitask_dataset(name)

    all_results = []

    for seed in seeds:
        set_seed(seed)
        train_df, val_df, test_df = scaffold_split(df, seed=seed)
        print(f"\n  Seed {seed}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        train_data = build_gin_dataset(train_df, task_cols)
        val_data = build_gin_dataset(val_df, task_cols)
        test_data = build_gin_dataset(test_df, task_cols)

        results, _ = train_gin(
            train_data, val_data, test_data,
            num_tasks=len(task_cols),
            task_names=task_cols,
            pretrained=None,
            lr=1e-3,
            epochs=100,
            patience=15,
            batch_size=64,
            model_name=f"gin_vn_{name}_s{seed}",
            backbone_type="vn",
            pool_type="mean",
        )
        all_results.append(results)

    test_aucs = [r["test"]["mean_roc_auc"] for r in all_results]
    test_praucs = [r["test"]["mean_pr_auc"] for r in all_results]

    return {
        "dataset": name,
        "num_tasks": len(task_cols),
        "task_columns": task_cols,
        "test_roc_auc_mean": float(np.mean(test_aucs)),
        "test_roc_auc_std": float(np.std(test_aucs)),
        "test_pr_auc_mean": float(np.mean(test_praucs)),
        "test_pr_auc_std": float(np.std(test_praucs)),
        "n_seeds": len(seeds),
        "per_seed": {str(s): r for s, r in zip(seeds, all_results)},
    }


def run_experiment():
    """Запуск на всех дополнительных датасетах."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E5: GIN+VN на ClinTox и BBBP")
    print("=" * 70)

    results = {}
    for name in EXTRA_DATASETS:
        print(f"\n{'#'*60}")
        print(f"# Датасет: {name}")
        print(f"{'#'*60}")
        results[name] = run_dataset(name, SEEDS)

    print(f"\n{'='*70}")
    print("СВОДКА (scaffold split, 3 seeds)")
    print(f"{'='*70}")
    print(f"{'Датасет':<15} {'Tasks':>6} {'Test ROC-AUC':>18} {'Test PR-AUC':>18}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<15} {r['num_tasks']:>6} "
              f"{r['test_roc_auc_mean']:.4f}±{r['test_roc_auc_std']:.4f}  "
              f"{r['test_pr_auc_mean']:.4f}±{r['test_pr_auc_std']:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e5_extra_datasets.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nСохранено: {METRICS_DIR / 'e5_extra_datasets.json'}")

    return results


if __name__ == "__main__":
    run_experiment()
