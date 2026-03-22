"""
Эксперимент E2: 2D графовая модель (MPNN) vs Pseudo-3D графовая модель.

Оценка на random и scaffold split'ах.
Результаты сохраняются в artifacts/metrics/.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data.loader import load_dataset
from src.data.splitter import random_split, scaffold_split
from src.data.featurizer import build_graph_dataset
from src.models.gnn import train_gnn
from src.models.gnn_3d import train_gnn_3d

ARTIFACTS = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"
MODELS_DIR = ARTIFACTS / "models"


def run_experiment(split_name, split_fn, df):
    print(f"\n{'#'*70}")
    print(f"# Разбиение: {split_name}")
    print(f"{'#'*70}")

    train_df, val_df, test_df = split_fn(df, seed=42)

    results = {}

    # ── GNN 2D ──
    print("\nПостроение 2D графового датасета...")
    train_2d = build_graph_dataset(train_df, mode="2d")
    val_2d = build_graph_dataset(val_df, mode="2d")
    test_2d = build_graph_dataset(test_df, mode="2d")

    node_dim = train_2d[0].x.size(1)
    edge_dim_2d = train_2d[0].edge_attr.size(1) if train_2d[0].edge_attr.size(0) > 0 else 7

    gnn_2d_results, _, _ = train_gnn(
        train_2d, val_2d, test_2d,
        node_dim=node_dim,
        edge_dim=edge_dim_2d,
        hidden_dim=128,
        n_layers=3,
        epochs=100,
        patience=15,
        save_dir=str(MODELS_DIR / split_name),
        model_name="gnn_2d_e2",
    )
    results["gnn_2d"] = gnn_2d_results

    # ── GNN 3D ──
    print("\nПостроение 3D графового датасета (может занять несколько минут)...")
    train_3d = build_graph_dataset(train_df, mode="3d")
    val_3d = build_graph_dataset(val_df, mode="3d")
    test_3d = build_graph_dataset(test_df, mode="3d")

    edge_dim_3d = train_3d[0].edge_attr.size(1) if train_3d[0].edge_attr.size(0) > 0 else 23

    n_3d = sum(1 for d in train_3d + val_3d + test_3d if getattr(d, "has_3d", False))
    n_total = len(train_3d) + len(val_3d) + len(test_3d)
    print(f"  Успешность генерации 3D-конформеров: {n_3d}/{n_total} ({100*n_3d/n_total:.1f}%)")

    gnn_3d_results, _, _ = train_gnn_3d(
        train_3d, val_3d, test_3d,
        node_dim=node_dim,
        edge_dim=edge_dim_3d,
        hidden_dim=128,
        n_layers=3,
        epochs=100,
        patience=15,
        save_dir=str(MODELS_DIR / split_name),
        model_name="gnn_3d",
    )
    results["gnn_3d"] = gnn_3d_results

    return results


def main():
    set_seed(42)
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E2: Граф (2D) vs Pseudo-3D графовая модель")
    print("=" * 70)

    df = load_dataset("tox21")

    all_results = {}

    for split_name, split_fn in [("random", random_split), ("scaffold", scaffold_split)]:
        all_results[split_name] = run_experiment(split_name, split_fn, df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e2_graph_vs_3d.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("ИТОГИ E2")
    print(f"{'='*70}")
    print(f"{'Модель':<20} {'Split':<12} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 55)
    for split_name in ["random", "scaffold"]:
        for model_name in ["gnn_2d", "gnn_3d"]:
            r = all_results[split_name].get(model_name, {})
            test = r.get("test", {})
            roc = test.get("roc_auc", float("nan"))
            pr = test.get("pr_auc", float("nan"))
            print(f"{model_name:<20} {split_name:<12} {roc:>10.4f} {pr:>10.4f}")

    print(f"\nРезультаты сохранены: {METRICS_DIR / 'e2_graph_vs_3d.json'}")


if __name__ == "__main__":
    main()
