"""
Эксперимент E1: Бейзлайн (ECFP4 + LR / XGB) vs Графовая модель (MPNN).

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
from src.models.baseline import train_and_evaluate as train_baseline
from src.models.gnn import train_gnn

ARTIFACTS = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"
MODELS_DIR = ARTIFACTS / "models"


def run_experiment(split_name, split_fn, df):
    print(f"\n{'#'*70}")
    print(f"# Разбиение: {split_name}")
    print(f"{'#'*70}")

    train_df, val_df, test_df = split_fn(df, seed=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    results = {}

    # ── Бейзлайн: LogisticRegression ──
    lr_results, _, _ = train_baseline(
        train_df, val_df, test_df,
        model_type="lr",
        save_dir=str(MODELS_DIR / split_name),
    )
    results["ecfp4_lr"] = lr_results

    # ── Бейзлайн: XGBoost ──
    xgb_results, _, _ = train_baseline(
        train_df, val_df, test_df,
        model_type="xgb",
        save_dir=str(MODELS_DIR / split_name),
    )
    results["ecfp4_xgb"] = xgb_results

    # ── GNN (2D) ──
    print("\nПостроение 2D графового датасета...")
    train_graphs = build_graph_dataset(train_df, mode="2d")
    val_graphs = build_graph_dataset(val_df, mode="2d")
    test_graphs = build_graph_dataset(test_df, mode="2d")

    if not train_graphs:
        print("ОШИБКА: нет валидных тренировочных графов!")
        return results

    node_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1) if train_graphs[0].edge_attr.size(0) > 0 else 7

    gnn_results, _, _ = train_gnn(
        train_graphs, val_graphs, test_graphs,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        n_layers=3,
        lr=1e-3,
        epochs=100,
        patience=15,
        batch_size=64,
        save_dir=str(MODELS_DIR / split_name),
        model_name="gnn_2d",
    )
    results["gnn_2d"] = gnn_results

    return results


def main():
    set_seed(42)
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E1: Бейзлайн vs Графовая модель")
    print("=" * 70)

    df = load_dataset("tox21")

    all_results = {}

    for split_name, split_fn in [("random", random_split), ("scaffold", scaffold_split)]:
        all_results[split_name] = run_experiment(split_name, split_fn, df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e1_baseline_vs_graph.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("ИТОГИ E1")
    print(f"{'='*70}")
    print(f"{'Модель':<20} {'Split':<12} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 55)
    for split_name in ["random", "scaffold"]:
        for model_name in ["ecfp4_lr", "ecfp4_xgb", "gnn_2d"]:
            r = all_results[split_name].get(model_name, {})
            test = r.get("test", {})
            roc = test.get("roc_auc", float("nan"))
            pr = test.get("pr_auc", float("nan"))
            print(f"{model_name:<20} {split_name:<12} {roc:>10.4f} {pr:>10.4f}")

    print(f"\nРезультаты сохранены: {METRICS_DIR / 'e1_baseline_vs_graph.json'}")


if __name__ == "__main__":
    main()
