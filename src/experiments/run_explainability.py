"""
Эксперименты по объяснимости PharmaKinetics: AOPC + IoU.

Запускается на scaffold-split тестовой выборке с использованием
лучшей GNN-модели из эксперимента E1.
"""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data.loader import load_dataset
from src.data.splitter import scaffold_split
from src.data.featurizer import build_graph_dataset
from src.models.gnn import MPNN, evaluate
from src.explain.perturbation import batch_aopc
from src.explain.stability import batch_stability

ARTIFACTS = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics"
MODELS_DIR = ARTIFACTS / "models"
EXPLAIN_DIR = ARTIFACTS / "explanations"


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ЭКСПЕРИМЕНТЫ ПО ОБЪЯСНИМОСТИ (AOPC + IoU)")
    print("=" * 70)

    df = load_dataset("tox21")
    _, _, test_df = scaffold_split(df, seed=42)

    print(f"Тестовая выборка: {len(test_df)} молекул")

    test_graphs = build_graph_dataset(test_df, mode="2d")
    if not test_graphs:
        print("ОШИБКА: нет валидных тестовых графов!")
        return

    node_dim = test_graphs[0].x.size(1)
    edge_dim = test_graphs[0].edge_attr.size(1) if test_graphs[0].edge_attr.size(0) > 0 else 7

    weights_path = MODELS_DIR / "scaffold" / "gnn_2d_weights.pt"
    if not weights_path.exists():
        print(f"ОШИБКА: веса модели не найдены: {weights_path}")
        print("Сначала запустите E1: python -m src.experiments.run_baseline_vs_graph")
        return

    model = MPNN(node_dim, edge_dim, hidden_dim=128, n_layers=3).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Модель загружена.")

    results = {}

    # ── AOPC (причинная состоятельность) ──
    print(f"\n{'─'*40}")
    print("Вычисление AOPC (faithfulness)...")
    print(f"{'─'*40}")

    for fraction in [0.10, 0.20]:
        print(f"\n  Доля маскируемых атомов: {fraction:.0%}")
        aopc_results = batch_aopc(
            model, test_graphs,
            fraction=fraction,
            device=device,
            max_molecules=300,
        )
        key = f"aopc_{int(fraction*100)}"
        results[key] = {
            "n_molecules": aopc_results["n_molecules"],
            "fraction": fraction,
            "mean_aopc_top": aopc_results["mean_aopc_top"],
            "mean_aopc_random": aopc_results["mean_aopc_random"],
            "mean_faithfulness_gain": aopc_results["mean_faithfulness_gain"],
            "std_faithfulness_gain": aopc_results["std_faithfulness_gain"],
        }
        print(f"  AOPC (top-{int(fraction*100)}%):                {aopc_results['mean_aopc_top']:.4f}")
        print(f"  AOPC (random-{int(fraction*100)}%):              {aopc_results['mean_aopc_random']:.4f}")
        print(f"  Прирост причинной состоятельности: {aopc_results['mean_faithfulness_gain']:.4f} "
              f"(+/- {aopc_results['std_faithfulness_gain']:.4f})")

    # ── IoU (устойчивость) ──
    print(f"\n{'─'*40}")
    print("Вычисление IoU (устойчивость)... (может занять несколько минут)")
    print(f"{'─'*40}")

    stability_results = batch_stability(
        model, test_graphs,
        fraction=0.20,
        n_random_smiles=20,
        device=device,
        max_molecules=300,
        mode="2d",
    )
    results["iou_20"] = {
        "n_molecules": stability_results.get("n_molecules", 0),
        "fraction": 0.20,
        "mean_iou": stability_results.get("mean_iou", float("nan")),
        "std_iou": stability_results.get("std_iou", float("nan")),
    }
    print(f"  IoU@20%: {stability_results.get('mean_iou', float('nan')):.4f} "
          f"(+/- {stability_results.get('std_iou', float('nan')):.4f})")
    print(f"  Молекул оценено: {stability_results.get('n_molecules', 0)}")

    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_DIR / "explainability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nРезультаты сохранены: {METRICS_DIR / 'explainability_results.json'}")

    # ── Итоги ──
    print(f"\n{'='*70}")
    print("ИТОГИ ОБЪЯСНИМОСТИ")
    print(f"{'='*70}")
    print(f"{'Метрика':<35} {'Значение':>10}")
    print("-" * 47)
    if "aopc_20" in results:
        print(f"{'AOPC top-20%':<35} {results['aopc_20']['mean_aopc_top']:>10.4f}")
        print(f"{'AOPC random-20%':<35} {results['aopc_20']['mean_aopc_random']:>10.4f}")
        print(f"{'Прирост faithfulness @20%':<35} {results['aopc_20']['mean_faithfulness_gain']:>10.4f}")
    if "iou_20" in results:
        print(f"{'IoU@20%':<35} {results['iou_20']['mean_iou']:>10.4f}")


if __name__ == "__main__":
    main()
