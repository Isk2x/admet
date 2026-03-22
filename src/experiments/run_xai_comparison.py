"""
Сравнение объяснимости (AOPC/IoU) для трёх GIN-моделей.

Загружает обученные веса из artifacts/models и вычисляет:
  - AOPC (faithfulness): насколько маскирование важных атомов меняет предсказание
  - IoU (stability): устойчивость объяснений при перенумерации атомов
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader_multitask import load_tox21_multitask, TOX21_TASKS
from src.data.splitter import scaffold_split
from src.data.featurizer_gin import build_gin_dataset, smiles_to_graph_gin
from src.models.gin_pretrained import GINMultiTask
from src.explain.atom_importance_gin import compute_atom_importance_gin, get_top_k_atoms_gin

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

MODEL_CONFIGS = {
    "gin_scratch": {"pretrained": None},
    "gin_pretrained": {"pretrained": "supervised_contextpred"},
    "gin_pretrained_xai": {"pretrained": "supervised_contextpred"},
}


def compute_aopc_gin(model, data, importance, fraction, device, n_random=20):
    """AOPC для одной молекулы (GIN-совместимый)."""
    model.eval()

    def predict(d):
        d = d.clone().to(device)
        d.batch = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(d)
            return torch.sigmoid(out).mean().item()

    original_pred = predict(data)
    n_atoms = len(importance)
    k = max(1, int(n_atoms * fraction))

    top_idx = get_top_k_atoms_gin(importance, fraction)

    masked = data.clone()
    node_emb = model.get_node_embeddings()
    h = model.backbone(data.clone().to(device).x,
                       data.clone().to(device).edge_index,
                       data.clone().to(device).edge_attr)
    h_masked = h.clone()
    for idx in top_idx:
        h_masked[idx] = 0.0
    from torch_geometric.nn import global_mean_pool
    batch_idx = torch.zeros(h_masked.size(0), dtype=torch.long, device=device)
    graph_repr = global_mean_pool(h_masked, batch_idx)
    pred_masked = torch.sigmoid(model.classifier(graph_repr)).mean().item()
    aopc_top = original_pred - pred_masked

    rng = np.random.RandomState(42)
    random_drops = []
    for _ in range(n_random):
        rand_idx = rng.choice(n_atoms, size=k, replace=False)
        h_rand = h.clone()
        for idx in rand_idx:
            h_rand[idx] = 0.0
        graph_repr_r = global_mean_pool(h_rand, batch_idx)
        pred_rand = torch.sigmoid(model.classifier(graph_repr_r)).mean().item()
        random_drops.append(original_pred - pred_rand)
    aopc_random = float(np.mean(random_drops))

    return {
        "original_pred": original_pred,
        "aopc_top": aopc_top,
        "aopc_random": aopc_random,
        "faithfulness_gain": aopc_top - aopc_random,
    }


def compute_stability_gin(model, smiles, fraction, device, n_random_smiles=10):
    """IoU-стабильность объяснений через перенумерацию атомов."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"iou": float("nan")}
    canon_smi = Chem.MolToSmiles(mol)
    num_atoms = mol.GetNumAtoms()

    ref_data = smiles_to_graph_gin(canon_smi)
    if ref_data is None:
        return {"iou": float("nan")}

    ref_imp = compute_atom_importance_gin(model, ref_data, device)
    ref_top = set(get_top_k_atoms_gin(ref_imp, fraction).tolist())

    rng = np.random.RandomState(42)
    ious = []
    for _ in range(n_random_smiles * 3):
        if len(ious) >= n_random_smiles:
            break
        atom_order = rng.permutation(num_atoms).tolist()
        renumbered = Chem.RenumberAtoms(mol, atom_order)
        alt_smi = Chem.MolToSmiles(renumbered, canonical=False)
        if alt_smi == canon_smi:
            continue

        alt_data = smiles_to_graph_gin(alt_smi)
        if alt_data is None:
            continue

        alt_imp = compute_atom_importance_gin(model, alt_data, device)
        alt_top_raw = set(get_top_k_atoms_gin(alt_imp, fraction).tolist())

        rank_ref = list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(canon_smi)))
        rank_alt = list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(alt_smi)))
        ref_map = {r: i for i, r in enumerate(rank_ref)}
        mapping = {}
        for alt_idx, alt_rank in enumerate(rank_alt):
            if alt_rank in ref_map:
                mapping[alt_idx] = ref_map[alt_rank]

        alt_top_mapped = {mapping.get(idx, -1) for idx in alt_top_raw}
        alt_top_mapped.discard(-1)

        if ref_top and alt_top_mapped:
            intersection = len(ref_top & alt_top_mapped)
            union = len(ref_top | alt_top_mapped)
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

    if not ious:
        return {"iou": float("nan"), "n_variants": 0}

    return {
        "iou": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "n_variants": len(ious),
    }


def run_xai_comparison(seed: int = 42, max_molecules: int = 200, fraction: float = 0.2):
    """Сравнение объяснимости для трёх моделей."""
    set_seed(seed)
    device = "cpu"

    print("=" * 70)
    print("СРАВНЕНИЕ ОБЪЯСНИМОСТИ: AOPC / IoU")
    print("=" * 70)

    df = load_tox21_multitask()
    _, _, test_df = scaffold_split(df, seed=seed)
    test_data = build_gin_dataset(test_df, TOX21_TASKS)
    print(f"Тестовых молекул: {len(test_data)}")

    n = min(len(test_data), max_molecules)
    results = {}

    for model_name, config in MODEL_CONFIGS.items():
        weights_path = MODELS_DIR / f"{model_name}_weights.pt"
        if not weights_path.exists():
            print(f"\n  [{model_name}] Веса не найдены: {weights_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"Модель: {model_name}")
        print(f"{'─'*60}")

        model = GINMultiTask(num_tasks=len(TOX21_TASKS))
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        aopc_results = []
        iou_results = []

        for i in range(n):
            data = test_data[i]
            if data.x.size(0) < 3:
                continue

            imp = compute_atom_importance_gin(model, data, device)
            aopc = compute_aopc_gin(model, data, imp, fraction, device)
            aopc_results.append(aopc)

            iou = compute_stability_gin(model, data.smiles, fraction, device, n_random_smiles=5)
            if not np.isnan(iou.get("iou", float("nan"))):
                iou_results.append(iou)

            if (i + 1) % 50 == 0:
                print(f"  Обработано {i+1}/{n} молекул...")

        metrics = {
            "n_molecules_aopc": len(aopc_results),
            "n_molecules_iou": len(iou_results),
            "mean_aopc_top": float(np.mean([r["aopc_top"] for r in aopc_results])) if aopc_results else float("nan"),
            "mean_aopc_random": float(np.mean([r["aopc_random"] for r in aopc_results])) if aopc_results else float("nan"),
            "mean_faithfulness_gain": float(np.mean([r["faithfulness_gain"] for r in aopc_results])) if aopc_results else float("nan"),
            "mean_iou": float(np.mean([r["iou"] for r in iou_results])) if iou_results else float("nan"),
        }
        results[model_name] = metrics

        print(f"  AOPC (top-{int(fraction*100)}%): {metrics['mean_aopc_top']:.4f}")
        print(f"  AOPC (random):  {metrics['mean_aopc_random']:.4f}")
        print(f"  Faithfulness:   {metrics['mean_faithfulness_gain']:.4f}")
        print(f"  IoU:            {metrics['mean_iou']:.4f}")

    print(f"\n{'='*70}")
    print("СВОДНАЯ ТАБЛИЦА ОБЪЯСНИМОСТИ")
    print(f"{'='*70}")
    print(f"{'Модель':<30} {'AOPC↑':>8} {'Faith.↑':>9} {'IoU↑':>8}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<30} {m['mean_aopc_top']:>8.4f} {m['mean_faithfulness_gain']:>9.4f} {m['mean_iou']:>8.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e3_xai_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nСохранено: {METRICS_DIR / 'e3_xai_comparison.json'}")

    return results


if __name__ == "__main__":
    run_xai_comparison()
