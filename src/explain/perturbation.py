"""
AOPC (Area Over Perturbation Curve) — тест причинной состоятельности.

Измеряет faithfulness: если важные атомы замаскированы (зануление признаков),
предсказание должно упасть сильнее, чем при маскировании случайных атомов.

AOPC@k% = среднее падение предсказания при маскировании top-k% атомов.
Прирост причинной состоятельности = AOPC_top - AOPC_random.
"""

import torch
import numpy as np
from torch_geometric.data import Data

from src.explain.atom_importance import compute_atom_importance, get_top_k_atoms


def _predict_single(model, data: Data, device: str) -> float:
    """Sigmoid-предсказание для одного графа."""
    model.eval()
    with torch.no_grad():
        data = data.clone().to(device)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        out = model(data)
        return torch.sigmoid(out).item()


def _mask_atoms(data: Data, atom_indices: np.ndarray) -> Data:
    """Зануление признаков указанных атомов."""
    masked = data.clone()
    masked.x = masked.x.clone()
    for idx in atom_indices:
        masked.x[idx] = 0.0
    return masked


def compute_aopc(
    model,
    data: Data,
    importance: np.ndarray,
    fraction: float = 0.2,
    device: str = "cpu",
    n_random_trials: int = 20,
) -> dict:
    """
    AOPC для одной молекулы.

    Возвращает:
        aopc_top: падение при маскировании top-k% важных атомов
        aopc_random: среднее падение при маскировании случайных k% атомов
        faithfulness_gain: aopc_top - aopc_random
    """
    original_pred = _predict_single(model, data, device)
    n_atoms = len(importance)
    k = max(1, int(n_atoms * fraction))

    # Маскирование top-k
    top_idx = get_top_k_atoms(importance, fraction)
    masked_data = _mask_atoms(data, top_idx)
    top_pred = _predict_single(model, masked_data, device)
    aopc_top = original_pred - top_pred

    # Случайный бейзлайн (среднее по n_random_trials)
    random_drops = []
    rng = np.random.RandomState(42)
    for _ in range(n_random_trials):
        rand_idx = rng.choice(n_atoms, size=k, replace=False)
        masked_data = _mask_atoms(data, rand_idx)
        rand_pred = _predict_single(model, masked_data, device)
        random_drops.append(original_pred - rand_pred)
    aopc_random = float(np.mean(random_drops))

    return {
        "original_pred": original_pred,
        "aopc_top": aopc_top,
        "aopc_random": aopc_random,
        "faithfulness_gain": aopc_top - aopc_random,
        "k": k,
        "n_atoms": n_atoms,
    }


def batch_aopc(
    model,
    dataset: list,
    fraction: float = 0.2,
    device: str = "cpu",
    max_molecules: int = 300,
) -> dict:
    """
    Агрегированные метрики AOPC по батчу молекул.

    Возвращает средние AOPC_top, AOPC_random, прирост причинной состоятельности.
    """
    results = []
    n = min(len(dataset), max_molecules)

    for i in range(n):
        data = dataset[i]
        if data.x.size(0) < 3:
            continue

        importance = compute_atom_importance(model, data, device)
        aopc = compute_aopc(model, data, importance, fraction, device)
        results.append(aopc)

    if not results:
        return {"error": "Нет валидных молекул для AOPC"}

    return {
        "n_molecules": len(results),
        "fraction": fraction,
        "mean_aopc_top": float(np.mean([r["aopc_top"] for r in results])),
        "mean_aopc_random": float(np.mean([r["aopc_random"] for r in results])),
        "mean_faithfulness_gain": float(np.mean([r["faithfulness_gain"] for r in results])),
        "std_faithfulness_gain": float(np.std([r["faithfulness_gain"] for r in results])),
        "per_molecule": results,
    }
