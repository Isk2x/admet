"""
Устойчивость объяснений PharmaKinetics (IoU).

Для каждой молекулы генерирует N эквивалентных SMILES-представлений,
вычисляет атомную атрибуцию для каждого и измеряет IoU
(пересечение / объединение) top-k% важных атомов.
"""

import numpy as np
from rdkit import Chem

from src.data.featurizer import smiles_to_graph, smiles_to_graph_3d
from src.explain.atom_importance import compute_atom_importance, get_top_k_atoms


def generate_random_smiles(smiles: str, n: int = 20, seed: int = 42) -> list:
    """Генерация N случайных (эквивалентных) SMILES для одной молекулы."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rng = np.random.RandomState(seed)
    results = set()
    results.add(Chem.MolToSmiles(mol))
    num_atoms = mol.GetNumAtoms()

    attempts = 0
    max_attempts = n * 20
    while len(results) < n and attempts < max_attempts:
        atom_order = rng.permutation(num_atoms).tolist()
        renumbered = Chem.RenumberAtoms(mol, atom_order)
        new_smi = Chem.MolToSmiles(renumbered, canonical=False)
        if Chem.MolFromSmiles(new_smi) is not None:
            results.add(new_smi)
        attempts += 1

    return list(results)[:n]


def _atom_order_mapping(smiles_ref: str, smiles_alt: str):
    """
    Маппинг атомных индексов из альтернативного SMILES
    в референсный (каноническая нумерация RDKit).
    """
    mol_ref = Chem.MolFromSmiles(smiles_ref)
    mol_alt = Chem.MolFromSmiles(smiles_alt)
    if mol_ref is None or mol_alt is None:
        return None

    canon_ref = Chem.MolToSmiles(mol_ref)
    canon_alt = Chem.MolToSmiles(mol_alt)
    if canon_ref != canon_alt:
        return None

    rank_ref = list(Chem.CanonicalRankAtoms(mol_ref))
    rank_alt = list(Chem.CanonicalRankAtoms(mol_alt))

    ref_rank_to_idx = {r: i for i, r in enumerate(rank_ref)}
    mapping = {}
    for alt_idx, alt_rank in enumerate(rank_alt):
        if alt_rank in ref_rank_to_idx:
            mapping[alt_idx] = ref_rank_to_idx[alt_rank]

    return mapping


def compute_iou(set_a: set, set_b: set) -> float:
    """Intersection over Union двух множеств."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_stability(
    model,
    smiles: str,
    fraction: float = 0.2,
    n_random_smiles: int = 20,
    device: str = "cpu",
    mode: str = "2d",
) -> dict:
    """
    Устойчивость объяснений для одной молекулы.

    Генерирует случайные SMILES, вычисляет атрибуции для каждого,
    маппит атомные индексы обратно в каноническую нумерацию
    и считает попарный IoU top-k% атомов.
    """
    graph_fn = smiles_to_graph if mode == "2d" else smiles_to_graph_3d

    canon_mol = Chem.MolFromSmiles(smiles)
    if canon_mol is None:
        return {"iou": float("nan"), "error": "невалидный SMILES"}
    canon_smi = Chem.MolToSmiles(canon_mol)

    ref_data = graph_fn(canon_smi, y=0)
    if ref_data is None:
        return {"iou": float("nan"), "error": "ошибка фичеризации"}

    ref_importance = compute_atom_importance(model, ref_data, device)
    ref_top = set(get_top_k_atoms(ref_importance, fraction).tolist())

    alt_smiles = generate_random_smiles(canon_smi, n=n_random_smiles)

    ious = []
    for alt_smi in alt_smiles:
        if alt_smi == canon_smi:
            continue

        alt_data = graph_fn(alt_smi, y=0)
        if alt_data is None:
            continue

        alt_importance = compute_atom_importance(model, alt_data, device)
        alt_top_raw = set(get_top_k_atoms(alt_importance, fraction).tolist())

        mapping = _atom_order_mapping(canon_smi, alt_smi)
        if mapping is None:
            continue

        alt_top_mapped = {mapping.get(idx, -1) for idx in alt_top_raw}
        alt_top_mapped.discard(-1)

        iou = compute_iou(ref_top, alt_top_mapped)
        ious.append(iou)

    if not ious:
        return {"iou": float("nan"), "n_variants": 0}

    return {
        "iou": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "n_variants": len(ious),
    }


def batch_stability(
    model,
    dataset: list,
    fraction: float = 0.2,
    n_random_smiles: int = 20,
    device: str = "cpu",
    max_molecules: int = 300,
    mode: str = "2d",
) -> dict:
    """Агрегированная устойчивость объяснений по батчу молекул."""
    results = []
    n = min(len(dataset), max_molecules)

    for i in range(n):
        data = dataset[i]
        if data.x.size(0) < 3:
            continue

        smi = data.smiles
        res = compute_stability(model, smi, fraction, n_random_smiles, device, mode)
        if not np.isnan(res["iou"]):
            results.append(res)

    if not results:
        return {"error": "Нет валидных молекул для оценки устойчивости"}

    return {
        "n_molecules": len(results),
        "fraction": fraction,
        "mean_iou": float(np.mean([r["iou"] for r in results])),
        "std_iou": float(np.std([r["iou"] for r in results])),
    }
