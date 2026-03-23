"""
3D фичеризация молекул для PharmaKinetics.

Конвертирует SMILES → PyG Data с 3D-координатами и атомными номерами
для SE(3)-инвариантных моделей (SchNet и др.).

Два режима:
  1. Single conformer: одна ETKDG-конформация на молекулу
  2. Multi-conformer: K конформаций на молекулу для ансамблирования

В отличие от pseudo-3D (featurizer.py), здесь:
  - НЕТ ручных edge_attr — SchNet строит рёбра по cutoff radius
  - pos: (N, 3) координаты тяжёлых атомов
  - z: (N,) атомные номера
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem


def _generate_conformer(mol_h, seed=42, max_attempts=3):
    """
    Генерация одной конформации через ETKDG.
    Возвращает (conformer, success: bool).
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True

    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        return None, False

    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass

    return mol_h.GetConformer(), True


def _generate_multi_conformers(mol_h, num_confs=5, seed=42):
    """
    Генерация нескольких конформаций через ETKDG.
    Возвращает список Conformer (может быть пустым при неудаче).
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = 0.5

    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
    if len(conf_ids) == 0:
        return []

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=200)
    except Exception:
        pass

    return [mol_h.GetConformer(cid) for cid in conf_ids]


def _extract_heavy_atom_coords(mol, mol_h, conformer):
    """
    Извлечение координат тяжёлых атомов из конформера с Hs.
    mol — без Hs (для индексации), mol_h — с Hs (для координат).
    """
    num_heavy = mol.GetNumAtoms()
    pos = np.zeros((num_heavy, 3), dtype=np.float32)
    for i in range(num_heavy):
        pt = conformer.GetAtomPosition(i)
        pos[i] = [pt.x, pt.y, pt.z]
    return pos


def smiles_to_3d(smiles: str, y_tasks: np.ndarray = None, seed=42) -> Data:
    """
    SMILES → PyG Data с 3D-координатами (single conformer).

    Возвращает Data(z, pos, y, smiles) или None при неудаче.
    SchNet сам строит radius graph по cutoff — edge_index не нужен.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    conformer, ok = _generate_conformer(mol_h, seed=seed)

    if not ok:
        return None

    pos = _extract_heavy_atom_coords(mol, mol_h, conformer)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)

    data = Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float),
        smiles=smiles,
    )

    if y_tasks is not None:
        data.y = torch.tensor(y_tasks, dtype=torch.float).unsqueeze(0)
    else:
        data.y = torch.zeros(1, 1, dtype=torch.float)

    return data


def smiles_to_3d_multi(smiles: str, y_tasks: np.ndarray = None,
                       num_confs: int = 5, seed: int = 42) -> list:
    """
    SMILES → список PyG Data (по одному на конформер).

    Каждый Data содержит одинаковые z/y/smiles, но разные pos.
    Если ни один конформер не сгенерирован — возвращает пустой список.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol_h = Chem.AddHs(mol)
    conformers = _generate_multi_conformers(mol_h, num_confs=num_confs, seed=seed)

    if not conformers:
        return []

    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    z_tensor = torch.tensor(z, dtype=torch.long)

    if y_tasks is not None:
        y_tensor = torch.tensor(y_tasks, dtype=torch.float).unsqueeze(0)
    else:
        y_tensor = torch.zeros(1, 1, dtype=torch.float)

    results = []
    for conf in conformers:
        pos = _extract_heavy_atom_coords(mol, mol_h, conf)
        data = Data(
            z=z_tensor.clone(),
            pos=torch.tensor(pos, dtype=torch.float),
            y=y_tensor.clone(),
            smiles=smiles,
        )
        results.append(data)

    return results


def build_3d_dataset(df, task_columns=None, seed=42):
    """
    Построение датасета с 3D-координатами из DataFrame.
    Молекулы, для которых конформер не сгенерирован, пропускаются.
    """
    dataset = []
    failed = 0
    for _, row in df.iterrows():
        if task_columns is not None:
            y_tasks = row[task_columns].values.astype(float)
        else:
            y_tasks = np.array([row.get("y", 0.0)])

        data = smiles_to_3d(row["smiles"], y_tasks, seed=seed)
        if data is not None:
            dataset.append(data)
        else:
            failed += 1

    if failed > 0:
        print(f"  3D: {failed}/{len(df)} молекул без конформера (пропущены)")
    return dataset


def build_3d_multi_dataset(df, task_columns=None, num_confs=5, seed=42):
    """
    Построение multi-conformer датасета из DataFrame.
    Возвращает list[list[Data]] — для каждой молекулы список конформеров.
    """
    dataset = []
    failed = 0
    for _, row in df.iterrows():
        if task_columns is not None:
            y_tasks = row[task_columns].values.astype(float)
        else:
            y_tasks = np.array([row.get("y", 0.0)])

        confs = smiles_to_3d_multi(row["smiles"], y_tasks,
                                   num_confs=num_confs, seed=seed)
        if confs:
            dataset.append(confs)
        else:
            failed += 1

    if failed > 0:
        print(f"  3D-multi: {failed}/{len(df)} молекул без конформеров (пропущены)")
    return dataset
