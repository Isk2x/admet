"""
Модуль фичеризации молекул PharmaKinetics.

Поддерживает три представления:
  - ECFP4: битовые отпечатки Моргана (radius=2, 2048 бит)
  - 2D граф: молекулярный граф для PyTorch Geometric (атомные + связевые признаки)
  - 3D граф: молекулярный граф + RBF-кодированные межатомные расстояния из конформеров
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs


# ── ECFP4 (отпечатки Моргана) ──────────────────────────────────────────

def smiles_to_ecfp4(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Конвертация SMILES в ECFP4 (Morgan radius=2) битовый вектор."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_ecfp4(smiles_list, n_bits: int = 2048) -> np.ndarray:
    """Батчевая конвертация списка SMILES в матрицу ECFP4 (N x n_bits)."""
    return np.stack([smiles_to_ecfp4(s, n_bits) for s in smiles_list])


# ── Атомные и связевые признаки для GNN ────────────────────────────────

ATOM_FEATURES = {
    "atomic_num": list(range(1, 120)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "is_aromatic": [False, True],
}

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, choices):
    """One-hot кодирование с категорией 'неизвестно'."""
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def atom_features(atom) -> list:
    """Вектор признаков атома: номер, степень, заряд, гибридизация, ароматичность."""
    return (
        one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
    )


def bond_features(bond) -> list:
    """Вектор признаков связи: тип, конъюгированность, принадлежность кольцу."""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [1 if bond.GetIsConjugated() else 0]
        + [1 if bond.IsInRing() else 0]
    )


# ── 2D молекулярный граф ───────────────────────────────────────────────

def smiles_to_graph(smiles: str, y: int = 0) -> Data:
    """Конвертация SMILES в PyG Data (2D молекулярный граф)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles,
    )
    return data


# ── 3D молекулярный граф (pseudo-3D с конформерами) ────────────────────

NUM_RBF = 16
RBF_MIN = 0.0
RBF_MAX = 10.0


def rbf_expansion(distances: np.ndarray, num_rbf=NUM_RBF,
                  d_min=RBF_MIN, d_max=RBF_MAX) -> np.ndarray:
    """Разложение расстояний по радиальным базисным функциям (Гаусс)."""
    centers = np.linspace(d_min, d_max, num_rbf)
    gamma = 1.0 / ((d_max - d_min) / num_rbf) ** 2
    return np.exp(-gamma * (distances[:, None] - centers[None, :]) ** 2)


def smiles_to_graph_3d(smiles: str, y: int = 0) -> Data:
    """
    Конвертация SMILES в PyG Data с 3D-признаками на рёбрах.

    Использует RDKit ETKDG для генерации конформеров.
    Если генерация не удалась — fallback на дефолтное расстояние 1.5 A.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    conf_ok = AllChem.EmbedMolecule(mol_h, params)
    if conf_ok == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        conformer = mol_h.GetConformer()
        has_3d = True
    else:
        has_3d = False

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr_2d = []
    distances = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr_2d.extend([bf, bf])
        if has_3d:
            pos_i = conformer.GetAtomPosition(i)
            pos_j = conformer.GetAtomPosition(j)
            dist = pos_i.Distance(pos_j)
            distances.extend([dist, dist])
        else:
            distances.extend([1.5, 1.5])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7 + NUM_RBF), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_2d = torch.tensor(edge_attr_2d, dtype=torch.float)
        dist_rbf = torch.tensor(
            rbf_expansion(np.array(distances)), dtype=torch.float
        )
        edge_attr = torch.cat([edge_attr_2d, dist_rbf], dim=1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles,
        has_3d=has_3d,
    )
    return data


# ── Построение датасета графов ─────────────────────────────────────────

def build_graph_dataset(df, mode="2d"):
    """Построение списка PyG Data из DataFrame. mode: '2d' или '3d'."""
    fn = smiles_to_graph if mode == "2d" else smiles_to_graph_3d
    dataset = []
    failed = 0
    for _, row in df.iterrows():
        data = fn(row["smiles"], int(row["y"]))
        if data is not None:
            dataset.append(data)
        else:
            failed += 1
    if failed > 0:
        print(f"  Предупреждение: {failed}/{len(df)} молекул не прошли фичеризацию")
    return dataset
