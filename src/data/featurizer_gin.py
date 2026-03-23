"""
GIN-совместимая фичеризация молекул для PharmaKinetics.

Два режима:
  1. Pretrained-совместимый (2 атомных + 2 bond признака):
     Hu et al. 2020 — [atomic_num, chirality_tag], [bond_type, bond_dir]
  2. OGB-расширенный (9 атомных + 4 bond признака):
     [atomic_num, chirality, degree, formal_charge, num_hs,
      num_radical_e, hybridization, is_aromatic, is_in_ring]
     [bond_type, bond_dir, bond_stereo, is_conjugated]
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE: 0,
    Chem.rdchem.BondDir.ENDUPRIGHT: 1,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 2,
}

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    Chem.rdchem.HybridizationType.S: 6,
}

BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def smiles_to_graph_gin(smiles: str, y_tasks: np.ndarray = None) -> Data:
    """
    Конвертация SMILES в PyG Data для pretrained GIN.

    Формат: 2 атомных + 2 bond целочисленных признака.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),
            int(atom.GetChiralTag()),
        ])
    x = torch.tensor(x, dtype=torch.long)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        bd = BOND_DIR_MAP.get(bond.GetBondDir(), 0)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([[bt, bd], [bt, bd]])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

    if y_tasks is not None:
        data.y = torch.tensor(y_tasks, dtype=torch.float).unsqueeze(0)
    else:
        data.y = torch.zeros(1, 1, dtype=torch.float)

    return data


def smiles_to_graph_gin_ogb(smiles: str, y_tasks: np.ndarray = None) -> Data:
    """
    Конвертация SMILES в PyG Data с OGB-расширенными признаками.

    Формат: 9 атомных + 4 bond целочисленных признака.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for atom in mol.GetAtoms():
        x.append([
            min(atom.GetAtomicNum(), 119),
            int(atom.GetChiralTag()),
            min(atom.GetTotalDegree(), 10),
            min(atom.GetFormalCharge() + 5, 10),
            min(atom.GetTotalNumHs(), 8),
            min(atom.GetNumRadicalElectrons(), 5),
            HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ])
    x = torch.tensor(x, dtype=torch.long)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        bd = BOND_DIR_MAP.get(bond.GetBondDir(), 0)
        bs = BOND_STEREO_MAP.get(bond.GetStereo(), 0)
        bc = int(bond.GetIsConjugated())
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([[bt, bd, bs, bc], [bt, bd, bs, bc]])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

    if y_tasks is not None:
        data.y = torch.tensor(y_tasks, dtype=torch.float).unsqueeze(0)
    else:
        data.y = torch.zeros(1, 1, dtype=torch.float)

    return data


def build_gin_dataset(df, task_columns=None, ogb_features=False):
    """
    Построение GIN-совместимого датасета из DataFrame.

    Параметры:
        df: DataFrame с колонкой 'smiles' и (опционально) колонками задач
        task_columns: список колонок с метками задач
        ogb_features: True → OGB-расширенные признаки (9 atom + 4 bond)
    """
    converter = smiles_to_graph_gin_ogb if ogb_features else smiles_to_graph_gin
    dataset = []
    failed = 0
    for _, row in df.iterrows():
        if task_columns is not None:
            y_tasks = row[task_columns].values.astype(float)
        else:
            y_tasks = np.array([row.get("y", 0.0)])

        data = converter(row["smiles"], y_tasks)
        if data is not None:
            dataset.append(data)
        else:
            failed += 1

    if failed > 0:
        print(f"  Предупреждение: {failed}/{len(df)} молекул не прошли фичеризацию")
    return dataset
