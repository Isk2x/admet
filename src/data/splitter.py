"""
Модуль разбиения данных PharmaKinetics.

Поддерживает два типа разбиения:
  - Random split: случайное 70/15/15 с фиксированным seed
  - Scaffold split: по каркасам Bemis-Murcko (70/15/15),
    исключает утечку структурно близких молекул между split'ами
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def random_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Случайное разбиение с фиксированным seed."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))

    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def get_scaffold(smiles: str) -> str:
    """Получение каркаса Murcko для SMILES-строки."""
    try:
        return MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разбиение по каркасам Bemis-Murcko (balanced allocation).

    Группирует молекулы по scaffold, сортирует группы по убыванию
    размера, затем каждую группу добавляет в тот split, который
    наиболее далёк от целевого размера — это предотвращает
    перекос распределения в пользу одного split.
    """
    scaffolds = defaultdict(list)
    for i, smi in enumerate(df["smiles"]):
        scaffolds[get_scaffold(smi)].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    targets = [n_train, n_val, n_test]
    buckets = [[], [], []]

    for group in scaffold_sets:
        gaps = [(targets[i] - len(buckets[i]), i) for i in range(3)]
        best = max(gaps, key=lambda x: x[0])[1]
        buckets[best].extend(group)

    train_idx, val_idx, test_idx = buckets

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


if __name__ == "__main__":
    from src.data.loader import load_dataset

    df = load_dataset("tox21")

    for name, split_fn in [("random", random_split), ("scaffold", scaffold_split)]:
        train, val, test = split_fn(df)
        print(f"\n{name} split:")
        print(f"  train: {len(train)} (pos={train['y'].mean():.3f})")
        print(f"  val:   {len(val)} (pos={val['y'].mean():.3f})")
        print(f"  test:  {len(test)} (pos={test['y'].mean():.3f})")
