"""
Multi-task загрузчик Tox21 для PharmaKinetics.

Возвращает per-task метки (12 задач) вместо агрегированной бинарной метки.
Это стандартная постановка MoleculeNet-бенчмарков, где итоговая метрика —
среднее ROC-AUC по всем задачам.
"""

import numpy as np
import pandas as pd
from rdkit import Chem

from src.data.loader import RAW_DIR, PROCESSED_DIR, DATASETS, _download

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]


def load_tox21_multitask(force_download: bool = False) -> pd.DataFrame:
    """
    Загрузка Tox21 с per-task метками (NaN сохраняются).

    Возвращает DataFrame: smiles + 12 столбцов задач (float, NaN = нет аннотации).
    """
    config = DATASETS["tox21"]
    raw_path = RAW_DIR / config["filename"]
    processed_path = PROCESSED_DIR / "tox21_multitask.csv"

    if processed_path.exists() and not force_download:
        df = pd.read_csv(processed_path)
        n_labeled = df[TOX21_TASKS].notna().sum().sum()
        print(f"[tox21-multitask] Из кэша: {len(df)} молекул, "
              f"{n_labeled} аннотаций по 12 задачам")
        return df

    if not raw_path.exists() or force_download:
        _download(config["url"], raw_path, config["compressed"])

    raw_df = pd.read_csv(raw_path)

    all_nan_mask = raw_df[TOX21_TASKS].isna().all(axis=1)
    raw_df = raw_df[~all_nan_mask].copy()

    canonical = []
    task_rows = []
    for _, row in raw_df.iterrows():
        mol = Chem.MolFromSmiles(str(row[config["smiles_col"]]))
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        canonical.append(smi)
        task_rows.append(row[TOX21_TASKS].values.astype(float))

    result = pd.DataFrame({"smiles": canonical})
    task_df = pd.DataFrame(task_rows, columns=TOX21_TASKS)
    result = pd.concat([result, task_df], axis=1)

    result = result.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(processed_path, index=False)

    n_labeled = result[TOX21_TASKS].notna().sum().sum()
    print(f"[tox21-multitask] {len(result)} молекул, "
          f"{n_labeled} аннотаций по 12 задачам")
    for task in TOX21_TASKS:
        valid = result[task].dropna()
        pos_rate = valid.mean() if len(valid) > 0 else 0
        print(f"  {task}: {len(valid)} аннотаций, pos_rate={pos_rate:.3f}")

    return result
