"""
Загрузчик токсикологических датасетов PharmaKinetics.

Поддерживаемые датасеты (MoleculeNet):
  - Tox21:    ~7500 молекул, 12 задач (ядерные рецепторы, стрессовые пути)
  - ClinTox:  ~1478 молекул, 2 задачи (одобрение FDA, клиническая токсичность)
  - ToxCast:  ~8576 молекул, 617 задач (высокопроизводительный скрининг)
  - SIDER:    ~1427 молекул, 27 задач (побочные эффекты лекарств)
  - BBBP:     ~2039 молекул, 1 задача (проницаемость гематоэнцефалического барьера)

Все датасеты скачиваются автоматически при первом обращении.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Конфигурация датасетов ──────────────────────────────────────────────

DATASETS = {
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv",
        "compressed": True,
        "smiles_col": "smiles",
        "task_columns": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
            "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "description": "Tox21: 12 задач бинарной классификации токсичности",
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv",
        "compressed": True,
        "smiles_col": "smiles",
        "task_columns": ["FDA_APPROVED", "CT_TOX"],
        "description": "ClinTox: одобрение FDA + клиническая токсичность",
    },
    "toxcast": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
        "filename": "toxcast.csv",
        "compressed": True,
        "smiles_col": "smiles",
        "task_columns": None,  # автоопределение — все столбцы кроме smiles
        "description": "ToxCast: 617 задач высокопроизводительного скрининга",
    },
    "sider": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "filename": "sider.csv",
        "compressed": True,
        "smiles_col": "smiles",
        "task_columns": None,  # автоопределение
        "description": "SIDER: 27 категорий побочных эффектов лекарств",
    },
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "bbbp.csv",
        "compressed": False,
        "smiles_col": "smiles",
        "task_columns": ["p_np"],
        "description": "BBBP: проницаемость гематоэнцефалического барьера",
    },
}


def _download(url: str, dest: Path, compressed: bool = False):
    """Скачивание файла с автоматической распаковкой gzip."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        gz_path = dest.with_suffix(dest.suffix + ".gz")
        print(f"  Скачивание {url} ...")
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
    else:
        print(f"  Скачивание {url} ...")
        urllib.request.urlretrieve(url, dest)

    print(f"  Сохранено: {dest}")


def _canonicalize_and_clean(df: pd.DataFrame, smiles_col: str,
                            task_columns: list) -> pd.DataFrame:
    """
    Каноникализация SMILES через RDKit, удаление невалидных
    и дублирующихся молекул, формирование агрегированной метки y.
    """
    present_tasks = [c for c in task_columns if c in df.columns]
    if not present_tasks:
        raise ValueError(f"Столбцы задач не найдены. Доступные: {list(df.columns)}")

    task_data = df[present_tasks]

    # Удаляем молекулы без единой аннотации
    all_nan_mask = task_data.isna().all(axis=1)
    df = df[~all_nan_mask].copy()
    task_data = task_data.loc[df.index]

    # Агрегированная метка: токсичен хотя бы по одной задаче
    df["y"] = (task_data.fillna(0).max(axis=1) > 0).astype(int)

    # Каноникализация SMILES
    canonical = []
    valid_idx = []
    for idx, smi in df[smiles_col].items():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canonical.append(Chem.MolToSmiles(mol))
            valid_idx.append(idx)

    result = pd.DataFrame({"smiles": canonical, "y": df.loc[valid_idx, "y"].values})
    result = result.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)

    return result


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """
    Загрузка датасета по имени.

    Параметры:
        name: имя датасета ('tox21', 'clintox', 'toxcast', 'sider', 'bbbp')
        force_download: принудительно перекачать

    Возвращает:
        DataFrame с колонками: smiles, y (бинарная агрегированная метка)
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Неизвестный датасет: '{name}'. Доступные: {available}")

    config = DATASETS[name]
    raw_path = RAW_DIR / config["filename"]
    processed_path = PROCESSED_DIR / f"{name}_processed.csv"

    # Если уже обработан — загружаем из кэша
    if processed_path.exists() and not force_download:
        df = pd.read_csv(processed_path)
        print(f"[{name}] Загружено из кэша: {len(df)} молекул, "
              f"доля позитивных = {df['y'].mean():.3f}")
        return df

    # Скачиваем если нужно
    if not raw_path.exists() or force_download:
        _download(config["url"], raw_path, config["compressed"])

    # Читаем и обрабатываем
    df = pd.read_csv(raw_path)

    # Определяем столбцы задач
    task_columns = config["task_columns"]
    if task_columns is None:
        # Автоопределение: все числовые столбцы кроме smiles
        task_columns = [
            c for c in df.columns
            if c != config["smiles_col"] and df[c].dtype in ["float64", "int64"]
        ]

    result = _canonicalize_and_clean(df, config["smiles_col"], task_columns)

    # Сохраняем обработанный
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(processed_path, index=False)

    print(f"[{name}] {config['description']}")
    print(f"  Молекул: {len(result)}, доля позитивных: {result['y'].mean():.3f}")

    return result


def load_all_datasets(force_download: bool = False) -> dict:
    """Загрузка всех доступных датасетов. Возвращает dict {name: DataFrame}."""
    results = {}
    for name in DATASETS:
        try:
            results[name] = load_dataset(name, force_download)
        except Exception as e:
            print(f"[{name}] ОШИБКА: {e}")
    return results


# Обратная совместимость
def load_tox21(force_download: bool = False) -> pd.DataFrame:
    """Загрузка Tox21 (обратная совместимость)."""
    return load_dataset("tox21", force_download)


if __name__ == "__main__":
    print("=" * 60)
    print("PharmaKinetics: загрузка всех датасетов")
    print("=" * 60)

    datasets = load_all_datasets()

    print(f"\n{'='*60}")
    print("СВОДКА")
    print(f"{'='*60}")
    print(f"{'Датасет':<12} {'Молекул':>10} {'Позитивных':>12}")
    print("-" * 36)
    for name, df in datasets.items():
        print(f"{name:<12} {len(df):>10} {df['y'].mean():>12.3f}")
    total = sum(len(df) for df in datasets.values())
    print(f"{'ИТОГО':<12} {total:>10}")
