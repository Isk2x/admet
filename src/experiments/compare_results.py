"""
Сравнение экспериментальных результатов PharmaKinetics
с целевыми метриками из исследовательских гипотез.

Читает метрики из artifacts/metrics/ и генерирует таблицу сравнения.
Вывод: stdout + docs/experiments.md.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Целевые метрики из исследовательских гипотез ──
TARGETS = {
    "H2: 3D-GNN ROC-AUC (scaffold)": {"target": 0.90, "condition": ">="},
    "H2: 3D-GNN PR-AUC (scaffold)": {"target": 0.60, "condition": ">="},
    "H1: AOPC top-20%": {"target": 0.10, "condition": ">=", "note": "прирост faithfulness >= 0.10"},
    "H1: IoU@20%": {"target": 0.50, "condition": ">="},
}


def load_actual_results() -> dict:
    """Загрузка результатов из файлов экспериментов."""
    actual = {}

    e1_path = METRICS_DIR / "e1_baseline_vs_graph.json"
    if e1_path.exists():
        e1 = json.loads(e1_path.read_text())
        for split in ["random", "scaffold"]:
            if split in e1:
                for model_key in ["ecfp4_lr", "ecfp4_xgb", "gnn_2d"]:
                    if model_key in e1[split]:
                        test = e1[split][model_key].get("test", {})
                        label = model_key.upper().replace("_", "+")
                        actual[f"{label} ROC-AUC ({split})"] = test.get("roc_auc")
                        actual[f"{label} PR-AUC ({split})"] = test.get("pr_auc")

    e2_path = METRICS_DIR / "e2_graph_vs_3d.json"
    if e2_path.exists():
        e2 = json.loads(e2_path.read_text())
        for split in ["random", "scaffold"]:
            if split in e2:
                for model_key in ["gnn_2d", "gnn_3d"]:
                    if model_key in e2[split]:
                        test = e2[split][model_key].get("test", {})
                        label = model_key.upper().replace("_", " ")
                        actual[f"{label} ROC-AUC ({split})"] = test.get("roc_auc")
                        actual[f"{label} PR-AUC ({split})"] = test.get("pr_auc")

    xai_path = METRICS_DIR / "explainability_results.json"
    if xai_path.exists():
        xai = json.loads(xai_path.read_text())
        if "aopc_20" in xai:
            actual["AOPC top-20%"] = xai["aopc_20"].get("mean_aopc_top")
            actual["AOPC random-20%"] = xai["aopc_20"].get("mean_aopc_random")
            actual["Faithfulness gain @20%"] = xai["aopc_20"].get("mean_faithfulness_gain")
        if "iou_20" in xai:
            actual["IoU@20%"] = xai["iou_20"].get("mean_iou")

    return actual


def build_results_table(actual: dict) -> str:
    """Таблица всех экспериментальных результатов."""
    lines = []
    lines.append("| Метрика | Значение |")
    lines.append("|---------|----------|")
    for metric, val in sorted(actual.items()):
        if val is not None and not (isinstance(val, float) and val != val):
            lines.append(f"| {metric} | {val:.4f} |")
        else:
            lines.append(f"| {metric} | N/A |")
    return "\n".join(lines)


def main():
    actual = load_actual_results()

    if not actual:
        print("Результаты экспериментов не найдены в artifacts/metrics/.")
        print("Запустите эксперименты:")
        print("  python -m src.experiments.run_baseline_vs_graph")
        print("  python -m src.experiments.run_graph_vs_3d")
        print("  python -m src.experiments.run_explainability")
        return

    results_table = build_results_table(actual)

    print("\n" + "=" * 70)
    print("ВСЕ ЭКСПЕРИМЕНТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(results_table)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# PharmaKinetics: результаты экспериментов

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Датасет

- **Источник:** Tox21 (MoleculeNet), https://deepchem.io/datasets/tox21.csv.gz
- **Предобработка:** каноникализация SMILES (RDKit), дедупликация, агрегированная бинарная метка
- **Разбиения:** Random 70/15/15 (seed=42) и Scaffold (Bemis-Murcko) 70/15/15 (seed=42)

## Модели

- **ECFP4+LR:** Morgan fingerprint (radius=2, 2048 бит) + LogisticRegression (balanced class weights)
- **ECFP4+XGB:** Morgan fingerprint (radius=2, 2048 бит) + XGBClassifier (300 деревьев, scale_pos_weight)
- **GNN 2D:** MPNN с NNConv (3 слоя, hidden=128, residual), обучение Adam, ранняя остановка по val PR-AUC
- **GNN 3D:** Та же MPNN + RBF-кодированные 3D межатомные расстояния на рёбрах (RDKit ETKDG конформеры)

## Результаты

{results_table}

## Воспроизводимость

```bash
source venv/bin/activate

# E1: Бейзлайн vs Графовая модель
python -m src.experiments.run_baseline_vs_graph

# E2: Граф (2D) vs Pseudo-3D
python -m src.experiments.run_graph_vs_3d

# Объяснимость (требует предварительного запуска E1)
python -m src.experiments.run_explainability

# Сравнение результатов
python -m src.experiments.compare_results
```

## Программно невоспроизводимые элементы

- **A/B тест (override rate):** требует экспертов-химиков, отмечено как TODO
- **H3 (SMILES-аугментация):** вне текущего скоупа
"""

    with open(DOCS_DIR / "experiments.md", "w") as f:
        f.write(report)
    print(f"\nОтчёт сохранён: {DOCS_DIR / 'experiments.md'}")


if __name__ == "__main__":
    main()
