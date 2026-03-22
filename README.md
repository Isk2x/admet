# PharmaKinetics — ML-сервис предсказания токсичности молекул

ML-пайплайн для предсказания токсичности лекарственных молекул
по SMILES-представлению с модулем атомной объяснимости.

## Датасеты

Все датасеты из [MoleculeNet](https://moleculenet.org/datasets-1), скачиваются автоматически при первом запуске.

| Датасет | Молекулы | Задачи | Описание |
|---------|---------|--------|----------|
| **Tox21** | ~7500 | 12 | Ядерные рецепторы + стрессовые пути |
| **ClinTox** | ~1478 | 2 | Одобрение FDA + клиническая токсичность |
| **ToxCast** | ~8576 | 617 | Высокопроизводительный скрининг EPA |
| **SIDER** | ~1427 | 27 | Побочные эффекты лекарств |
| **BBBP** | ~2039 | 1 | Проницаемость гематоэнцефалического барьера |

## Быстрый старт

```bash
# 1. Активировать виртуальное окружение
source venv/bin/activate

# 2. Установить зависимости (если ещё не установлены)
pip install -r requirements.txt

# 3. Запустить все эксперименты (датасеты скачаются автоматически)
python run_all.py

# Или быстрый режим (~10 мин, без 3D и объяснимости):
python run_all.py --quick
```

## Запуск отдельных экспериментов

```bash
source venv/bin/activate

# Загрузить и проверить все датасеты
python -m src.data.loader

# E1: Бейзлайн (ECFP4+LR, ECFP4+XGB) vs Графовая модель (MPNN)
python -m src.experiments.run_baseline_vs_graph

# E2: Граф (2D MPNN) vs Pseudo-3D (MPNN + конформерные расстояния)
python -m src.experiments.run_graph_vs_3d

# Объяснимость: AOPC + IoU (требует E1)
python -m src.experiments.run_explainability

# E3: SOTA-сравнение (GIN scratch vs pretrained vs pretrained+XAI)
python -m src.experiments.run_sota_comparison

# Сравнение объяснимости GIN-моделей
python -m src.experiments.run_xai_comparison
```

## Структура проекта

```
├── run_all.py                 # Главный скрипт — запуск всех экспериментов
├── requirements.txt           # Зависимости Python
├── venv/                      # Виртуальное окружение
├── src/
│   ├── data/
│   │   ├── loader.py            # Загрузка датасетов (Tox21, ClinTox, ...)
│   │   ├── loader_multitask.py  # Multi-task загрузчик Tox21 (12 задач)
│   │   ├── splitter.py          # Random + Scaffold (Murcko) split
│   │   ├── featurizer.py        # ECFP4, 2D граф, 3D граф
│   │   └── featurizer_gin.py    # GIN-совместимая фичеризация
│   ├── models/
│   │   ├── baseline.py          # ECFP4 + LR / XGBoost
│   │   ├── gnn.py               # 2D MPNN (NNConv + global mean pooling)
│   │   ├── gnn_3d.py            # Pseudo-3D MPNN (+ RBF расстояния)
│   │   ├── gin_pretrained.py    # Pretrained GIN + multi-task head
│   │   └── xai_loss.py          # Explainability-Aware Loss
│   ├── explain/
│   │   ├── atom_importance.py     # Градиентная атомная атрибуция (MPNN)
│   │   ├── atom_importance_gin.py # Атрибуция для GIN (L2-норма embeddings)
│   │   ├── perturbation.py        # AOPC (зануление атомов)
│   │   └── stability.py           # IoU (устойчивость объяснений)
│   ├── experiments/
│   │   ├── run_baseline_vs_graph.py  # E1
│   │   ├── run_graph_vs_3d.py        # E2
│   │   ├── run_explainability.py     # XAI метрики (MPNN)
│   │   ├── run_sota_comparison.py    # E3: GIN scratch vs pretrained vs XAI
│   │   ├── run_xai_comparison.py     # Сравнение объяснимости GIN-моделей
│   │   └── compare_results.py        # Итоговая таблица
│   └── utils/
│       ├── metrics.py         # ROC-AUC, PR-AUC, multi-task метрики
│       └── seed.py            # Воспроизводимость
├── data/                      # Датасеты (автозагрузка)
│   ├── raw/                   # Исходные CSV
│   └── processed/             # Обработанные данные
├── artifacts/
│   ├── models/                # Веса моделей
│   ├── pretrained/            # Pretrained GIN backbone
│   ├── metrics/               # Метрики (JSON)
│   └── explanations/          # Карты объяснимости
└── docs/
    ├── experiments.md         # Отчёт по экспериментам
    └── ...                    # Документация проекта
```

## Модели

| Модель | Описание | Признаки |
|--------|----------|----------|
| ECFP4+LR | Logistic Regression | Morgan fingerprint (r=2, 2048 бит) |
| ECFP4+XGB | XGBoost | Morgan fingerprint (r=2, 2048 бит) |
| MPNN 2D | NNConv (3 слоя, hidden=128) | Атомные + связевые one-hot |
| MPNN 3D | MPNN + RBF расстояния | + 3D конформерные расстояния |
| **GIN (scratch)** | 5-слойная GIN (emb=300), multi-task | Integer atom/bond embeddings |
| **GIN (pretrained)** | Pretrained backbone (Hu et al. 2020) | 2M молекул pretraining |
| **GIN (pretrained+XAI)** | + Explainability-Aware Loss | L_faith + L_stab |

## Ключевые результаты (Tox21, scaffold split)

| Модель | Test ROC-AUC | IoU стабильность |
|--------|-------------|-----------------|
| ECFP4+XGB | 0.645 | — |
| MPNN 2D | 0.736 | 0.978 |
| **GIN (scratch)** | **0.761** | 0.892 |
| GIN (pretrained) | 0.718 | 0.907 |
| **GIN (pretrained+XAI)** | **0.760** | **0.913** |

## Метрики

- **ROC-AUC**: площадь под ROC-кривой (per-task → среднее)
- **PR-AUC**: площадь под Precision-Recall кривой
- **AOPC@k%**: падение предсказания при зануении top-k% атомов (faithfulness)
- **IoU@k%**: устойчивость top-k% объяснений при перенумерации атомов (stability)

## Научная новизна

**Explainability-Aware Fine-tuning** — регуляризация, встраивающая метрики объяснимости в обучение:

```
L_total = L_task + λ · (L_faithfulness + L_stability)
```

- XAI-loss не ухудшает предсказательную способность (0.760 vs 0.761)
- Повышает стабильность объяснений: IoU 0.913 vs 0.892 (scratch)

## Ожидаемое время выполнения

| Эксперимент | CPU | A100 GPU |
|-------------|-----|----------|
| E1 (бейзлайн + MPNN) | ~10-15 мин | ~3 мин |
| E2 (2D + 3D MPNN) | ~20-30 мин | ~5 мин |
| **E3 (GIN SOTA-сравнение)** | **~30 мин** | **~10 мин** |
| XAI метрики | ~5-10 мин | ~2 мин |
| **Итого** | **~65-85 мин** | **~20 мин** |
