"""
Эксперимент E4: Улучшенное сравнение GIN-моделей (multi-seed).

Запускает все варианты модели с 3 seeds и scaffold split:
  1. GIN scratch (baseline)
  2. GIN + Virtual Node
  3. GIN + OGB features + Virtual Node
  4. GIN + VN + Attention pooling
  5. GIN + VN + Focal loss
  6. GIN pretrained + VN + XAI loss
  7. GIN + VN + Uncertainty-Weighted MTL (Kendall 2018)
  8. GIN + VN + Toxicophore-Guided XAI
  9. GIN pretrained + VN + UncMTL + XAI + Toxicophore (full)

Отчёт: mean ± std ROC-AUC по 3 seeds.
"""

import json
import sys
from pathlib import Path
from functools import partial

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader_multitask import load_tox21_multitask, TOX21_TASKS
from src.data.splitter import scaffold_split
from src.data.featurizer_gin import build_gin_dataset
from src.models.gin_pretrained import train_gin, download_pretrained
from src.models.xai_loss import combined_xai_loss, compute_toxicophore_loss

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

SEEDS = [42, 0, 1]

MODEL_CONFIGS = [
    {
        "name": "gin_scratch",
        "backbone_type": "standard",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": False,
        "xai_lambda": 0.0,
        "ogb_features": False,
    },
    {
        "name": "gin_vn",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": False,
        "xai_lambda": 0.0,
        "ogb_features": False,
    },
    {
        "name": "gin_ogb_vn",
        "backbone_type": "ogb_vn",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": False,
        "xai_lambda": 0.0,
        "ogb_features": True,
    },
    {
        "name": "gin_vn_attn",
        "backbone_type": "vn",
        "pool_type": "attention",
        "pretrained": None,
        "use_focal_loss": False,
        "xai_lambda": 0.0,
        "ogb_features": False,
    },
    {
        "name": "gin_vn_focal",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": True,
        "xai_lambda": 0.0,
        "ogb_features": False,
    },
    {
        "name": "gin_pretrained_vn_xai",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": "supervised_contextpred",
        "use_focal_loss": False,
        "xai_lambda": 0.1,
        "ogb_features": False,
    },
    {
        "name": "gin_vn_uncmtl",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": False,
        "use_uncertainty_loss": True,
        "xai_lambda": 0.0,
        "toxicophore_lambda": 0.0,
        "ogb_features": False,
    },
    {
        "name": "gin_vn_toxguided",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": None,
        "use_focal_loss": False,
        "use_uncertainty_loss": False,
        "xai_lambda": 0.1,
        "toxicophore_lambda": 0.05,
        "ogb_features": False,
    },
    {
        "name": "gin_pretrained_vn_full",
        "backbone_type": "vn",
        "pool_type": "mean",
        "pretrained": "supervised_contextpred",
        "use_focal_loss": False,
        "use_uncertainty_loss": True,
        "xai_lambda": 0.1,
        "toxicophore_lambda": 0.05,
        "ogb_features": False,
    },
]


def run_single_seed(seed: int, model_configs: list):
    """Запуск всех моделей для одного seed."""
    set_seed(seed)

    df = load_tox21_multitask()
    train_df, val_df, test_df = scaffold_split(df, seed=seed)
    print(f"\nSeed {seed}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_std = build_gin_dataset(train_df, TOX21_TASKS, ogb_features=False)
    val_std = build_gin_dataset(val_df, TOX21_TASKS, ogb_features=False)
    test_std = build_gin_dataset(test_df, TOX21_TASKS, ogb_features=False)

    train_ogb = build_gin_dataset(train_df, TOX21_TASKS, ogb_features=True)
    val_ogb = build_gin_dataset(val_df, TOX21_TASKS, ogb_features=True)
    test_ogb = build_gin_dataset(test_df, TOX21_TASKS, ogb_features=True)

    seed_results = {}

    for cfg in model_configs:
        set_seed(seed)

        if cfg["ogb_features"]:
            td, vd, ted = train_ogb, val_ogb, test_ogb
        else:
            td, vd, ted = train_std, val_std, test_std

        xai_fn = None
        if cfg["xai_lambda"] > 0:
            xai_fn = partial(combined_xai_loss, faith_weight=0.5, stab_weight=0.5)

        tox_fn = None
        tox_lambda = cfg.get("toxicophore_lambda", 0.0)
        if tox_lambda > 0:
            tox_fn = compute_toxicophore_loss

        if cfg["pretrained"]:
            download_pretrained(cfg["pretrained"])

        results, model = train_gin(
            td, vd, ted,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            pretrained=cfg["pretrained"],
            xai_loss_fn=xai_fn,
            xai_lambda=cfg["xai_lambda"],
            lr=1e-3,
            epochs=100,
            patience=15,
            batch_size=64,
            save_dir=str(MODELS_DIR),
            model_name=f"{cfg['name']}_s{seed}",
            backbone_type=cfg["backbone_type"],
            pool_type=cfg["pool_type"],
            use_focal_loss=cfg["use_focal_loss"],
            use_uncertainty_loss=cfg.get("use_uncertainty_loss", False),
            toxicophore_loss_fn=tox_fn,
            toxicophore_lambda=tox_lambda,
        )
        seed_results[cfg["name"]] = results

    return seed_results


def run_experiment():
    """Запуск полного multi-seed эксперимента."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E4: Улучшенное сравнение GIN (3 seeds)")
    print("=" * 70)

    if any(c["pretrained"] for c in MODEL_CONFIGS):
        download_pretrained("supervised_contextpred")

    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED = {seed}")
        print(f"{'#'*70}")
        all_seed_results[seed] = run_single_seed(seed, MODEL_CONFIGS)

    aggregated = {}
    model_names = [c["name"] for c in MODEL_CONFIGS]

    for mname in model_names:
        test_aucs = []
        val_aucs = []
        test_praucs = []
        for seed in SEEDS:
            if mname in all_seed_results[seed]:
                res = all_seed_results[seed][mname]
                test_aucs.append(res["test"]["mean_roc_auc"])
                val_aucs.append(res["val"]["mean_roc_auc"])
                test_praucs.append(res["test"]["mean_pr_auc"])

        aggregated[mname] = {
            "test_roc_auc_mean": float(np.mean(test_aucs)),
            "test_roc_auc_std": float(np.std(test_aucs)),
            "val_roc_auc_mean": float(np.mean(val_aucs)),
            "val_roc_auc_std": float(np.std(val_aucs)),
            "test_pr_auc_mean": float(np.mean(test_praucs)),
            "test_pr_auc_std": float(np.std(test_praucs)),
            "n_seeds": len(test_aucs),
            "per_seed": {
                str(s): all_seed_results[s][mname]
                for s in SEEDS if mname in all_seed_results[s]
            },
        }

    print(f"\n{'='*80}")
    print("СВОДКА РЕЗУЛЬТАТОВ (scaffold split, 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Модель':<30} {'Test ROC-AUC':>18} {'Test PR-AUC':>18} {'Val ROC-AUC':>18}")
    print("-" * 86)
    for mname in model_names:
        a = aggregated[mname]
        print(f"{mname:<30} "
              f"{a['test_roc_auc_mean']:.4f}±{a['test_roc_auc_std']:.4f}  "
              f"{a['test_pr_auc_mean']:.4f}±{a['test_pr_auc_std']:.4f}  "
              f"{a['val_roc_auc_mean']:.4f}±{a['val_roc_auc_std']:.4f}")

    print(f"\nОпубликованный SOTA (Hu et al.): 0.7512 ± 0.0079")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "e4_improved_comparison.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=float)
    print(f"\nСохранено: {METRICS_DIR / 'e4_improved_comparison.json'}")

    return aggregated


if __name__ == "__main__":
    run_experiment()
