"""
Эксперимент E7: 3D молекулярные модели (multi-seed).

Сравнение 2D (GIN) и 3D (SchNet, Hybrid) подходов:
  1. SchNet standalone (single conformer)
  2. MultiConf-SchNet (5 конформеров, attention aggregation)
  3. Hybrid GIN(2D) + SchNet(3D) (attention gate fusion)
  4. Hybrid + Uncertainty-Weighted MTL

Baseline: лучшая GIN модель из E4 (pretrained + VN + XAI).

Отчёт: mean ± std ROC-AUC по 3 seeds, scaffold split.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.data.loader_multitask import load_tox21_multitask, TOX21_TASKS
from src.data.splitter import scaffold_split
from src.data.featurizer_gin import smiles_to_graph_gin, build_gin_dataset
from src.data.featurizer_3d import smiles_to_3d, build_3d_dataset, build_3d_multi_dataset
from src.models.gin_pretrained import train_gin, download_pretrained
from src.models.schnet_model import train_schnet, train_multiconf_schnet
from src.models.hybrid_fusion import build_paired_dataset, train_hybrid

ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"

SEEDS = [42, 0, 1]


def run_experiment():
    """Запуск полного 3D-эксперимента."""
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ E7: 3D модели vs 2D GIN (3 seeds)")
    print("=" * 70)

    download_pretrained("supervised_contextpred")
    df = load_tox21_multitask()

    all_results = {}

    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED = {seed}")
        print(f"{'#'*70}")

        set_seed(seed)
        train_df, val_df, test_df = scaffold_split(df, seed=seed)
        print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        seed_results = {}

        # ── 1. GIN baseline (pretrained + VN) ────────────────────────
        print("\n--- GIN baseline (pretrained + VN) ---")
        set_seed(seed)
        train_gin_data = build_gin_dataset(train_df, TOX21_TASKS, ogb_features=False)
        val_gin_data = build_gin_dataset(val_df, TOX21_TASKS, ogb_features=False)
        test_gin_data = build_gin_dataset(test_df, TOX21_TASKS, ogb_features=False)

        gin_res, _ = train_gin(
            train_gin_data, val_gin_data, test_gin_data,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            pretrained="supervised_contextpred",
            lr=1e-3, epochs=100, patience=15, batch_size=64,
            model_name=f"gin_baseline_3d_s{seed}",
            backbone_type="vn",
        )
        seed_results["gin_pretrained_vn"] = gin_res

        # ── 2. SchNet standalone ──────────────────────────────────────
        print("\n--- SchNet standalone ---")
        set_seed(seed)
        train_3d = build_3d_dataset(train_df, TOX21_TASKS, seed=seed)
        val_3d = build_3d_dataset(val_df, TOX21_TASKS, seed=seed)
        test_3d = build_3d_dataset(test_df, TOX21_TASKS, seed=seed)

        print(f"  3D data: train={len(train_3d)}, val={len(val_3d)}, test={len(test_3d)}")

        schnet_res, _ = train_schnet(
            train_3d, val_3d, test_3d,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            hidden_channels=128,
            num_interactions=6,
            cutoff=10.0,
            lr=1e-3, epochs=100, patience=15, batch_size=64,
            model_name=f"schnet_s{seed}",
        )
        seed_results["schnet"] = schnet_res

        # ── 3. Multi-Conformer SchNet ─────────────────────────────────
        print("\n--- MultiConf-SchNet (5 conformers) ---")
        set_seed(seed)
        train_mc = build_3d_multi_dataset(train_df, TOX21_TASKS, num_confs=5, seed=seed)
        val_mc = build_3d_multi_dataset(val_df, TOX21_TASKS, num_confs=5, seed=seed)
        test_mc = build_3d_multi_dataset(test_df, TOX21_TASKS, num_confs=5, seed=seed)

        print(f"  MultiConf data: train={len(train_mc)}, val={len(val_mc)}, test={len(test_mc)}")

        mc_res, _ = train_multiconf_schnet(
            train_mc, val_mc, test_mc,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            hidden_channels=128,
            num_interactions=6,
            cutoff=10.0,
            lr=1e-3, epochs=60, patience=10,
            model_name=f"schnet_multiconf_s{seed}",
        )
        seed_results["schnet_multiconf"] = mc_res

        # ── 4. Hybrid GIN + SchNet ────────────────────────────────────
        print("\n--- Hybrid GIN(2D) + SchNet(3D) ---")
        set_seed(seed)

        train_2d, train_3d_h = build_paired_dataset(
            train_df, TOX21_TASKS, smiles_to_graph_gin,
            lambda s, y: smiles_to_3d(s, y, seed=seed),
        )
        val_2d, val_3d_h = build_paired_dataset(
            val_df, TOX21_TASKS, smiles_to_graph_gin,
            lambda s, y: smiles_to_3d(s, y, seed=seed),
        )
        test_2d, test_3d_h = build_paired_dataset(
            test_df, TOX21_TASKS, smiles_to_graph_gin,
            lambda s, y: smiles_to_3d(s, y, seed=seed),
        )

        print(f"  Paired data: train={len(train_2d)}, val={len(val_2d)}, test={len(test_2d)}")

        hybrid_res, _ = train_hybrid(
            train_2d, train_3d_h,
            val_2d, val_3d_h,
            test_2d, test_3d_h,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            pretrained="supervised_contextpred",
            lr=1e-3, backbone_lr=1e-4,
            epochs=100, patience=15, batch_size=64,
            model_name=f"hybrid_gin_schnet_s{seed}",
        )
        seed_results["hybrid_gin_schnet"] = hybrid_res

        # ── 5. Hybrid + UncMTL ────────────────────────────────────────
        print("\n--- Hybrid + Uncertainty-Weighted MTL ---")
        set_seed(seed)

        hybrid_unc_res, _ = train_hybrid(
            train_2d, train_3d_h,
            val_2d, val_3d_h,
            test_2d, test_3d_h,
            num_tasks=len(TOX21_TASKS),
            task_names=TOX21_TASKS,
            pretrained="supervised_contextpred",
            lr=1e-3, backbone_lr=1e-4,
            epochs=100, patience=15, batch_size=64,
            model_name=f"hybrid_gin_schnet_unc_s{seed}",
            use_uncertainty_loss=True,
        )
        seed_results["hybrid_gin_schnet_unc"] = hybrid_unc_res

        all_results[seed] = seed_results

    # ── Агрегация ─────────────────────────────────────────────────────
    model_names = [
        "gin_pretrained_vn", "schnet", "schnet_multiconf",
        "hybrid_gin_schnet", "hybrid_gin_schnet_unc",
    ]
    aggregated = {}

    for mname in model_names:
        test_aucs = []
        val_aucs = []
        test_praucs = []
        for seed in SEEDS:
            if mname in all_results[seed]:
                res = all_results[seed][mname]
                test_aucs.append(res["test"]["mean_roc_auc"])
                val_aucs.append(res["val"]["mean_roc_auc"])
                test_praucs.append(res["test"]["mean_pr_auc"])

        aggregated[mname] = {
            "test_roc_auc": f"{np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}",
            "val_roc_auc": f"{np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}",
            "test_pr_auc": f"{np.mean(test_praucs):.4f} ± {np.std(test_praucs):.4f}",
            "_test_roc_auc_mean": float(np.mean(test_aucs)),
            "per_seed": {
                str(s): all_results[s].get(mname, {})
                for s in SEEDS
            },
        }

    print(f"\n{'='*86}")
    print("СВОДКА: 3D МОДЕЛИ vs 2D GIN (Tox21, scaffold split, 3 seeds)")
    print(f"{'='*86}")
    print(f"{'Модель':<30} {'Test ROC-AUC':>18} {'Test PR-AUC':>18} {'Val ROC-AUC':>18}")
    print("-" * 86)
    for mname in model_names:
        a = aggregated[mname]
        print(f"{mname:<30} {a['test_roc_auc']:>18} {a['test_pr_auc']:>18} {a['val_roc_auc']:>18}")
    print("-" * 86)
    print(f"{'Hu et al. 2020 SOTA':<30} {'0.7512 ± 0.0079':>18}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = METRICS_DIR / "e7_3d_comparison.json"
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
             for k, v in aggregated.items()}
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2, default=float)
    print(f"\nСохранено: {save_path}")

    return aggregated


if __name__ == "__main__":
    run_experiment()
