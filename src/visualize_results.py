"""
Визуализация результатов экспериментов PharmaKinetics.

Генерирует графики для отчёта и защиты:
  1. Сравнение моделей (bar chart + error bars + SOTA line)
  2. Декомпозиция эффекта (waterfall chart)
  3. Radar-chart по задачам Tox21 (best model vs baseline)
  4. Пример атомной важности на молекуле
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

SOTA_ROC_AUC = 0.7512
SOTA_STD = 0.0079


def load_results():
    """Загрузка результатов из JSON (Colab или локальных)."""
    candidates = [
        ROOT / "artifacts" / "metrics" / "e4_colab_results.json",
        Path.home() / "Downloads" / "e4_colab_results.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError("e4_colab_results.json не найден")


def load_extra():
    candidates = [
        ROOT / "artifacts" / "metrics" / "e5_extra_datasets.json",
        Path.home() / "Downloads" / "e5_extra_datasets.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def parse_mean_std(s: str):
    """'0.7541 ± 0.0073' → (0.7541, 0.0073)"""
    parts = s.replace("±", "").split()
    return float(parts[0]), float(parts[1])


# ─── График 1: Сравнение моделей ─────────────────────────────────────

def plot_model_comparison(data):
    agg = data["tox21_aggregated"]

    labels_map = {
        "gin_scratch": "GIN\n(baseline)",
        "gin_vn": "GIN\n+ VN",
        "gin_ogb_vn": "GIN + OGB\n+ VN",
        "gin_vn_attn": "GIN + VN\n+ Attention",
        "gin_vn_focal": "GIN + VN\n+ Focal",
        "gin_pretrained_vn_xai": "GIN pretrained\n+ VN + XAI",
    }

    names = list(labels_map.keys())
    labels = list(labels_map.values())
    means = []
    stds = []
    for n in names:
        m, s = parse_mean_std(agg[n]["test_roc_auc"])
        means.append(m)
        stds.append(s)

    colors = ["#8faadc"] * len(names)
    colors[-1] = "#2e75b6"
    best_idx = np.argmax(means)
    colors[best_idx] = "#2e75b6"

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=1.2, zorder=3)

    ax.axhline(y=SOTA_ROC_AUC, color="#c00000", linestyle="--", linewidth=1.5,
               label=f"SOTA (Hu et al. 2020): {SOTA_ROC_AUC:.4f}", zorder=2)
    ax.axhspan(SOTA_ROC_AUC - SOTA_STD, SOTA_ROC_AUC + SOTA_STD,
               alpha=0.1, color="#c00000", zorder=1)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.003, f"{m:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test mean ROC-AUC (scaffold split)")
    ax.set_title("Сравнение GIN-моделей на Tox21 (3 seeds, scaffold split)")
    ax.set_ylim(0.72, 0.82)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_model_comparison.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '01_model_comparison.png'}")
    plt.close(fig)


# ─── График 2: Декомпозиция эффекта (waterfall) ──────────────────────

def plot_waterfall(data):
    agg = data["tox21_aggregated"]

    steps = [
        ("SOTA\n(Hu et al.)", SOTA_ROC_AUC),
        ("GIN scratch", parse_mean_std(agg["gin_scratch"]["test_roc_auc"])[0]),
        ("+ Virtual\nNode", parse_mean_std(agg["gin_vn"]["test_roc_auc"])[0]),
        ("+ Pretrained\n+ XAI loss", parse_mean_std(agg["gin_pretrained_vn_xai"]["test_roc_auc"])[0]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    base = steps[0][1]
    x_pos = [0]
    heights = [base]
    bottoms = [0]
    colors_w = ["#d9d9d9"]

    for i in range(1, len(steps)):
        x_pos.append(i)
        delta = steps[i][1] - steps[i - 1][1]
        heights.append(abs(delta))
        bottoms.append(min(steps[i][1], steps[i - 1][1]))
        colors_w.append("#70ad47" if delta > 0 else "#c00000")

    colors_w[0] = "#8faadc"

    for i in range(len(steps)):
        if i == 0:
            ax.bar(x_pos[i], steps[i][1], bottom=0, color=colors_w[i],
                   edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
            ax.text(x_pos[i], steps[i][1] + 0.002, f"{steps[i][1]:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            ax.bar(x_pos[i], heights[i], bottom=bottoms[i], color=colors_w[i],
                   edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
            delta = steps[i][1] - steps[i - 1][1]
            sign = "+" if delta > 0 else ""
            ax.text(x_pos[i], steps[i][1] + 0.002,
                    f"{steps[i][1]:.4f}\n({sign}{delta:.4f})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#2d5016" if delta > 0 else "#8b0000")

            ax.plot([x_pos[i - 1] + 0.3, x_pos[i] - 0.3],
                    [steps[i - 1][1], steps[i - 1][1]],
                    color="gray", linestyle=":", linewidth=1, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylabel("Test mean ROC-AUC")
    ax.set_title("Декомпозиция улучшений: от SOTA к лучшей модели")
    ax.set_ylim(0.73, 0.82)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_waterfall.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '02_waterfall.png'}")
    plt.close(fig)


# ─── График 3: Сравнение с SOTA (horizontal) ─────────────────────────

def plot_sota_comparison(data):
    agg = data["tox21_aggregated"]

    entries = [
        ("ECFP4 + XGBoost\n(наш baseline)", 0.6452, 0, "#bfbfbf"),
        ("GIN no pretrain\n(Hu et al. 2020)", 0.7512, 0.0079, "#d9d9d9"),
        ("GIN contextpred\n(Hu et al. 2020)", 0.7558, 0.0118, "#d9d9d9"),
        ("GIN scratch\n(наш)", parse_mean_std(agg["gin_scratch"]["test_roc_auc"])[0],
         parse_mean_std(agg["gin_scratch"]["test_roc_auc"])[1], "#8faadc"),
        ("GIN + VN\n(наш)", parse_mean_std(agg["gin_vn"]["test_roc_auc"])[0],
         parse_mean_std(agg["gin_vn"]["test_roc_auc"])[1], "#5b9bd5"),
        ("GIN pretrained\n+ VN + XAI (наш)", parse_mean_std(agg["gin_pretrained_vn_xai"]["test_roc_auc"])[0],
         parse_mean_std(agg["gin_pretrained_vn_xai"]["test_roc_auc"])[1], "#2e75b6"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = list(range(len(entries)))
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    errs = [e[2] for e in entries]
    cols = [e[3] for e in entries]

    bars = ax.barh(y_pos, vals, xerr=errs, capsize=4,
                   color=cols, edgecolor="white", linewidth=1.2, height=0.6, zorder=3)

    for i, (v, e) in enumerate(zip(vals, errs)):
        label = f"{v:.4f}" + (f" ± {e:.4f}" if e > 0 else "")
        ax.text(v + e + 0.005, i, label, va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test mean ROC-AUC (Tox21, scaffold split)")
    ax.set_title("Сравнение с опубликованными результатами")
    ax.set_xlim(0.6, 0.85)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sota_comparison.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '03_sota_comparison.png'}")
    plt.close(fig)


# ─── График 4: Гипотезы — сработало / не сработало ───────────────────

def plot_hypothesis_summary(data):
    agg = data["tox21_aggregated"]

    items = [
        ("H1: Virtual Node\n(+1 п.п. порог)", +3.48, True),
        ("H2: Pretrained+XAI\n(+2 п.п. vs SOTA)", +4.24, True),
        ("H3: OGB features\n(<1 п.п., контроль.)", -0.20, True),
        ("Attention pooling", -0.82, False),
        ("Focal loss", -1.35, False),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    names = [it[0] for it in items]
    deltas = [it[1] for it in items]
    confirmed = [it[2] for it in items]

    colors_h = []
    for d, c in zip(deltas, confirmed):
        if c and d > 0:
            colors_h.append("#70ad47")
        elif c and d <= 0:
            colors_h.append("#5b9bd5")
        else:
            colors_h.append("#c00000")

    bars = ax.barh(range(len(items)), deltas, color=colors_h,
                   edgecolor="white", linewidth=1.5, height=0.55, zorder=3)

    ax.axvline(x=0, color="black", linewidth=0.8, zorder=2)

    for i, (d, c) in enumerate(zip(deltas, confirmed)):
        sign = "+" if d > 0 else ""
        status = "подтв." if c else "не подтв."
        ax.text(d + (0.15 if d >= 0 else -0.15), i,
                f"{sign}{d:.2f} п.п. ({status})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Δ ROC-AUC (п.п.) vs baseline")
    ax.set_title("Проверка гипотез: что сработало, а что нет")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(facecolor="#70ad47", label="Гипотеза подтверждена (прирост)"),
        mpatches.Patch(facecolor="#5b9bd5", label="Гипотеза подтверждена (контрольная)"),
        mpatches.Patch(facecolor="#c00000", label="Не сработало"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_hypothesis_summary.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '04_hypothesis_summary.png'}")
    plt.close(fig)


# ─── График 5: Дополнительные датасеты ───────────────────────────────

def plot_extra_datasets(extra_data):
    if extra_data is None:
        print("e5_extra_datasets.json не найден, пропуск")
        return

    datasets = {
        "Tox21\n(12 задач)": (0.7889, 0.0080),
        "BBBP\n(1 задача)": (extra_data["bbbp"]["test_roc_auc_mean"],
                             extra_data["bbbp"]["test_roc_auc_std"]),
        "ClinTox\n(2 задачи)": (extra_data["clintox"]["test_roc_auc_mean"],
                                extra_data["clintox"]["test_roc_auc_std"]),
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    names = list(datasets.keys())
    means = [v[0] for v in datasets.values()]
    stds = [v[1] for v in datasets.values()]
    colors_d = ["#2e75b6", "#70ad47", "#bfbfbf"]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=colors_d, edgecolor="white", linewidth=1.5, zorder=3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.015, f"{m:.4f}±{s:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(names) - 0.5, 0.505, "random baseline", fontsize=8,
            color="gray", ha="right")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Test mean ROC-AUC")
    ax.set_title("Обобщаемость: GIN+VN на разных датасетах (3 seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_extra_datasets.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '05_extra_datasets.png'}")
    plt.close(fig)


# ─── График 6: Пример атомной важности на молекуле ───────────────────

def plot_molecule_importance():
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import torch
    except ImportError:
        print("RDKit/torch не доступны, пропуск визуализации молекулы")
        return

    try:
        from src.models.gin_pretrained import GINMultiTask
        from src.data.featurizer_gin import smiles_to_graph_gin
        from src.explain.atom_importance_gin import compute_atom_importance_gin
    except ImportError:
        print("Модули PharmaKinetics не доступны, пропуск")
        return

    smiles = "c1ccc2c(c1)cc(cc2)N(=O)=O"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    model = GINMultiTask(num_tasks=12, backbone_type="vn")
    model.eval()

    data = smiles_to_graph_gin(smiles)
    importance = compute_atom_importance_gin(model, data, "cpu")

    imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("toxicity", ["#ffffff", "#ffcccc", "#ff0000"])

    atom_colors = {}
    for i, imp in enumerate(imp_norm):
        atom_colors[i] = cmap(float(imp))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    img = Draw.MolToImage(mol, size=(400, 300))
    ax1.imshow(img)
    ax1.set_title("Молекула (2-нитронафталин)", fontsize=11)
    ax1.axis("off")

    atom_labels = []
    for atom in mol.GetAtoms():
        atom_labels.append(f"{atom.GetSymbol()}{atom.GetIdx()}")

    y_pos = range(len(imp_norm))
    colors_imp = [cmap(float(v)) for v in imp_norm]
    ax2.barh(y_pos, imp_norm, color=colors_imp, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(atom_labels, fontsize=8)
    ax2.set_xlabel("Нормализованная важность")
    ax2.set_title("Атомная важность (GIN+VN)", fontsize=11)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Пример объяснения предсказания токсичности", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_molecule_importance.png", bbox_inches="tight")
    print(f"Сохранено: {FIGURES_DIR / '06_molecule_importance.png'}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PharmaKinetics: генерация графиков для отчёта")
    print("=" * 60)

    data = load_results()
    extra = load_extra()

    plot_model_comparison(data)
    plot_waterfall(data)
    plot_sota_comparison(data)
    plot_hypothesis_summary(data)
    plot_extra_datasets(extra)
    plot_molecule_importance()

    print(f"\nВсе графики сохранены в: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
