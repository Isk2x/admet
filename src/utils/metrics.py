"""Метрики качества моделей PharmaKinetics."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Вычисление ROC-AUC и PR-AUC с обработкой граничных случаев."""
    result = {}
    if len(np.unique(y_true)) < 2:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
        return result
    result["roc_auc"] = roc_auc_score(y_true, y_score)
    result["pr_auc"] = average_precision_score(y_true, y_score)
    return result


def compute_multitask_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_names: list = None,
) -> dict:
    """
    Per-task ROC-AUC и PR-AUC для multi-task задач (стандарт MoleculeNet).

    NaN-метки маскируются. Итоговая метрика — среднее по задачам с
    достаточным количеством валидных аннотаций.
    """
    num_tasks = y_true.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]

    per_task_roc = {}
    per_task_pr = {}

    for i, name in enumerate(task_names):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 10:
            continue
        yt = y_true[mask, i]
        ys = y_score[mask, i]
        if len(np.unique(yt)) < 2:
            continue
        per_task_roc[name] = float(roc_auc_score(yt, ys))
        per_task_pr[name] = float(average_precision_score(yt, ys))

    valid_rocs = list(per_task_roc.values())
    valid_prs = list(per_task_pr.values())

    return {
        "mean_roc_auc": float(np.mean(valid_rocs)) if valid_rocs else float("nan"),
        "mean_pr_auc": float(np.mean(valid_prs)) if valid_prs else float("nan"),
        "per_task_roc_auc": per_task_roc,
        "per_task_pr_auc": per_task_pr,
        "num_evaluated_tasks": len(valid_rocs),
    }
