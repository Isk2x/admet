"""
Бейзлайн-модели PharmaKinetics: ECFP4 + LogisticRegression / XGBoost.

Классические QSAR-подходы для сравнения с графовыми моделями.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.featurizer import batch_ecfp4
from src.utils.metrics import compute_metrics


def train_and_evaluate(
    train_df,
    val_df,
    test_df,
    model_type: str = "lr",
    save_dir: str = None,
) -> dict:
    """
    Обучение и оценка бейзлайн-модели.

    Параметры:
        model_type: 'lr' (LogisticRegression) или 'xgb' (XGBoost)
        save_dir: директория для сохранения модели и метрик

    Возвращает: (результаты, модель, предсказания на тесте)
    """
    print(f"\n{'='*60}")
    print(f"Обучение бейзлайна: ECFP4 + {model_type.upper()}")
    print(f"{'='*60}")

    X_train = batch_ecfp4(train_df["smiles"])
    y_train = train_df["y"].values
    X_val = batch_ecfp4(val_df["smiles"])
    y_val = val_df["y"].values
    X_test = batch_ecfp4(test_df["smiles"])
    y_test = test_df["y"].values

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Доля позитивных (train): {y_train.mean():.3f}")

    if model_type == "lr":
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "xgb":
        scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_scores)
    test_metrics = compute_metrics(y_test, test_scores)

    print(f"\nVal  ROC-AUC: {val_metrics['roc_auc']:.4f}  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}  PR-AUC: {test_metrics['pr_auc']:.4f}")

    results = {
        "model_type": model_type,
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / f"baseline_{model_type}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(save_path / f"baseline_{model_type}_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        np.save(save_path / f"baseline_{model_type}_test_scores.npy", test_scores)
        print(f"Сохранено в {save_path}")

    return results, model, test_scores
