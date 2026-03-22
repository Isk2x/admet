"""
Графовая нейронная сеть PharmaKinetics (MPNN).

Архитектура: NNConv слои с residual-соединениями + global mean pooling.
Используется для предсказания токсичности на молекулярных графах
и обеспечения атомно-ориентированной объяснимости.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_metrics


class MPNN(nn.Module):
    """
    Message Passing Neural Network.

    NNConv + BatchNorm + Residual, затем global mean pooling и классификатор.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr="mean"))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_embed(x)
        for conv, bn in zip(self.convs, self.bn_layers):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # residual

        self._node_embeddings = x

        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)

    def get_node_embeddings(self):
        """Доступ к эмбеддингам узлов (для модуля объяснимости)."""
        return self._node_embeddings


def train_epoch(model, loader, optimizer, device):
    """Один эпох обучения."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Оценка модели на загрузчике данных."""
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.sigmoid(out).cpu().numpy()
        y_score.append(probs)
        y_true.append(batch.y.squeeze().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    return compute_metrics(y_true, y_score), y_true, y_score


def train_gnn(
    train_data,
    val_data,
    test_data,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 128,
    n_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    device: str = None,
    save_dir: str = None,
    model_name: str = "gnn",
):
    """
    Обучение GNN с ранней остановкой по val PR-AUC.

    Возвращает: (метрики, модель, предсказания на тесте)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Обучение {model_name}: MPNN (hidden={hidden_dim}, layers={n_layers})")
    print(f"Устройство: {device}, Эпох: {epochs}, Patience: {patience}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = MPNN(node_dim, edge_dim, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_pr_auc = -1
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["pr_auc"])

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Эпоха {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                  f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics, _, _ = evaluate(model, val_loader, device)
    test_metrics, y_true, y_score = evaluate(model, test_loader, device)

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  ROC-AUC: {val_metrics['roc_auc']:.4f}  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}  PR-AUC: {test_metrics['pr_auc']:.4f}")

    results = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        np.save(save_path / f"{model_name}_test_scores.npy", y_score)
        print(f"Сохранено в {save_path}")

    return results, model, y_score
