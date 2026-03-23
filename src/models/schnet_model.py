"""
SchNet для multi-task токсичности PharmaKinetics.

SE(3)-инвариантная 3D GNN, работающая с атомными координатами.
В отличие от pseudo-3D MPNN, SchNet:
  - Строит radius graph (cutoff) — видит ВСЕ пары атомов в радиусе
  - Использует continuous-filter convolutions (RBF + MLP)
  - Не зависит от поворота/сдвига молекулы

Поддерживает:
  1. Single conformer — стандартный SchNet
  2. Multi-conformer ensemble — K конформаций, attention aggregation
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from src.utils.metrics import compute_multitask_metrics


class SchNetMultiTask(nn.Module):
    """
    SchNet backbone + multi-task classification head.

    Использует interaction layers из PyG SchNet для per-atom представлений,
    затем global_mean_pool → MLP classifier. Не используем встроенные
    lin1/lin2/readout SchNet (они для скалярной регрессии энергии).
    """

    def __init__(self, num_tasks: int = 12,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_channels = hidden_channels

        self.schnet_backbone = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            atomref=None,
            mean=None,
            std=None,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_tasks),
        )

        self._graph_repr = None

    def _extract_node_repr(self, z, pos, batch):
        """
        Извлечение per-atom представлений из SchNet (до lin1/lin2/readout).
        Возвращает: (node_repr, graph_repr)
        """
        sn = self.schnet_backbone
        h = sn.embedding(z)

        edge_index, edge_weight = sn.interaction_graph(pos, batch)
        edge_attr = sn.distance_expansion(edge_weight)

        for interaction in sn.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        graph_repr = global_mean_pool(h, batch)
        return h, graph_repr

    def forward(self, data):
        z = data.z
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
            torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        _, graph_repr = self._extract_node_repr(z, pos, batch)
        self._graph_repr = graph_repr
        return self.classifier(graph_repr)

    def get_graph_repr(self):
        return self._graph_repr


class MultiConfSchNet(nn.Module):
    """
    Multi-Conformer SchNet: K конформаций → shared SchNet → attention aggregation.

    Для каждой молекулы генерируется K конформеров. SchNet (shared weights)
    обрабатывает каждый, затем attention-механизм взвешивает конформеры
    для финального представления.
    """

    def __init__(self, num_tasks: int = 12,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_channels = hidden_channels

        self.schnet_backbone = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            atomref=None,
            mean=None,
            std=None,
        )

        self.conf_attention = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_tasks),
        )

    def _extract_graph_repr(self, data):
        """Извлечение graph-level представления из одного конформера."""
        sn = self.schnet_backbone
        z = data.z
        pos = data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
            torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        h = sn.embedding(z)
        edge_index, edge_weight = sn.interaction_graph(pos, batch)
        edge_attr = sn.distance_expansion(edge_weight)

        for interaction in sn.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return global_mean_pool(h, batch)

    def forward(self, conf_list):
        """
        conf_list: список Data (K конформеров одной молекулы).
        Возвращает logits (num_tasks,).
        """
        reprs = []
        for data in conf_list:
            h = self._extract_graph_repr(data)
            reprs.append(h)

        reprs = torch.stack(reprs, dim=0)  # (K, 1, hidden) → squeeze
        reprs = reprs.squeeze(1)  # (K, hidden)
        attn_scores = self.conf_attention(reprs)  # (K, 1)
        attn_weights = F.softmax(attn_scores, dim=0)  # (K, 1)
        fused = (attn_weights * reprs).sum(dim=0)  # (hidden,)

        return self.classifier(fused)


# ── Loss ──────────────────────────────────────────────────────────────

def masked_bce_loss(pred, target):
    """BCE loss с маскированием NaN (стандарт multi-task MoleculeNet)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(pred[mask], target[mask])


# ── Обучение SchNet (single conformer) ───────────────────────────────

def train_epoch_schnet(model, loader, optimizer, device, loss_fn=None):
    """Один эпох обучения SchNet."""
    if loss_fn is None:
        loss_fn = masked_bce_loss
    model.train()
    total_loss = 0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs


@torch.no_grad()
def evaluate_schnet(model, loader, device, task_names=None):
    """Оценка SchNet (per-task ROC-AUC)."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_multitask_metrics(all_targets, all_preds, task_names)


def train_schnet(
    train_data, val_data, test_data,
    num_tasks: int = 12,
    task_names: list = None,
    hidden_channels: int = 128,
    num_interactions: int = 6,
    cutoff: float = 10.0,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    device: str = None,
    save_dir: str = None,
    model_name: str = "schnet",
):
    """
    Обучение SchNet multi-task модели (single conformer).

    Параметры аналогичны train_gin — для единообразного API.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Обучение {model_name}: SchNet (hidden={hidden_channels}, "
          f"layers={num_interactions}, cutoff={cutoff})")
    print(f"Задач: {num_tasks}, Устройство: {device}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = SchNetMultiTask(
        num_tasks=num_tasks,
        hidden_channels=hidden_channels,
        num_interactions=num_interactions,
        cutoff=cutoff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_metric = -1
    best_epoch = 0
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch_schnet(model, train_loader, optimizer, device)

        val_metrics = evaluate_schnet(model, val_loader, device, task_names)
        val_auc = val_metrics["mean_roc_auc"]
        scheduler.step(val_auc)

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Эпоха {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val mean-ROC-AUC: {val_auc:.4f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics = evaluate_schnet(model, val_loader, device, task_names)
    test_metrics = evaluate_schnet(model, test_loader, device, task_names)
    elapsed = time.time() - start

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  mean-ROC-AUC: {val_metrics['mean_roc_auc']:.4f}")
    print(f"Test mean-ROC-AUC: {test_metrics['mean_roc_auc']:.4f}")
    print(f"Время: {elapsed:.0f} сек")

    results = {
        "model_name": model_name,
        "model_type": "SchNet",
        "hidden_channels": hidden_channels,
        "num_interactions": num_interactions,
        "cutoff": cutoff,
        "num_params": n_params,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "elapsed_sec": round(elapsed, 1),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Сохранено в {save_path}")

    return results, model


# ── Обучение Multi-Conformer SchNet ──────────────────────────────────

def train_multiconf_schnet(
    train_data, val_data, test_data,
    num_tasks: int = 12,
    task_names: list = None,
    hidden_channels: int = 128,
    num_interactions: int = 6,
    cutoff: float = 10.0,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    device: str = None,
    save_dir: str = None,
    model_name: str = "schnet_multiconf",
):
    """
    Обучение MultiConfSchNet.

    train_data/val_data/test_data: list[list[Data]] — для каждой молекулы
    список конформеров (Data с z, pos, y).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Обучение {model_name}: MultiConf-SchNet (hidden={hidden_channels})")
    print(f"Задач: {num_tasks}, Конформеров/мол: ~{len(train_data[0]) if train_data else 0}")
    print(f"{'='*60}")

    model = MultiConfSchNet(
        num_tasks=num_tasks,
        hidden_channels=hidden_channels,
        num_interactions=num_interactions,
        cutoff=cutoff,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    best_val_metric = -1
    best_epoch = 0
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        np.random.shuffle(train_data)

        for mol_confs in train_data:
            confs_on_device = [c.to(device) for c in mol_confs]
            optimizer.zero_grad()
            out = model(confs_on_device)
            target = confs_on_device[0].y.squeeze(0)
            loss = masked_bce_loss(out.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)

        val_preds, val_targets = _eval_multiconf(model, val_data, device)
        val_metrics = compute_multitask_metrics(val_targets, val_preds, task_names)
        val_auc = val_metrics["mean_roc_auc"]
        scheduler.step(val_auc)

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Эпоха {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val mean-ROC-AUC: {val_auc:.4f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_preds, val_targets = _eval_multiconf(model, val_data, device)
    val_metrics = compute_multitask_metrics(val_targets, val_preds, task_names)

    test_preds, test_targets = _eval_multiconf(model, test_data, device)
    test_metrics = compute_multitask_metrics(test_targets, test_preds, task_names)
    elapsed = time.time() - start

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  mean-ROC-AUC: {val_metrics['mean_roc_auc']:.4f}")
    print(f"Test mean-ROC-AUC: {test_metrics['mean_roc_auc']:.4f}")
    print(f"Время: {elapsed:.0f} сек")

    results = {
        "model_name": model_name,
        "model_type": "MultiConf-SchNet",
        "hidden_channels": hidden_channels,
        "num_interactions": num_interactions,
        "cutoff": cutoff,
        "num_params": n_params,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "elapsed_sec": round(elapsed, 1),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

    return results, model


@torch.no_grad()
def _eval_multiconf(model, data, device):
    """Evaluation helper для multi-conformer датасета."""
    model.eval()
    all_preds = []
    all_targets = []

    for mol_confs in data:
        confs_on_device = [c.to(device) for c in mol_confs]
        out = model(confs_on_device)
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(confs_on_device[0].y.squeeze(0).cpu().numpy())

    return np.array(all_preds), np.array(all_targets)
