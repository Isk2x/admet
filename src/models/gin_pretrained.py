"""
Pretrained GIN для PharmaKinetics.

Реализация GIN (Graph Isomorphism Network) совместимая с pretrained весами
из Hu et al. 2020 "Strategies for Pre-training Graph Neural Networks".

Поддерживает:
  - Загрузку pretrained backbone (supervised + contextpred)
  - Multi-task classification head (12 задач Tox21)
  - Обучение from scratch для сравнения
  - Virtual Node (OGB-style) для улучшения глобальной агрегации
  - Attention pooling вместо mean pooling
  - Focal loss для несбалансированных задач
  - Расширенные OGB-style атомные признаки (9 + 3 bond)
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader

from src.utils.metrics import compute_multitask_metrics

NUM_ATOM_TYPE = 120
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3

# OGB-style расширенные признаки
NUM_DEGREE = 11
NUM_FORMAL_CHARGE = 11
NUM_NUM_HS = 9
NUM_NUM_RADICAL_E = 6
NUM_HYBRIDIZATION = 7
NUM_IS_AROMATIC = 2
NUM_IS_IN_RING = 2

NUM_BOND_STEREO = 6
NUM_IS_CONJUGATED = 2

PRETRAINED_URLS = {
    "supervised_contextpred": "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/refs/heads/master/chem/model_gin/supervised_contextpred.pth",
    "supervised": "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/refs/heads/master/chem/model_gin/supervised.pth",
    "contextpred": "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/refs/heads/master/chem/model_gin/contextpred.pth",
}

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "pretrained"


class GINConvLayer(MessagePassing):
    """GIN conv layer совместимый с pretrained весами (Hu et al. 2020)."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype,
                                     device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = (self.edge_embedding1(edge_attr[:, 0].long())
                     + self.edge_embedding2(edge_attr[:, 1].long()))

        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINBackbone(nn.Module):
    """
    GIN backbone совместимый с pretrained весами.

    Архитектура: 5 GINConv слоев + BatchNorm + Dropout.
    """

    def __init__(self, num_layer: int = 5, emb_dim: int = 300,
                 drop_ratio: float = 0.5):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConvLayer(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        return h_list[-1]


class GINBackboneOGB(nn.Module):
    """
    GIN backbone с расширенными OGB-style признаками (9 атомных + 3 bond).

    Не совместим с pretrained весами — обучается только from scratch.
    """

    def __init__(self, num_layer: int = 5, emb_dim: int = 300,
                 drop_ratio: float = 0.5):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim

        self.atom_embs = nn.ModuleList([
            nn.Embedding(NUM_ATOM_TYPE, emb_dim),
            nn.Embedding(NUM_CHIRALITY_TAG, emb_dim),
            nn.Embedding(NUM_DEGREE, emb_dim),
            nn.Embedding(NUM_FORMAL_CHARGE, emb_dim),
            nn.Embedding(NUM_NUM_HS, emb_dim),
            nn.Embedding(NUM_NUM_RADICAL_E, emb_dim),
            nn.Embedding(NUM_HYBRIDIZATION, emb_dim),
            nn.Embedding(NUM_IS_AROMATIC, emb_dim),
            nn.Embedding(NUM_IS_IN_RING, emb_dim),
        ])
        for emb in self.atom_embs:
            nn.init.xavier_uniform_(emb.weight.data)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConvLayerOGB(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        h = sum(emb(x[:, i]) for i, emb in enumerate(self.atom_embs))

        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        return h_list[-1]


class GINConvLayerOGB(MessagePassing):
    """GIN conv layer с OGB-style bond features (3 признака)."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.bond_emb1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.bond_emb2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.bond_emb3 = nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.bond_emb4 = nn.Embedding(NUM_IS_CONJUGATED, emb_dim)
        for emb in [self.bond_emb1, self.bond_emb2, self.bond_emb3, self.bond_emb4]:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        n_feats = edge_attr.size(1)
        self_loop_attr = torch.zeros(x.size(0), n_feats, dtype=edge_attr.dtype,
                                     device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        edge_emb = (self.bond_emb1(edge_attr[:, 0].long())
                     + self.bond_emb2(edge_attr[:, 1].long()))
        if n_feats >= 3:
            edge_emb = edge_emb + self.bond_emb3(edge_attr[:, 2].long())
        if n_feats >= 4:
            edge_emb = edge_emb + self.bond_emb4(edge_attr[:, 3].long())

        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class VirtualNodeMixin(nn.Module):
    """
    Virtual Node для GIN — стандартное улучшение из OGB.

    Виртуальный узел связан со всеми атомами, агрегируя и рассылая
    глобальную информацию между слоями GIN без квадратичной сложности.
    """

    def __init__(self, num_layer: int, emb_dim: int):
        super().__init__()
        self.vn_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.vn_embedding.weight.data, 0)

        self.vn_mlps = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.vn_mlps.append(nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            ))

    def vn_update(self, h, batch, vn_h, layer_idx):
        """Обновление virtual node после GIN-слоя layer_idx."""
        if layer_idx >= len(self.vn_mlps):
            return h, vn_h

        from torch_geometric.nn import global_mean_pool as _gmp
        vn_h_new = _gmp(h, batch) + vn_h
        vn_h = self.vn_mlps[layer_idx](vn_h_new)
        h = h + vn_h[batch]
        return h, vn_h


class GINBackboneVN(nn.Module):
    """GIN backbone + Virtual Node (pretrained-совместимый формат входа)."""

    def __init__(self, num_layer: int = 5, emb_dim: int = 300,
                 drop_ratio: float = 0.5):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConvLayer(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        self.vn = VirtualNodeMixin(num_layer, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        vn_h = self.vn.vn_embedding(
            torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)
        )
        h = h + vn_h[batch]

        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h, vn_h = self.vn.vn_update(h, batch, vn_h, layer)
            h_list.append(h)

        return h_list[-1]


class GINBackboneOGBVN(nn.Module):
    """GIN backbone + OGB features + Virtual Node (from scratch only)."""

    def __init__(self, num_layer: int = 5, emb_dim: int = 300,
                 drop_ratio: float = 0.5):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim

        self.atom_embs = nn.ModuleList([
            nn.Embedding(NUM_ATOM_TYPE, emb_dim),
            nn.Embedding(NUM_CHIRALITY_TAG, emb_dim),
            nn.Embedding(NUM_DEGREE, emb_dim),
            nn.Embedding(NUM_FORMAL_CHARGE, emb_dim),
            nn.Embedding(NUM_NUM_HS, emb_dim),
            nn.Embedding(NUM_NUM_RADICAL_E, emb_dim),
            nn.Embedding(NUM_HYBRIDIZATION, emb_dim),
            nn.Embedding(NUM_IS_AROMATIC, emb_dim),
            nn.Embedding(NUM_IS_IN_RING, emb_dim),
        ])
        for emb in self.atom_embs:
            nn.init.xavier_uniform_(emb.weight.data)

        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINConvLayerOGB(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        self.vn = VirtualNodeMixin(num_layer, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = sum(emb(x[:, i]) for i, emb in enumerate(self.atom_embs))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        vn_h = self.vn.vn_embedding(
            torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)
        )
        h = h + vn_h[batch]

        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h, vn_h = self.vn.vn_update(h, batch, vn_h, layer)
            h_list.append(h)

        return h_list[-1]


class GINMultiTask(nn.Module):
    """
    GIN + multi-task classification head.

    Поддерживает загрузку pretrained backbone и обучение from scratch.
    Параметр backbone_type выбирает тип backbone:
      'standard'  — оригинальный GIN (2 atom features, совместим с pretrained)
      'vn'        — GIN + Virtual Node (2 atom features, совместим с pretrained)
      'ogb'       — OGB-style (9 atom features, from scratch only)
      'ogb_vn'    — OGB-style + Virtual Node (from scratch only)

    Параметр pool_type:
      'mean'      — global_mean_pool
      'attention' — GlobalAttention (learnable gate)
    """

    def __init__(self, num_tasks: int = 12, num_layer: int = 5,
                 emb_dim: int = 300, drop_ratio: float = 0.5,
                 backbone_type: str = "standard",
                 pool_type: str = "mean"):
        super().__init__()
        self.num_tasks = num_tasks
        self.emb_dim = emb_dim
        self.backbone_type = backbone_type
        self.pool_type = pool_type

        if backbone_type == "standard":
            self.backbone = GINBackbone(num_layer, emb_dim, drop_ratio)
        elif backbone_type == "vn":
            self.backbone = GINBackboneVN(num_layer, emb_dim, drop_ratio)
        elif backbone_type == "ogb":
            self.backbone = GINBackboneOGB(num_layer, emb_dim, drop_ratio)
        elif backbone_type == "ogb_vn":
            self.backbone = GINBackboneOGBVN(num_layer, emb_dim, drop_ratio)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        if pool_type == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.pool = AttentionalAggregation(gate_nn)
        else:
            self.pool = global_mean_pool

        self.classifier = nn.Linear(emb_dim, num_tasks)
        self._node_embeddings = None

    def load_pretrained(self, model_name: str = "supervised_contextpred"):
        """Загрузка pretrained backbone весов."""
        if self.backbone_type in ("ogb", "ogb_vn"):
            print(f"  ⚠ OGB backbone несовместим с pretrained весами — пропуск")
            return

        weights_path = WEIGHTS_DIR / f"{model_name}.pth"

        if not weights_path.exists():
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            url = PRETRAINED_URLS[model_name]
            print(f"Скачивание pretrained GIN ({model_name})...")
            urllib.request.urlretrieve(url, weights_path)
            print(f"Сохранено: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

        if self.backbone_type == "vn":
            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith("x_embedding") or k.startswith("gnns.") or k.startswith("batch_norms."):
                    backbone_state[k] = v
            self.backbone.load_state_dict(backbone_state, strict=False)
        else:
            self.backbone.load_state_dict(state_dict, strict=False)

        print(f"Pretrained backbone загружен ({model_name})")

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        if self.backbone_type in ("vn", "ogb_vn"):
            node_repr = self.backbone(x, edge_index, edge_attr, batch)
        else:
            node_repr = self.backbone(x, edge_index, edge_attr)
        self._node_embeddings = node_repr

        graph_repr = self.pool(node_repr, batch)
        return self.classifier(graph_repr)

    def get_node_embeddings(self):
        return self._node_embeddings


def download_pretrained(model_name: str = "supervised_contextpred"):
    """Скачивание pretrained весов."""
    weights_path = WEIGHTS_DIR / f"{model_name}.pth"
    if weights_path.exists():
        print(f"Веса уже скачаны: {weights_path}")
        return weights_path

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = PRETRAINED_URLS[model_name]
    print(f"Скачивание {url} ...")
    urllib.request.urlretrieve(url, weights_path)
    print(f"Сохранено: {weights_path}")
    return weights_path


# ── Обучение и оценка ──────────────────────────────────────────────────

def masked_bce_loss(pred, target):
    """BCE loss с маскированием NaN-меток (стандарт для multi-task MoleculeNet)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_masked = pred[mask]
    target_masked = target[mask]
    return F.binary_cross_entropy_with_logits(pred_masked, target_masked)


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-Weighted Multi-Task Loss (Kendall et al. 2018).

    Каждая задача получает обучаемый параметр log(σ²). Потери взвешиваются
    как L_i / (2·σ²_i) + log(σ_i), позволяя модели автоматически снижать
    вес шумных/сложных задач и повышать вес информативных.

    Это принципиально отличается от равного взвешивания или ручного подбора:
    модель обучает оптимальный баланс из данных.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=pred.device)
        num_valid_tasks = 0

        for t in range(pred.size(1)):
            task_mask = mask[:, t]
            if task_mask.sum() < 2:
                continue

            task_pred = pred[task_mask, t]
            task_target = target[task_mask, t]

            bce = F.binary_cross_entropy_with_logits(
                task_pred, task_target, reduction="mean"
            )

            precision = torch.exp(-self.log_vars[t])
            task_loss = precision * bce + self.log_vars[t]
            total_loss = total_loss + task_loss
            num_valid_tasks += 1

        if num_valid_tasks == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss / num_valid_tasks

    def get_task_weights(self):
        """Текущие веса задач (1/σ²) для интроспекции."""
        with torch.no_grad():
            return torch.exp(-self.log_vars).cpu().numpy()


def masked_focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    Focal Loss с маскированием NaN-меток.

    Снижает вес «лёгких» примеров: FL = -alpha * (1-p_t)^gamma * log(p_t).
    Полезен при сильном дисбалансе классов (Tox21: pos_rate 5-15%).
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_masked = pred[mask]
    target_masked = target[mask]

    bce = F.binary_cross_entropy_with_logits(pred_masked, target_masked, reduction="none")
    p_t = torch.sigmoid(pred_masked) * target_masked + (1 - torch.sigmoid(pred_masked)) * (1 - target_masked)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def train_epoch_multitask(model, loader, optimizer, device, xai_loss_fn=None,
                          xai_lambda=0.0, loss_fn=None,
                          toxicophore_loss_fn=None, toxicophore_lambda=0.0):
    """Один эпох multi-task обучения с опциональным XAI и toxicophore loss."""
    if loss_fn is None:
        loss_fn = masked_bce_loss
    model.train()
    total_loss = 0
    total_xai = 0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss_task = loss_fn(out, batch.y)

        loss_xai = torch.tensor(0.0, device=device)
        if xai_loss_fn is not None and xai_lambda > 0:
            loss_xai = xai_loss_fn(model, batch, device)

        loss_tox = torch.tensor(0.0, device=device)
        if toxicophore_loss_fn is not None and toxicophore_lambda > 0:
            loss_tox = toxicophore_loss_fn(model, batch, device)

        loss = loss_task + xai_lambda * loss_xai + toxicophore_lambda * loss_tox
        loss.backward()
        optimizer.step()

        total_loss += loss_task.item() * batch.num_graphs
        total_xai += (loss_xai.item() + loss_tox.item()) * batch.num_graphs
        n_graphs += batch.num_graphs

    return total_loss / n_graphs, total_xai / n_graphs


@torch.no_grad()
def evaluate_multitask(model, loader, device, task_names=None):
    """Оценка multi-task модели (per-task ROC-AUC → среднее)."""
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


def train_gin(
    train_data, val_data, test_data,
    num_tasks: int = 12,
    task_names: list = None,
    pretrained: str = None,
    xai_loss_fn=None,
    xai_lambda: float = 0.0,
    lr: float = 1e-3,
    backbone_lr: float = None,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    device: str = None,
    save_dir: str = None,
    model_name: str = "gin",
    backbone_type: str = "standard",
    pool_type: str = "mean",
    use_focal_loss: bool = False,
    use_uncertainty_loss: bool = False,
    toxicophore_loss_fn=None,
    toxicophore_lambda: float = 0.0,
):
    """
    Обучение GIN multi-task модели.

    Параметры:
        pretrained: имя pretrained backbone ('supervised_contextpred', None)
        xai_loss_fn: функция XAI-loss (или None)
        xai_lambda: вес XAI-loss
        backbone_lr: LR для backbone (если None — используется lr для всего)
        backbone_type: 'standard' | 'vn' | 'ogb' | 'ogb_vn'
        pool_type: 'mean' | 'attention'
        use_focal_loss: использовать focal loss вместо BCE
        use_uncertainty_loss: Uncertainty-Weighted MTL (Kendall et al. 2018)
        toxicophore_loss_fn: Toxicophore-Guided XAI loss (или None)
        toxicophore_lambda: вес toxicophore loss
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_label = pretrained or "scratch"
    xai_label = f"+XAI(λ={xai_lambda})" if xai_lambda > 0 else ""
    vn_label = "+VN" if "vn" in backbone_type else ""
    ogb_label = "+OGB" if "ogb" in backbone_type else ""
    attn_label = "+Attn" if pool_type == "attention" else ""
    focal_label = "+Focal" if use_focal_loss else ""
    unc_label = "+UncMTL" if use_uncertainty_loss else ""
    tox_label = f"+Tox(λ={toxicophore_lambda})" if toxicophore_lambda > 0 else ""

    print(f"\n{'='*60}")
    print(f"Обучение {model_name}: GIN ({pretrained_label}{vn_label}{ogb_label}{attn_label}{focal_label}{unc_label}{tox_label}{xai_label})")
    print(f"Задач: {num_tasks}, Устройство: {device}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = GINMultiTask(
        num_tasks=num_tasks,
        backbone_type=backbone_type,
        pool_type=pool_type,
    ).to(device)

    if pretrained:
        model.load_pretrained(pretrained)

    uncertainty_loss = None
    if use_uncertainty_loss:
        uncertainty_loss = UncertaintyWeightedLoss(num_tasks).to(device)
        print(f"  Uncertainty-Weighted MTL включён ({num_tasks} обучаемых σ²)")

    if backbone_lr is not None and backbone_lr != lr:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
        if pool_type == "attention":
            param_groups.append({"params": model.pool.parameters(), "lr": lr})
        if uncertainty_loss is not None:
            param_groups.append({"params": uncertainty_loss.parameters(), "lr": lr})
        print(f"  Differential LR: backbone={backbone_lr}, head={lr}")
    else:
        all_params = list(model.parameters())
        if uncertainty_loss is not None:
            all_params += list(uncertainty_loss.parameters())
        param_groups = all_params

    optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    if use_uncertainty_loss and uncertainty_loss is not None:
        loss_fn = uncertainty_loss
    elif use_focal_loss:
        loss_fn = masked_focal_loss
    else:
        loss_fn = masked_bce_loss

    best_val_metric = -1
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss, xai_loss = train_epoch_multitask(
            model, train_loader, optimizer, device,
            xai_loss_fn=xai_loss_fn, xai_lambda=xai_lambda,
            loss_fn=loss_fn,
            toxicophore_loss_fn=toxicophore_loss_fn,
            toxicophore_lambda=toxicophore_lambda,
        )

        val_metrics = evaluate_multitask(model, val_loader, device, task_names)
        val_mean_auc = val_metrics["mean_roc_auc"]
        scheduler.step(val_mean_auc)

        if val_mean_auc > best_val_metric:
            best_val_metric = val_mean_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            xai_str = f" XAI: {xai_loss:.4f}" if xai_lambda > 0 else ""
            print(f"  Эпоха {epoch:3d} | Loss: {loss:.4f}{xai_str} | "
                  f"Val mean-ROC-AUC: {val_mean_auc:.4f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics = evaluate_multitask(model, val_loader, device, task_names)
    test_metrics = evaluate_multitask(model, test_loader, device, task_names)

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  mean-ROC-AUC: {val_metrics['mean_roc_auc']:.4f}")
    print(f"Test mean-ROC-AUC: {test_metrics['mean_roc_auc']:.4f}")

    results = {
        "model_name": model_name,
        "pretrained": pretrained_label,
        "backbone_type": backbone_type,
        "pool_type": pool_type,
        "use_focal_loss": use_focal_loss,
        "use_uncertainty_loss": use_uncertainty_loss,
        "xai_lambda": xai_lambda,
        "toxicophore_lambda": toxicophore_lambda,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }

    if uncertainty_loss is not None:
        weights = uncertainty_loss.get_task_weights()
        names = task_names or [f"task_{i}" for i in range(num_tasks)]
        results["learned_task_weights"] = {
            n: round(float(w), 4) for n, w in zip(names, weights)
        }
        print(f"\nОбученные веса задач (1/σ²):")
        for n, w in zip(names, weights):
            print(f"  {n}: {w:.4f}")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Сохранено в {save_path}")

    return results, model
