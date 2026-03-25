"""
Hybrid GIN(2D) + SchNet(3D) Fusion для PharmaKinetics.

Научная новизна: объединение топологической (GIN, 2D граф) и
геометрической (SchNet, 3D координаты) информации через attention gate.

GIN ловит: подструктуры, функциональные группы, топологию
SchNet ловит: стерические взаимодействия, пространственную близость, 3D-форму

Attention gate учит оптимальный баланс 2D/3D из данных:
  alpha = sigmoid(W @ [h_2d; h_3d])
  h_fused = alpha * h_2d + (1 - alpha) * h_3d
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

from src.models.gin_pretrained import (
    GINMultiTask,
    masked_bce_loss,
    UncertaintyWeightedLoss,
)
from src.utils.metrics import compute_multitask_metrics


class HybridGINSchNet(nn.Module):
    """
    Hybrid 2D+3D fusion: Pretrained GIN (2D граф) + SchNet (3D координаты).

    Два encoder'а обрабатывают молекулу параллельно. Attention gate
    динамически взвешивает вклад каждого представления.

    SchNet interaction layers используются напрямую (без lin1/lin2/readout,
    которые предназначены для скалярной регрессии энергии).
    """

    def __init__(self, num_tasks: int = 12,
                 gin_emb_dim: int = 300,
                 gin_backbone_type: str = "vn",
                 schnet_hidden: int = 128,
                 schnet_interactions: int = 6,
                 schnet_cutoff: float = 10.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.gin_emb_dim = gin_emb_dim
        self.schnet_hidden = schnet_hidden

        self.gin = GINMultiTask(
            num_tasks=num_tasks,
            backbone_type=gin_backbone_type,
            pool_type="mean",
        )

        self.schnet_backbone = SchNet(
            hidden_channels=schnet_hidden,
            num_filters=schnet_hidden,
            num_interactions=schnet_interactions,
            num_gaussians=50,
            cutoff=schnet_cutoff,
            atomref=None,
            mean=None,
            std=None,
        )

        self.proj_2d = nn.Linear(gin_emb_dim, gin_emb_dim)
        self.proj_3d = nn.Linear(schnet_hidden, gin_emb_dim)

        self.gate = nn.Sequential(
            nn.Linear(gin_emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(gin_emb_dim, gin_emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gin_emb_dim // 2, num_tasks),
        )

        self._alpha = None

    def _extract_3d_repr(self, z, pos, batch):
        """Извлечение graph-level 3D представления из SchNet interaction layers."""
        sn = self.schnet_backbone
        h = sn.embedding(z)
        edge_index, edge_weight = sn.interaction_graph(pos, batch)
        edge_attr = sn.distance_expansion(edge_weight)

        for interaction in sn.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return global_mean_pool(h, batch)

    def forward(self, data_2d, data_3d):
        """
        data_2d: PyG Batch (GIN-формат: x, edge_index, edge_attr, batch)
        data_3d: PyG Batch (SchNet-формат: z, pos, batch)
        """
        x_2d, ei_2d, ea_2d, batch_2d = (
            data_2d.x, data_2d.edge_index, data_2d.edge_attr, data_2d.batch
        )

        if self.gin.backbone_type in ("vn", "ogb_vn"):
            node_repr_2d = self.gin.backbone(x_2d, ei_2d, ea_2d, batch_2d)
        else:
            node_repr_2d = self.gin.backbone(x_2d, ei_2d, ea_2d)

        h_2d = global_mean_pool(node_repr_2d, batch_2d)

        h_3d = self._extract_3d_repr(data_3d.z, data_3d.pos, data_3d.batch)

        h_2d_proj = self.proj_2d(h_2d)
        h_3d_proj = self.proj_3d(h_3d)

        alpha = self.gate(torch.cat([h_2d_proj, h_3d_proj], dim=1))
        self._alpha = alpha.detach()

        h_fused = alpha * h_2d_proj + (1 - alpha) * h_3d_proj
        return self.classifier(h_fused)

    def load_gin_pretrained(self, model_name: str = "supervised_contextpred"):
        """Загрузка pretrained весов только для GIN backbone."""
        self.gin.load_pretrained(model_name)

    def get_alpha(self):
        """Возвращает attention gate alpha (для интроспекции)."""
        return self._alpha


# ── Paired Dataset ────────────────────────────────────────────────────

class PairedBatch:
    """Контейнер для paired 2D+3D батчей."""
    def __init__(self, batch_2d, batch_3d, y):
        self.batch_2d = batch_2d
        self.batch_3d = batch_3d
        self.y = y

    def to(self, device):
        self.batch_2d = self.batch_2d.to(device)
        self.batch_3d = self.batch_3d.to(device)
        self.y = self.y.to(device)
        return self


def build_paired_dataset(df, task_columns, build_gin_fn, build_3d_fn):
    """
    Строит параллельные датасеты 2D (GIN) и 3D (SchNet) для одних молекул.

    Молекула попадает в итоговый датасет только если оба представления
    успешно созданы. Возвращает (list_2d, list_3d) одинаковой длины.
    """
    dataset_2d = []
    dataset_3d = []
    failed = 0

    for _, row in df.iterrows():
        y_tasks = row[task_columns].values.astype(float)
        smiles = row["smiles"]

        try:
            data_2d = build_gin_fn(smiles, y_tasks)
            data_3d = build_3d_fn(smiles, y_tasks)
        except Exception:
            data_2d, data_3d = None, None

        if data_2d is not None and data_3d is not None:
            dataset_2d.append(data_2d)
            dataset_3d.append(data_3d)
        else:
            failed += 1

    if failed > 0:
        print(f"  Paired: {failed}/{len(df)} молекул пропущены (нет 2D или 3D)")
    return dataset_2d, dataset_3d


# ── Обучение Hybrid ──────────────────────────────────────────────────

def _collate_paired(batch_2d_list, batch_3d_list, device):
    """Создание paired батча из списков 2D и 3D Data."""
    from torch_geometric.data import Batch
    b2d = Batch.from_data_list(batch_2d_list)
    b3d = Batch.from_data_list(batch_3d_list)
    y = b2d.y
    return PairedBatch(b2d, b3d, y).to(device)


def train_hybrid(
    train_2d, train_3d,
    val_2d, val_3d,
    test_2d, test_3d,
    num_tasks: int = 12,
    task_names: list = None,
    pretrained: str = "supervised_contextpred",
    lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 64,
    device: str = None,
    save_dir: str = None,
    model_name: str = "hybrid_gin_schnet",
    use_uncertainty_loss: bool = False,
):
    """
    Обучение Hybrid GIN+SchNet fusion модели.

    train_2d/train_3d: параллельные списки Data (одинаковой длины).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    unc_label = "+UncMTL" if use_uncertainty_loss else ""
    print(f"\n{'='*60}")
    print(f"Обучение {model_name}: Hybrid GIN(2D) + SchNet(3D){unc_label}")
    print(f"Задач: {num_tasks}, Устройство: {device}")
    print(f"{'='*60}")

    assert len(train_2d) == len(train_3d), "2D и 3D train датасеты разной длины"

    model = HybridGINSchNet(num_tasks=num_tasks).to(device)

    if pretrained:
        model.load_gin_pretrained(pretrained)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {n_params:,}")

    uncertainty_loss = None
    if use_uncertainty_loss:
        uncertainty_loss = UncertaintyWeightedLoss(num_tasks).to(device)
        print(f"  Uncertainty-Weighted MTL включён")

    gin_params = list(model.gin.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("gin.")]
    if uncertainty_loss is not None:
        other_params += list(uncertainty_loss.parameters())

    param_groups = [
        {"params": gin_params, "lr": backbone_lr},
        {"params": other_params, "lr": lr},
    ]
    print(f"  Differential LR: GIN={backbone_lr}, rest={lr}")

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    loss_fn = uncertainty_loss if use_uncertainty_loss else masked_bce_loss

    best_val_metric = -1
    best_epoch = 0
    best_state = None
    start = time.time()

    n_train = len(train_2d)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        indices = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = indices[i:i+batch_size]
            b2d = [train_2d[j] for j in idx]
            b3d = [train_3d[j] for j in idx]
            paired = _collate_paired(b2d, b3d, device)

            optimizer.zero_grad()
            out = model(paired.batch_2d, paired.batch_3d)
            loss = loss_fn(out, paired.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        val_metrics = _evaluate_hybrid(model, val_2d, val_3d, batch_size,
                                       device, task_names)
        val_auc = val_metrics["mean_roc_auc"]
        scheduler.step(val_auc)

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            alpha_mean = model.get_alpha().mean().item() if model.get_alpha() is not None else 0
            print(f"  Эпоха {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val ROC-AUC: {val_auc:.4f} | α(2D): {alpha_mean:.3f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics = _evaluate_hybrid(model, val_2d, val_3d, batch_size,
                                   device, task_names)
    test_metrics = _evaluate_hybrid(model, test_2d, test_3d, batch_size,
                                    device, task_names)
    elapsed = time.time() - start

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  mean-ROC-AUC: {val_metrics['mean_roc_auc']:.4f}")
    print(f"Test mean-ROC-AUC: {test_metrics['mean_roc_auc']:.4f}")
    print(f"Время: {elapsed:.0f} сек")

    results = {
        "model_name": model_name,
        "model_type": "Hybrid-GIN-SchNet",
        "pretrained_gin": pretrained or "scratch",
        "num_params": n_params,
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "use_uncertainty_loss": use_uncertainty_loss,
        "train_size": len(train_2d),
        "val_size": len(val_2d),
        "test_size": len(test_2d),
        "elapsed_sec": round(elapsed, 1),
    }

    if uncertainty_loss is not None:
        weights = uncertainty_loss.get_task_weights()
        names = task_names or [f"task_{i}" for i in range(num_tasks)]
        results["learned_task_weights"] = {
            n: round(float(w), 4) for n, w in zip(names, weights)
        }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Сохранено в {save_path}")

    return results, model


@torch.no_grad()
def _evaluate_hybrid(model, data_2d, data_3d, batch_size, device, task_names=None):
    """Evaluation helper для hybrid модели."""
    model.eval()
    all_preds = []
    all_targets = []

    n = len(data_2d)
    for i in range(0, n, batch_size):
        b2d = data_2d[i:i+batch_size]
        b3d = data_3d[i:i+batch_size]
        paired = _collate_paired(b2d, b3d, device)

        out = model(paired.batch_2d, paired.batch_3d)
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(paired.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_multitask_metrics(all_targets, all_preds, task_names)
