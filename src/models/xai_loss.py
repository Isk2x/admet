"""
Explainability-Aware Loss для PharmaKinetics.

Регуляризация по объяснимости, встроенная в процесс обучения:
  - L_faithfulness: штраф за нечувствительность предсказаний к маскированию
    важных атомов (причинная состоятельность)
  - L_stability: штраф за нестабильность объяснений при возмущении графа
    (drop рёбер)

L_total = L_task + λ₁·L_faithfulness + λ₂·L_stability
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge


def _get_node_importance_from_embeddings(model):
    """
    Приближение важности через L2-норму node embeddings.
    Не требует второго backward pass — быстрее чем gradient*input.
    """
    h = model.get_node_embeddings()
    if h is None:
        return None
    return h.norm(dim=-1)


def compute_faithfulness_loss(model, batch, device, top_frac=0.2):
    """
    L_faithfulness: модель должна менять предсказание при маскировании
    важных атомов.

    Алгоритм:
      1. Прямой проход → предсказание p и node embeddings h
      2. Важность = ||h_i||₂
      3. Мягкое маскирование top-k% атомов через sigmoid gate
      4. Повторный проход → предсказание p_masked
      5. Loss = max(0, margin - |p - p_masked|)
    """
    node_importance = _get_node_importance_from_embeddings(model)
    if node_importance is None:
        return torch.tensor(0.0, device=device)

    num_nodes_per_graph = torch.bincount(batch.batch)
    max_nodes = num_nodes_per_graph.max().item()

    offset = 0
    mask_weights = torch.ones(node_importance.size(0), device=device)
    for i, n in enumerate(num_nodes_per_graph):
        n = n.item()
        k = max(1, int(n * top_frac))
        imp = node_importance[offset:offset + n]
        _, top_idx = imp.topk(k)
        mask_weights[offset + top_idx] = 0.0
        offset += n

    original_x = batch.x.clone()
    if batch.x.dtype == torch.long:
        with torch.no_grad():
            h_original = model.backbone(batch.x, batch.edge_index, batch.edge_attr)
        h_masked = h_original * mask_weights.unsqueeze(-1)
        from torch_geometric.nn import global_mean_pool
        graph_repr_masked = global_mean_pool(h_masked, batch.batch)
        pred_masked = model.classifier(graph_repr_masked)
    else:
        batch.x = batch.x * mask_weights.unsqueeze(-1)
        pred_masked = model(batch)
        batch.x = original_x

    pred_original = model(batch)
    pred_original_prob = torch.sigmoid(pred_original)
    pred_masked_prob = torch.sigmoid(pred_masked)

    target_mask = ~torch.isnan(batch.y)
    if target_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    diff = (pred_original_prob - pred_masked_prob).abs()
    diff_masked = diff[target_mask]

    margin = 0.1
    loss = F.relu(margin - diff_masked).mean()

    return loss


def compute_stability_loss(model, batch, device, edge_drop_ratio=0.1):
    """
    L_stability: объяснения должны быть устойчивы при возмущении графа.

    Алгоритм:
      1. Прямой проход на оригинальном графе → importance₁
      2. Dropout рёбер → прямой проход → importance₂
      3. Loss = 1 - cosine_similarity(importance₁, importance₂)
    """
    imp_original = _get_node_importance_from_embeddings(model)
    if imp_original is None:
        return torch.tensor(0.0, device=device)
    imp_original = imp_original.detach()

    dropped_edge_index, edge_mask = dropout_edge(
        batch.edge_index, p=edge_drop_ratio, training=True
    )
    dropped_edge_attr = batch.edge_attr[edge_mask]

    node_repr = model.backbone(batch.x, dropped_edge_index, dropped_edge_attr)
    imp_perturbed = node_repr.norm(dim=-1)

    cos_sim = F.cosine_similarity(
        imp_original.unsqueeze(0),
        imp_perturbed.unsqueeze(0),
    )

    loss = 1.0 - cos_sim.mean()
    return loss


def combined_xai_loss(model, batch, device,
                      faith_weight=0.5, stab_weight=0.5,
                      top_frac=0.2, edge_drop_ratio=0.1):
    """Комбинированный XAI loss: λ₁·L_faith + λ₂·L_stab."""
    l_faith = compute_faithfulness_loss(model, batch, device, top_frac)
    l_stab = compute_stability_loss(model, batch, device, edge_drop_ratio)
    return faith_weight * l_faith + stab_weight * l_stab
