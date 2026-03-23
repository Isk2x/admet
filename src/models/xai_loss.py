"""
Explainability-Aware Loss для PharmaKinetics.

Три компонента XAI-регуляризации:
  - L_faithfulness: штраф за нечувствительность предсказаний к маскированию
    важных атомов (причинная состоятельность)
  - L_stability: штраф за нестабильность объяснений при возмущении графа
    (drop рёбер)
  - L_toxicophore: штраф за несоответствие importance и известных
    токсикофоров (доменная регуляризация)

L_total = L_task + λ₁·L_faithfulness + λ₂·L_stability + λ₃·L_toxicophore
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


# ── Toxicophore-Guided XAI ───────────────────────────────────────────────

TOXICOPHORE_SMARTS = {
    "nitro": "[N+](=O)[O-]",
    "aromatic_amine": "[NH2]c",
    "epoxide": "C1OC1",
    "aldehyde": "[CH]=O",
    "acyl_halide": "C(=O)[F,Cl,Br,I]",
    "michael_acceptor": "C=CC(=O)",
    "aryl_halide": "c[F,Cl,Br,I]",
    "nitroso": "[N]=O",
    "azide": "[N-]=[N+]=[N-]",
    "isocyanate": "N=C=O",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br,I]",
    "phosphate_ester": "P(=O)(O)(O)O",
}


def _precompute_toxicophore_masks():
    """Компиляция SMARTS-паттернов (один раз при импорте)."""
    from rdkit import Chem
    compiled = {}
    for name, smarts in TOXICOPHORE_SMARTS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled[name] = pat
    return compiled


_COMPILED_PATTERNS = None


def _get_patterns():
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = _precompute_toxicophore_masks()
    return _COMPILED_PATTERNS


def find_toxicophore_atoms(smiles: str) -> set:
    """
    Поиск атомов, принадлежащих известным токсикофорам.

    Возвращает set атомных индексов, попавших хотя бы в один
    из SMARTS-паттернов в TOXICOPHORE_SMARTS.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    patterns = _get_patterns()
    tox_atoms = set()
    for pat in patterns.values():
        for match in mol.GetSubstructMatches(pat):
            tox_atoms.update(match)

    return tox_atoms


def compute_toxicophore_loss(model, batch, device):
    """
    L_toxicophore: importance атомов-токсикофоров >= средняя importance.

    Для каждого графа в батче:
      1. Определяем атомы, входящие в известные токсикофоры (SMARTS)
      2. Сравниваем среднюю importance токсикофорных атомов с общей
      3. Штраф = max(0, mean_all - mean_toxicophore)

    Интуиция: если в молекуле есть нитро-группа и модель считает
    её токсичной, importance нитро-атомов не должна быть ниже средней.
    Это инъекция доменного знания в процесс обучения.
    """
    node_importance = _get_node_importance_from_embeddings(model)
    if node_importance is None:
        return torch.tensor(0.0, device=device)

    if not hasattr(batch, 'smiles') or batch.smiles is None:
        return torch.tensor(0.0, device=device)

    num_nodes_per_graph = torch.bincount(batch.batch)
    penalties = []

    offset = 0
    for i, n_nodes in enumerate(num_nodes_per_graph):
        n = n_nodes.item()
        graph_imp = node_importance[offset:offset + n]

        smiles_i = batch.smiles[i] if isinstance(batch.smiles, list) else batch.smiles
        tox_atoms = find_toxicophore_atoms(smiles_i)

        if tox_atoms and len(tox_atoms) < n:
            tox_idx = [a for a in tox_atoms if a < n]
            if tox_idx:
                tox_imp = graph_imp[tox_idx].mean()
                all_imp = graph_imp.mean()
                penalty = F.relu(all_imp - tox_imp)
                penalties.append(penalty)

        offset += n

    if not penalties:
        return torch.tensor(0.0, device=device)

    return torch.stack(penalties).mean()
