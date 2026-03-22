"""
Атомная атрибуция для GIN-моделей PharmaKinetics.

Для GIN с integer-фичами gradient*input не работает напрямую (x — индексы).
Используем L2-норму node embeddings как прокси важности атомов.
"""

import torch
import numpy as np
from torch_geometric.data import Data


def compute_atom_importance_gin(model, data: Data, device: str = "cpu") -> np.ndarray:
    """
    Важность атомов через L2-норму node embeddings после backbone.

    Для GIN-моделей с integer-входами gradient*input неприменим,
    поэтому используем magnitude embeddings — стандартный подход
    для моделей с embedding-слоями.
    """
    model.eval()
    data = data.clone().to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        _ = model(data)
        node_emb = model.get_node_embeddings()

    importance = node_emb.norm(dim=-1)
    return importance.cpu().numpy()


def get_top_k_atoms_gin(importance: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Индексы top-k% наиболее важных атомов."""
    k = max(1, int(len(importance) * fraction))
    return np.argsort(importance)[-k:]
