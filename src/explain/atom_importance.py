"""
Атомная атрибуция PharmaKinetics.

Вычисляет важность каждого атома через метод gradient * input:
для каждого атома — L2-норма произведения градиента предсказания
на вектор признаков узла.
"""

import torch
import numpy as np
from torch_geometric.data import Data


def compute_atom_importance(model, data: Data, device: str = "cpu") -> np.ndarray:
    """
    Вычисление важности атомов через gradient * input.

    Возвращает массив (num_atoms,) с неотрицательными оценками важности.
    """
    model.eval()
    data = data.clone().to(device)
    data.x.requires_grad_(True)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    out = model(data)
    prob = torch.sigmoid(out)
    prob.backward()

    grad = data.x.grad  # (num_atoms, node_dim)
    importance = (grad * data.x).abs().sum(dim=1)  # (num_atoms,)

    return importance.detach().cpu().numpy()


def get_top_k_atoms(importance: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Индексы top-k% наиболее важных атомов."""
    k = max(1, int(len(importance) * fraction))
    return np.argsort(importance)[-k:]
