"""
Pseudo-3D графовая модель PharmaKinetics.

Та же архитектура MPNN, но обучается на графах с 3D-признаками:
межатомные расстояния из конформеров (RDKit ETKDG), закодированные
через радиальные базисные функции (RBF) на рёбрах.

Фичеризация происходит в src/data/featurizer.smiles_to_graph_3d(),
модель — та же MPNN с увеличенным edge_dim (7 + 16 = 23).
"""

from src.models.gnn import train_gnn


def train_gnn_3d(
    train_data,
    val_data,
    test_data,
    node_dim: int,
    edge_dim: int,
    **kwargs,
):
    """Обучение pseudo-3D GNN. edge_dim включает RBF-дистанции."""
    return train_gnn(
        train_data,
        val_data,
        test_data,
        node_dim=node_dim,
        edge_dim=edge_dim,
        model_name=kwargs.pop("model_name", "gnn_3d"),
        **kwargs,
    )
