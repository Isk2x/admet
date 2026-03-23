"""
ChemBERTa-2 для PharmaKinetics.

Transformer-модель на основе RoBERTa, pretrained на 77M молекул из PubChem.
Работает напрямую со SMILES-строками (без графовой конвертации).

Поддерживает:
  - Fine-tuning на Tox21 (12 multi-task задач)
  - Извлечение attention-based атомной атрибуции
  - Маппинг SMILES-токенов → атомы молекулы
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.metrics import compute_multitask_metrics

CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"


class SmilesDataset(Dataset):
    """Датасет SMILES + multi-task метки для ChemBERTa."""

    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles_list[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }
        return item


class ChemBERTaMultiTask(nn.Module):
    """
    ChemBERTa-2 + multi-task classification head.

    Архитектура: RoBERTa backbone (6 слоёв, 768 dim, ~80M params) →
    [CLS] representation → MLP → 12 logits (по задачам Tox21).
    """

    def __init__(self, num_tasks=12, dropout=0.1):
        super().__init__()
        from transformers import RobertaModel

        self.num_tasks = num_tasks
        self.backbone = RobertaModel.from_pretrained(CHEMBERTA_MODEL_NAME)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tasks),
        )

        self._last_attentions = None

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        self._last_attentions = outputs.attentions
        logits = self.classifier(cls_repr)
        return logits

    def get_attention_weights(self):
        """Attention weights последнего forward pass (все слои)."""
        return self._last_attentions

    @staticmethod
    def from_weights(weights_path, num_tasks=12, device="cpu"):
        """Загрузка модели из сохранённых весов."""
        model = ChemBERTaMultiTask(num_tasks=num_tasks)
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        return model


def masked_bce_loss(pred, target):
    """BCE loss с маскированием NaN-меток."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(pred[mask], target[mask])


# ── Attention → атомная атрибуция ────────────────────────────────────────

def smiles_token_to_atom_map(smiles: str, tokenizer) -> dict:
    """
    Маппинг SMILES-токенов → индексы атомов.

    Проходит по символам SMILES и сопоставляет каждый токен с атомом,
    пропуская служебные символы (скобки, цифры, =, #, +, -, и т.д.).
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    tokens = tokenizer.tokenize(smiles)
    atom_symbols = set()
    for atom in mol.GetAtoms():
        atom_symbols.add(atom.GetSymbol())
        atom_symbols.add(atom.GetSymbol().lower())

    token_to_atom = {}
    atom_idx = 0
    num_atoms = mol.GetNumAtoms()

    for tok_idx, token in enumerate(tokens):
        clean = token.replace("Ġ", "").replace("##", "")
        if not clean:
            continue

        is_atom_token = False
        for sym in sorted(atom_symbols, key=len, reverse=True):
            if clean.startswith(sym) or clean.startswith(sym.upper()):
                is_atom_token = True
                break

        if is_atom_token and atom_idx < num_atoms:
            token_to_atom[tok_idx] = atom_idx
            atom_idx += 1

    return token_to_atom


def compute_atom_importance_chemberta(model, smiles, tokenizer, device="cpu"):
    """
    Атомная атрибуция через attention weights ChemBERTa.

    Агрегирует attention с [CLS]-токена на все остальные токены
    по последнему слою (среднее по головам), затем маппит на атомы.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(0)

    num_atoms = mol.GetNumAtoms()
    enc = tokenizer(
        smiles, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
        attentions = model.get_attention_weights()

    last_layer_attn = attentions[-1].squeeze(0)
    cls_attn = last_layer_attn[:, 0, :].mean(dim=0).cpu().numpy()

    token_to_atom = smiles_token_to_atom_map(smiles, tokenizer)

    atom_importance = np.zeros(num_atoms)
    for tok_idx, atom_idx in token_to_atom.items():
        offset = tok_idx + 1
        if offset < len(cls_attn):
            atom_importance[atom_idx] += cls_attn[offset]

    if atom_importance.max() > 0:
        atom_importance = atom_importance / atom_importance.max()

    return atom_importance


# ── Обучение ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = masked_bce_loss(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        n += input_ids.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, task_names=None):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_multitask_metrics(all_targets, all_preds, task_names)


def train_chemberta(
    train_smiles, train_labels,
    val_smiles, val_labels,
    test_smiles, test_labels,
    num_tasks=12,
    task_names=None,
    lr=2e-5,
    epochs=20,
    patience=5,
    batch_size=32,
    device=None,
    save_dir=None,
    model_name="chemberta",
):
    """
    Fine-tuning ChemBERTa-2 на multi-task Tox21.

    Параметры:
        lr: learning rate (2e-5 стандарт для fine-tuning трансформеров)
        epochs: максимум эпох
        patience: early stopping patience
    """
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Fine-tuning ChemBERTa-2 ({CHEMBERTA_MODEL_NAME})")
    print(f"Задач: {num_tasks}, Устройство: {device}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)

    train_ds = SmilesDataset(train_smiles, train_labels, tokenizer)
    val_ds = SmilesDataset(val_smiles, val_labels, tokenizer)
    test_ds = SmilesDataset(test_smiles, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = ChemBERTaMultiTask(num_tasks=num_tasks).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_metric = -1
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, task_names)
        val_auc = val_metrics["mean_roc_auc"]

        if val_auc > best_val_metric:
            best_val_metric = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Эпоха {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val mean-ROC-AUC: {val_auc:.4f}")

        if epoch - best_epoch >= patience:
            print(f"  Ранняя остановка на эпохе {epoch} (лучшая: {best_epoch})")
            break

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics = evaluate(model, val_loader, device, task_names)
    test_metrics = evaluate(model, test_loader, device, task_names)

    print(f"\nЛучшая эпоха: {best_epoch}")
    print(f"Val  mean-ROC-AUC: {val_metrics['mean_roc_auc']:.4f}")
    print(f"Test mean-ROC-AUC: {test_metrics['mean_roc_auc']:.4f}")

    results = {
        "model_name": model_name,
        "backbone": "ChemBERTa-77M-MLM",
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path / f"{model_name}_weights.pt")
        with open(save_path / f"{model_name}_metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"Сохранено в {save_path}")

    return results, model
