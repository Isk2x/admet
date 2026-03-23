"""
PharmaKinetics MVP — FastAPI бэкенд.

Принимает SMILES-строку, прогоняет через выбранную модель
(GIN или ChemBERTa), возвращает предсказания токсичности
по 12 задачам Tox21 и атомную атрибуцию.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import cm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gin_pretrained import GINMultiTask
from src.data.featurizer_gin import smiles_to_graph_gin
from src.explain.atom_importance_gin import compute_atom_importance_gin
from src.data.loader_multitask import TOX21_TASKS

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
DEVICE = "cpu"

# ── Реестр моделей ───────────────────────────────────────────────────────

AVAILABLE_MODELS = {}

GIN_WEIGHTS_PRIORITY = [
    ("gin_pretrained_vn_xai_s42_weights.pt", "vn"),
    ("gin_pretrained_vn_xai_weights.pt", "vn"),
    ("gin_pretrained_xai_weights.pt", "standard"),
    ("gin_pretrained_weights.pt", "standard"),
]

CHEMBERTA_WEIGHTS = [
    "chemberta_tox21_s42_weights.pt",
    "chemberta_tox21_weights.pt",
]


def _load_gin():
    """Загрузка GIN-модели с лучшими доступными весами."""
    for name, backbone_type in GIN_WEIGHTS_PRIORITY:
        path = MODELS_DIR / name
        if path.exists():
            model = GINMultiTask(num_tasks=12, backbone_type=backbone_type, pool_type="mean")
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state, strict=False)
            model.eval().to(DEVICE)
            print(f"[GIN] Загружен: {name} (backbone: {backbone_type})")
            return model, backbone_type
    return None, None


def _load_chemberta():
    """Загрузка ChemBERTa-модели (если веса есть)."""
    for name in CHEMBERTA_WEIGHTS:
        path = MODELS_DIR / name
        if path.exists():
            try:
                from src.models.chemberta import ChemBERTaMultiTask
                model = ChemBERTaMultiTask.from_weights(path, num_tasks=12, device=DEVICE)
                print(f"[ChemBERTa] Загружен: {name}")
                return model
            except Exception as e:
                print(f"[ChemBERTa] Ошибка загрузки {name}: {e}")
                return None
    print("[ChemBERTa] Веса не найдены — модель будет недоступна до fine-tuning")
    return None


gin_model, gin_backbone = _load_gin()
if gin_model:
    AVAILABLE_MODELS["gin"] = {
        "model": gin_model,
        "backbone": gin_backbone,
        "display_name": f"GIN {'+ VN' if 'vn' in (gin_backbone or '') else 'Pretrained'}",
        "params": "~2M",
        "type": "graph",
    }

chemberta_model = _load_chemberta()
if chemberta_model:
    AVAILABLE_MODELS["chemberta"] = {
        "model": chemberta_model,
        "display_name": "ChemBERTa-2 (77M)",
        "params": "~80M",
        "type": "transformer",
    }

print(f"\nДоступные модели: {list(AVAILABLE_MODELS.keys())}")


# ── Визуализация молекулы ────────────────────────────────────────────────

def _importance_to_colors(importance: np.ndarray) -> dict:
    if importance.max() - importance.min() < 1e-8:
        norm = np.ones_like(importance) * 0.5
    else:
        norm = (importance - importance.min()) / (importance.max() - importance.min())

    colormap = cm.get_cmap("RdYlGn_r")
    colors = {}
    for i, val in enumerate(norm):
        rgba = colormap(float(val))
        colors[i] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    return colors


def draw_molecule_with_importance(smiles, importance, width=500, height=400):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    atom_colors = _importance_to_colors(importance)
    radii = {}
    for i, val in enumerate(importance):
        norm_val = float((val - importance.min()) / (importance.max() - importance.min() + 1e-8))
        radii[i] = float(0.25 + norm_val * 0.25)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.padding = 0.15

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return base64.b64encode(svg.encode("utf-8")).decode("utf-8")


# ── Предсказание ─────────────────────────────────────────────────────────

def _predict_gin(smiles, canonical):
    info = AVAILABLE_MODELS["gin"]
    model = info["model"]

    data = smiles_to_graph_gin(canonical)
    if data is None:
        raise HTTPException(400, f"Не удалось построить граф: {canonical}")

    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    with torch.no_grad():
        logits = model(data.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    importance = compute_atom_importance_gin(model, data, DEVICE)
    if importance.max() > 0:
        importance = importance / importance.max()

    return probs, importance


def _predict_chemberta(smiles, canonical):
    from transformers import AutoTokenizer
    from src.models.chemberta import (
        compute_atom_importance_chemberta,
        CHEMBERTA_MODEL_NAME,
    )

    info = AVAILABLE_MODELS["chemberta"]
    model = info["model"]

    if not hasattr(_predict_chemberta, "_tokenizer"):
        _predict_chemberta._tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    tokenizer = _predict_chemberta._tokenizer

    enc = tokenizer(canonical, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    importance = compute_atom_importance_chemberta(model, canonical, tokenizer, DEVICE)

    return probs, importance


# ── API ──────────────────────────────────────────────────────────────────

app = FastAPI(title="PharmaKinetics API", version="2.0.0")


class PredictRequest(BaseModel):
    smiles: str
    model: str = "gin"


class AtomInfo(BaseModel):
    index: int
    symbol: str
    importance: float


class TaskPrediction(BaseModel):
    task: str
    probability: float
    toxic: bool


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    num_atoms: int
    model_id: str
    model_name: str
    model_params: str
    model_type: str
    tasks: list[TaskPrediction]
    overall_toxicity: float
    atom_importance: list[AtomInfo]
    molecule_svg_base64: str


class ModelInfo(BaseModel):
    id: str
    name: str
    params: str
    type: str
    available: bool


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": list(AVAILABLE_MODELS.keys()),
        "tasks": len(TOX21_TASKS),
        "device": DEVICE,
    }


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    all_models = [
        ModelInfo(id="gin", name="GIN + Virtual Node", params="~2M",
                  type="graph", available="gin" in AVAILABLE_MODELS),
        ModelInfo(id="chemberta", name="ChemBERTa-2 (77M)", params="~80M",
                  type="transformer", available="chemberta" in AVAILABLE_MODELS),
    ]
    return all_models


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    smiles = req.smiles.strip()
    model_id = req.model.strip().lower()

    if not smiles:
        raise HTTPException(400, "Пустая SMILES-строка")

    if model_id not in AVAILABLE_MODELS:
        available = list(AVAILABLE_MODELS.keys())
        raise HTTPException(400, f"Модель '{model_id}' недоступна. Доступные: {available}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, f"Невалидный SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol)

    if model_id == "chemberta":
        probs, importance = _predict_chemberta(smiles, canonical)
    else:
        probs, importance = _predict_gin(smiles, canonical)

    info = AVAILABLE_MODELS[model_id]

    tasks = []
    for i, task_name in enumerate(TOX21_TASKS):
        p = float(probs[i])
        tasks.append(TaskPrediction(
            task=task_name,
            probability=round(p, 4),
            toxic=p > 0.5,
        ))

    atoms = []
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        imp = float(importance[i]) if i < len(importance) else 0.0
        atoms.append(AtomInfo(
            index=i,
            symbol=mol.GetAtomWithIdx(i).GetSymbol(),
            importance=round(imp, 4),
        ))

    svg_b64 = draw_molecule_with_importance(canonical, importance)

    return PredictResponse(
        smiles=smiles,
        canonical_smiles=canonical,
        num_atoms=num_atoms,
        model_id=model_id,
        model_name=info["display_name"],
        model_params=info["params"],
        model_type=info.get("type", "unknown"),
        tasks=tasks,
        overall_toxicity=round(float(probs.max()), 4),
        atom_importance=atoms,
        molecule_svg_base64=svg_b64,
    )


FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
