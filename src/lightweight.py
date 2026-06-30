"""Lightweight histopathology classifier (CNN-based) for Streamlit Cloud deployment."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from . import core as C
from .data import prepare_image_for_model

ROOT = Path(__file__).resolve().parent.parent
LIGHT_MODELS_DIR = ROOT / "models"
LIGHT_MODEL_PCAM_PATH = LIGHT_MODELS_DIR / "lightweight-pcam.pt"
LIGHT_MODEL_NCT_PATH = LIGHT_MODELS_DIR / "lightweight-nct.pt"

LIGHT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResNetClassifier(nn.Module):
    """ResNet-18 with a replaceable classifier head. Backbone frozen by default."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


@dataclass
class LightweightMetadata:
    task: str
    num_classes: int
    train_samples: int
    val_samples: int
    epochs: int
    final_loss: float
    accuracy: float
    timestamp: str


def save_lightweight_model(model: ResNetClassifier, path: Path,
                           metadata: Optional[LightweightMetadata] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if metadata is not None:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
    return path


def load_lightweight_model(task: str = "pcam",
                           path: Optional[Path] = None) -> Optional[ResNetClassifier]:
    if path is None:
        path = LIGHT_MODEL_PCAM_PATH if task == "pcam" else LIGHT_MODEL_NCT_PATH
    if not path.exists():
        return None
    num_classes = 2 if task == "pcam" else 9
    model = ResNetClassifier(num_classes=num_classes, freeze_backbone=True)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        return None


def load_lightweight_metadata(task: str = "pcam") -> dict:
    path = LIGHT_MODEL_PCAM_PATH if task == "pcam" else LIGHT_MODEL_NCT_PATH
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _to_tensor(image: Image.Image, use_augmentation: bool = False) -> torch.Tensor:
    image = prepare_image_for_model(image, target=224)
    tfm = TRAIN_AUGMENT if use_augmentation else LIGHT_TRANSFORM
    return tfm(image).unsqueeze(0)


@torch.inference_mode()
def predict_lightweight_pcam(model: ResNetClassifier, image: Image.Image) -> dict:
    """Predict PCam binary label using lightweight CNN. Returns pred, pred_name, yes_prob, no_prob."""
    tensor = _to_tensor(image)
    logits = model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    yes_prob = float(probs[1])
    no_prob = float(probs[0])
    pred = 1 if yes_prob >= no_prob else 0
    return {"pred": pred, "pred_name": "yes" if pred == 1 else "no",
            "yes_prob": yes_prob, "no_prob": no_prob}


@torch.inference_mode()
def predict_lightweight_nct(model: ResNetClassifier, image: Image.Image) -> dict:
    """Predict NCT-CRC class (0-8) using lightweight CNN. Returns pred, pred_name, class_probs."""
    tensor = _to_tensor(image)
    logits = model(tensor)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred = int(probs.argmax(dim=-1))
    return {"pred": pred, "pred_name": C.NCT_CRC_LABEL_NAMES[pred],
            "class_probs": [float(p) for p in probs]}


def train_lightweight(
    task: str = "pcam",
    train_size: int = 2000,
    val_size: int = 500,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    save_path=None,
    verbose: bool = True,
    use_real_data: bool = False,
):
    """Train a lightweight ResNet-18 classifier.

    By default uses **synthetic data** (fast, always works offline, <30s for 2000 samples).
    Set ``use_real_data=True`` to attempt downloading real PCam/NCT-CRC (slower, needs network).

    Training runs on CPU in:
    - <30s for 2000 synthetic PCam samples
    - 2-5 min for 5000 synthetic samples
    - 10-30 min for real PCam data (includes HF Hub download)
    """
    import random
    from torch.utils.data import DataLoader, Dataset
    from . import data as D

    torch.manual_seed(seed)
    random.seed(seed)

    num_classes = 2 if task == "pcam" else 9
    model = ResNetClassifier(num_classes=num_classes, freeze_backbone=True)
    model.train()

    if verbose:
        print(f"[lightweight] building {task} dataset (train={train_size}, val={val_size})")
        if use_real_data:
            print("[lightweight] using REAL data (may download from HF Hub / Zenodo)")
        else:
            print("[lightweight] using SYNTHETIC data (fast, no network needed)")

    if use_real_data:
        if task == "pcam":
            try:
                train_rows = D.load_pcam("train", train_size, seed=seed)
                val_rows = D.load_pcam("validation", val_size, seed=seed)
            except Exception:
                if verbose:
                    print("[lightweight] real PCam unavailable, falling back to synthetic")
                train_rows = D.make_synthetic_pcam(train_size, seed=seed)
                val_rows = D.make_synthetic_pcam(val_size, seed=seed + 1)
        else:
            try:
                train_rows = D.load_nct_crc(train_size, seed=seed)
                val_rows = D.load_nct_crc(val_size, seed=seed + 1)
            except Exception:
                if verbose:
                    print("[lightweight] real NCT-CRC unavailable, falling back to synthetic")
                train_rows = D.make_synthetic_nct_crc(train_size, seed=seed)
                val_rows = D.make_synthetic_nct_crc(val_size, seed=seed + 1)
    else:
        if task == "pcam":
            train_rows = D.make_synthetic_pcam(train_size, seed=seed)
            val_rows = D.make_synthetic_pcam(val_size, seed=seed + 1)
        else:
            train_rows = D.make_synthetic_nct_crc(train_size, seed=seed)
            val_rows = D.make_synthetic_nct_crc(val_size, seed=seed + 1)

    class _ImageDataset(Dataset):
        def __init__(self, rows, augment=False):
            self.rows = rows
            self.augment = augment
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            img = prepare_image_for_model(r["image"], target=224)
            tensor = TRAIN_AUGMENT(img) if self.augment else LIGHT_TRANSFORM(img)
            return tensor, r["label"]

    train_ds = _ImageDataset(train_rows, augment=True)
    val_ds = _ImageDataset(val_rows, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = correct / total if total else 0.0
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                preds = outputs.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        if val_acc > best_acc:
            best_acc = val_acc
        if verbose:
            print(f"  epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} | "
                  f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    if verbose:
        print(f"[lightweight] best val accuracy: {best_acc:.4f}")

    model.eval()
    if save_path is None:
        save_path = LIGHT_MODEL_PCAM_PATH if task == "pcam" else LIGHT_MODEL_NCT_PATH
    meta = LightweightMetadata(
        task=task, num_classes=num_classes,
        train_samples=len(train_rows), val_samples=len(val_rows),
        epochs=epochs, final_loss=avg_loss, accuracy=best_acc,
        timestamp=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    )
    save_lightweight_model(model, save_path, metadata=meta)
    if verbose:
        print(f"[lightweight] model saved to {save_path}")
    return model


def lightweight_models_available() -> Dict[str, bool]:
    """Return which lightweight models are present on disk."""
    return {"pcam": LIGHT_MODEL_PCAM_PATH.exists(), "nct_crc": LIGHT_MODEL_NCT_PATH.exists()}


__all__ = [
    "ResNetClassifier", "LIGHT_MODEL_PCAM_PATH", "LIGHT_MODEL_NCT_PATH",
    "lightweight_models_available", "save_lightweight_model",
    "load_lightweight_model", "load_lightweight_metadata",
    "predict_lightweight_pcam", "predict_lightweight_nct",
    "train_lightweight", "LIGHT_TRANSFORM", "TRAIN_AUGMENT",
]
