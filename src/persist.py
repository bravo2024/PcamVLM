"""Save / load the trained LoRA adapter + processor + training metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass
class TrainingMetadata:
    base_model: str
    dataset: str
    train_size: int
    val_size: int
    epochs: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_pixels: int
    min_pixels: int
    final_train_loss: float
    timestamp: str


def save_adapter(model, processor, save_dir: str | Path,
                 metadata: Optional[TrainingMetadata] = None) -> Path:
    """Save the LoRA adapter, processor, and (optional) training metadata.

    Tolerates ``model=None`` / ``processor=None`` so smoke tests can verify
    metadata round-tripping without instantiating a real model.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if model is not None:
        model.save_pretrained(save_dir)
    if processor is not None:
        processor.save_pretrained(save_dir)
    if metadata is None:
        metadata = TrainingMetadata(
            base_model="unknown", dataset="unknown",
            train_size=0, val_size=0, epochs=0,
            lora_r=0, lora_alpha=0, lora_dropout=0.0,
            learning_rate=0.0, per_device_batch_size=0,
            gradient_accumulation_steps=0,
            max_pixels=0, min_pixels=0,
            final_train_loss=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    with open(save_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
    return save_dir


def load_metadata(adapter_dir: str | Path) -> dict:
    p = Path(adapter_dir) / "training_metadata.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "MODELS_DIR",
    "TrainingMetadata",
    "save_adapter",
    "load_metadata",
]