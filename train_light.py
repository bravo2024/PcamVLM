"""Lightweight training entrypoint — trains a ResNet-18 on synthetic or real data.

**Default (synthetic data, fast, no network):**
    python train_light.py                         # ~30s for 2000 samples
    python train_light.py --task nct_crc --epochs 3

**Real data (slower, needs HF Hub / Zenodo download):**
    python train_light.py --real-data              # uses real PCam from HuggingFace Hub
    python train_light.py --real-data --task nct_crc

**Fine-tune entire backbone (better accuracy, slower):**
    python train_light.py --no-freeze-backbone

After training the model is saved to ``models/lightweight-pcam.pt``
(or ``models/lightweight-nct.pt``). The Streamlit app picks it up automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import lightweight as L  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight histopathology classifier")
    p.add_argument("--task", default="pcam", choices=["pcam", "nct_crc"],
                   help="Dataset task (default: pcam)")
    p.add_argument("--train-size", type=int, default=2000,
                   help="Number of training samples (default: 2000)")
    p.add_argument("--val-size", type=int, default=500,
                   help="Number of validation samples (default: 500)")
    p.add_argument("--epochs", type=int, default=5,
                   help="Training epochs (default: 5)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no-freeze-backbone", action="store_true",
                   help="Fine-tune entire ResNet-18 instead of just the classifier head")
    p.add_argument("--real-data", action="store_true",
                   help="Use real PCam/NCT-CRC data instead of synthetic (slower, needs network)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[train_light] task={args.task}, train={args.train_size}, "
          f"val={args.val_size}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"[train_light] data: {'REAL' if args.real_data else 'SYNTHETIC'}")
    if args.no_freeze_backbone:
        print("[train_light] fine-tuning entire backbone (slower)")

    if args.no_freeze_backbone:
        orig_init = L.ResNetClassifier.__init__
        def _patched_init(self, num_classes=2, freeze_backbone=True):
            orig_init(self, num_classes=num_classes, freeze_backbone=False)
        L.ResNetClassifier.__init__ = _patched_init

    L.train_lightweight(
        task=args.task,
        train_size=args.train_size,
        val_size=args.val_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        use_real_data=args.real_data,
    )
    print(f"[train_light] done. Model saved to models/lightweight-{args.task}.pt")
    print("[train_light] The Streamlit app will now use this trained model automatically.")


if __name__ == "__main__":
    main()
