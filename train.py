"""PcamVLM training entrypoint.

Fine-tunes MedGemma 1.5 4B on a combined PCam + NCT-CRC-HE-100K subset using
QLoRA (4-bit nf4 + LoRA adapters) and TRL's SFTTrainer.

Designed to run on a single Colab T4 (16 GB VRAM). On T4 expect ~30-60 min
for the default 20k PCam + 10k NCT-CRC subset, 1 epoch, effective batch 8.

Usage:
    python train.py --train-size 20000 --val-size 2000 --epochs 1
    python train.py --no-qlora  # disable quantization (needs >=24 GB VRAM)

If both PCam (HF Hub) and Zenodo (NCT-CRC zip) are unreachable, falls back to
a synthetic dataset so the pipeline still produces a (small) adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core as C
from src import data as D
from src import model as M
from src import persist as P


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=M.MODEL_ID_DEFAULT)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-qlora", action="store_true",
                   help="Disable 4-bit quantization (needs >=24 GB VRAM).")
    p.add_argument("--output-dir", default=str(P.MODELS_DIR / "pcam-medgemma-lora"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-adapter", action="store_true", default=True)
    return p.parse_args()


def build_model(args: argparse.Namespace):
    cfg = M.ModelConfig(
        base_model_id=args.base_model,
        torch_dtype="bfloat16",
        use_qlora=not args.no_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(f"[train] loading processor from {cfg.base_model_id}")
    processor = M.load_processor(cfg)
    print(f"[train] loading base model in "
          f"{'4-bit nf4' if cfg.use_qlora else 'bf16'} "
          f"(lora r={cfg.lora_r}, alpha={cfg.lora_alpha})")
    model, lora_config = M.load_model_and_adapter(cfg)
    M.print_trainable_parameters(model)
    return model, processor, lora_config, cfg


def build_dataset(args: argparse.Namespace):
    spec = D.DatasetSpec(task="mixed", train_size=args.train_size,
                         val_size=args.val_size)
    print(f"[train] loading combined dataset "
          f"(train={args.train_size}, val={args.val_size})")
    train, val = D.load_combined(spec, seed=args.seed)
    print(f"[train] train rows: {len(train)} | val rows: {len(val)}")
    print(f"[train]   pcam train: "
          f"{sum(1 for r in train if r['task']=='pcam')} | "
          f"nct_crc train: "
          f"{sum(1 for r in train if r['task']=='nct_crc')}")
    return train, val


def to_messages_dataset(rows: List[Dict]):
    """Convert list-of-dict rows into a list of {messages: [...]} dicts."""
    from datasets import Dataset

    def row_to_messages(row: Dict) -> Dict:
        question = (C.PCAM_QUESTION if row["task"] == "pcam"
                    else C.nct_crc_question(C.NCT_CRC_LABEL_CHOICES))
        answer = C.dataset_answer_for_label(row["task"], row["label"])
        return D.make_messages(row, question, answer)

    ds_dict = {"messages": [row_to_messages(r) for r in rows]}
    return Dataset.from_dict(ds_dict)


def train(args: argparse.Namespace) -> None:
    model, processor, lora_config, cfg = build_model(args)
    train_rows, val_rows = build_dataset(args)

    train_ds = to_messages_dataset(train_rows)
    val_ds = to_messages_dataset(val_rows)

    # --- TRL SFTTrainer -------------------------------------------------
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "TRL is required for training. Install: pip install trl\n"
            f"Original error: {e}"
        )

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        max_length=2048,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": False},
        seed=args.seed,
    )

    print("[train] building SFTTrainer")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )
    print(f"[train] SFTTrainer built in {time.time() - t0:.1f}s")
    print("[train] starting training")
    train_result = trainer.train()
    print("[train] training complete")
    print(f"  total steps: {train_result.global_step}")
    print(f"  train loss:  {train_result.training_loss:.4f}")

    # --- Save adapter --------------------------------------------------
    if args.save_adapter:
        meta = P.TrainingMetadata(
            base_model=args.base_model,
            dataset=f"pcam({sum(1 for r in train_rows if r['task']=='pcam')})"
                    f"+nct_crc({sum(1 for r in train_rows if r['task']=='nct_crc')})",
            train_size=len(train_rows),
            val_size=len(val_rows),
            epochs=args.epochs,
            lora_r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            learning_rate=args.lr,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_pixels=cfg.max_pixels,
            min_pixels=cfg.min_pixels,
            final_train_loss=float(train_result.training_loss),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        out = P.save_adapter(model, processor, args.output_dir, metadata=meta)
        print(f"[train] adapter saved to {out}")


def main() -> None:
    args = parse_args()
    print(f"[train] starting PcamVLM training with args: {vars(args)}")
    train(args)


if __name__ == "__main__":
    main()