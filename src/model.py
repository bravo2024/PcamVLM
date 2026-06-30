"""PcamVLM model wiring: MedGemma 1.5 4B + QLoRA.

Targets a single Colab T4 (16 GB VRAM). With QLoRA (4-bit nf4), MedGemma 1.5
4B + ~30M trainable LoRA params fits comfortably and trains in 30-60 min for
20k+10k samples.

Key design decisions
--------------------
- ``torch_dtype=torch.bfloat16`` (required for Gemma3 stability).
- 4-bit nf4 quantization with double-quant for extra VRAM headroom.
- LoRA on attention projections AND MLP gate/up/down linears (Gemma3's
  fused QKV + MLP triad).
- ``task_type=TaskType.CAUSAL_LM`` because MedGemma is a decoder-only model
  (unlike the original PaliGemma2 which was seq2seq).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch

MODEL_ID_DEFAULT = "google/medgemma-1.5-4b-it"

LORA_TARGETS_GEMMA3 = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LORA_TARGETS_SIGLIP = [
    "qkv_proj", "out_proj",
    "fc1", "fc2",
]


@dataclass
class ModelConfig:
    """All knobs needed to build the model + adapter."""

    base_model_id: str = MODEL_ID_DEFAULT
    torch_dtype: str = "bfloat16"
    use_qlora: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: list(LORA_TARGETS_GEMMA3))
    attn_implementation: str = "eager"
    device_map: str = "auto"
    min_pixels: int = 224 * 224
    max_pixels: int = 896 * 896
    image_size: int = 224
    trust_remote_code: bool = False


def build_bnb_config(cfg: ModelConfig):
    """Return a BitsAndBytesConfig for QLoRA, or ``None`` if disabled."""
    if not cfg.use_qlora:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("bitsandbytes is required for QLoRA")
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_processor(cfg: ModelConfig):
    """Load the MedGemma processor and constrain image dimensions."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=cfg.trust_remote_code,
    )
    # MedGemma's processor respects ``min_pixels`` / ``max_pixels`` via the
    # image_processor config. Setting these to a small range keeps training
    # VRAM low on T4.
    try:
        ip = processor.image_processor
        ip.min_pixels = cfg.min_pixels
        ip.max_pixels = cfg.max_pixels
        ip.size = {"height": cfg.image_size, "width": cfg.image_size}
    except AttributeError:
        pass
    return processor


def load_model_and_adapter(cfg: ModelConfig):
    """Load base MedGemma + attach LoRA adapter. Returns (model, peft_config)."""
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText  # type: ignore

    bnb_config = build_bnb_config(cfg)
    torch_dtype = getattr(torch, cfg.torch_dtype)

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
    )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model_id, **model_kwargs
    )

    # Prepare for k-bit training (sets gradient checkpointing + casts LN to fp32)
    if cfg.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, lora_config


def load_trained(base_model_id: str, adapter_path: str, torch_dtype: str = "bfloat16"):
    """Reload a fine-tuned adapter on top of the base model.

    Used at inference time by ``app.py`` and the notebook's reload cell.
    """
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText  # type: ignore
    from peft import PeftModel

    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=getattr(torch, torch_dtype),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def print_trainable_parameters(model) -> None:
    """Print trainable parameter counts (PEFT convention)."""
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"[model] trainable: {trainable:,} / total: {total:,} ({pct:.3f}%)")


def build_chat_messages(image, question: str) -> list:
    """Build a Gemma3 chat-template messages list (image + text)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


__all__ = [
    "MODEL_ID_DEFAULT",
    "LORA_TARGETS_GEMMA3",
    "LORA_TARGETS_SIGLIP",
    "ModelConfig",
    "build_bnb_config",
    "load_processor",
    "load_model_and_adapter",
    "load_trained",
    "print_trainable_parameters",
    "build_chat_messages",
]