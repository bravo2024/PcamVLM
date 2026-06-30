"""Inference helpers for PcamVLM.

Wraps the MedGemma 1.5 4B processor + chat template into a small ``predict_*``
API used by both the Streamlit app and the notebook's evaluation cells.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from . import core as C
from .data import prepare_image_for_model


def predict_pcam(model, processor, image: Image.Image,
                 use_logits: bool = True,
                 max_new_tokens: int = 10) -> dict:
    """Predict PCam binary label for a single image.

    When ``use_logits`` is True, compares the logits at the final prompt
    position over the {yes, no} tokens (most reliable).
    Otherwise decodes a few new tokens and parses the first word.
    """
    image = prepare_image_for_model(image, target=224)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": C.PCAM_QUESTION},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Cast only floating-point tensors to model's dtype (bf16)
    dtype = next(model.parameters()).dtype
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype)

    with torch.inference_mode():
        out = {}
        if use_logits:
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            scores = C.yes_no_logit_scores(last_logits, processor.tokenizer)
            pred = 1 if scores["yes"] >= scores["no"] else 0
            out["yes_prob"] = scores["yes"]
            out["no_prob"] = scores["no"]
        else:
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            new_ids = gen[0, inputs["input_ids"].shape[-1]:]
            decoded = processor.decode(new_ids, skip_special_tokens=True)
            pred = C.parse_answer("pcam", decoded)
            out["raw"] = decoded
    out["pred"] = pred
    out["pred_name"] = "yes" if pred == 1 else "no"
    return out


def predict_nct_crc(model, processor, image: Image.Image,
                   max_new_tokens: int = 8) -> dict:
    """Predict NCT-CRC class (0-8) for a single image by free-text decoding."""
    image = prepare_image_for_model(image, target=224)
    question = C.nct_crc_question(C.NCT_CRC_LABEL_CHOICES)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    dtype = next(model.parameters()).dtype
    for k, v in list(inputs.items()):
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
    with torch.inference_mode():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    new_ids = gen[0, inputs["input_ids"].shape[-1]:]
    decoded = processor.decode(new_ids, skip_special_tokens=True).strip()
    pred = C.parse_answer("nct_crc", decoded, C.NCT_CRC_LABEL_CHOICES)
    return {
        "pred": pred,
        "pred_name": C.NCT_CRC_LABEL_NAMES[pred],
        "raw": decoded,
    }


__all__ = ["predict_pcam", "predict_nct_crc"]