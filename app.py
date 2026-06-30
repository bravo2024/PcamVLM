"""PcamVLM — Interactive dashboard for histopathology patch classification.

Two model modes
---------------
1. **Lightweight CNN (default)** — ResNet-18 (ImageNet-pretrained). Runs on CPU
   in <200 MB RAM, <1 s per image. Perfect for Streamlit Cloud free tier.
2. **MedGemma VLM (GPU)** — MedGemma 1.5 4B with optional LoRA adapter. Needs
   a GPU (T4+). Best accuracy but heavy.

Three data sources
------------------
1. **Synthetic** — no downloads required. Generates fake H&E patches on the fly.
2. **PCam (real)** — loads ``1aurent/PatchCamelyon`` from HuggingFace Hub.
3. **NCT-CRC-HE-100K (real)** — auto-downloads from Zenodo on first run.

When using lightweight mode the app deploys on Streamlit Cloud free tier with
all features working. The VLM mode requires a GPU host (Colab, HF Spaces with
T4, or local).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core as C  # noqa: E402
from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PcamVLM — Histopathology VLM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --pv-primary: #1e3a5f;
        --pv-accent: #2563eb;
        --pv-success: #059669;
        --pv-danger: #dc2626;
        --pv-surface: #f8fafc;
        --pv-border: #e2e8f0;
    }
    .stApp { font-family: 'Inter', sans-serif; }
    .pv-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #1e40af 100%);
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }
    .pv-hero h1 { margin: 0 0 0.3rem 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
    .pv-hero p { margin: 0; opacity: 0.9; font-size: 1.0rem; font-weight: 400; }
    .pv-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-top: 0.5rem;
    }
    .pv-card {
        background: white;
        border: 1px solid var(--pv-border);
        border-radius: 0.75rem;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .pv-pred-yes { color: var(--pv-danger); font-weight: 700; }
    .pv-pred-no { color: var(--pv-success); font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    model_mode = st.selectbox(
        "Model mode",
        options=["lightweight", "medgemma_vlm"],
        index=0,
        format_func=lambda m: "Lightweight CNN (CPU, default)" if m == "lightweight"
                              else "MedGemma VLM (GPU, needs T4+)",
        help="Lightweight CNN works on any machine (Streamlit Cloud, local). "
             "MedGemma VLM needs a GPU and 5+ GB free RAM.",
    )
    task = st.selectbox(
        "Task",
        options=["pcam", "nct_crc"],
        format_func=lambda t: "PCam (binary tumor/normal)" if t == "pcam"
                              else "NCT-CRC-HE-100K (9 tissue types)",
        index=0,
    )
    source = st.selectbox(
        "Data source",
        options=["synthetic", "pcam (HF Hub)", "nct_crc (Zenodo)"],
        index=0,
        help="Synthetic data lets the dashboard demo without downloads.",
    )
    n_samples = st.slider("Samples to show", min_value=4, max_value=32,
                          value=12, step=4)

    # VLM-specific settings (only shown when VLM mode is selected)
    if model_mode == "medgemma_vlm":
        base_model_id = st.text_input("Base model ID", value="google/medgemma-1.5-4b-it")
        adapter_dir = st.text_input("Adapter directory",
                                    value=str(ROOT / "models" / "pcam-medgemma-lora"))
        use_logits = st.checkbox("Use logit-level yes/no (PCam only)",
                                 value=True, help="Faster + more reliable than text decoding.")
    else:
        base_model_id = "google/medgemma-1.5-4b-it"
        adapter_dir = str(ROOT / "models" / "pcam-medgemma-lora")
        use_logits = True

    run_eval = st.button("Run held-out evaluation")
    st.markdown("---")
    if model_mode == "lightweight":
        st.markdown(
            "**ResNet-18** lightweight classifier. "
            "[Train it](train_light.py) with: `python train_light.py`"
        )
    else:
        st.markdown(
            "Built on **MedGemma 1.5 4B** (Google Health, Jan 2026) "
            "fine-tuned with QLoRA. Datasets: PCam (CC0-1.0) and NCT-CRC-HE-100K "
            "(CC BY 4.0)."
        )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="pv-hero">
      <h1>PcamVLM — Histopathology VLM</h1>
      <p>QLoRA fine-tune of MedGemma 1.5 4B for H&E patch classification.
         Trained on PatchCamelyon + NCT-CRC-HE-100K.</p>
      <span class="pv-badge">MedGemma 1.5 4B</span>
      <span class="pv-badge">QLoRA</span>
      <span class="pv-badge">bf16</span>
      <span class="pv-badge">PCam + NCT-CRC-HE-100K</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load samples
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_samples(task_name: str, source_label: str, n: int) -> List[dict]:
    """Build a list of {image, label, task} rows for display."""
    src_key = source_label.split(" ")[0]
    if src_key == "synthetic":
        if task_name == "pcam":
            return D.make_synthetic_pcam(n, seed=42)
        return D.make_synthetic_nct_crc(n, seed=42)
    if src_key == "pcam":
        return D.load_pcam("validation", n, seed=42)
    if src_key == "nct_crc":
        return D.load_nct_crc(n, seed=42)
    return D.make_synthetic_pcam(n, seed=42)


samples = load_samples(task, source, n_samples)
if not samples:
    st.error("Could not load any samples. Try the synthetic data source.")
    st.stop()


# ---------------------------------------------------------------------------
# Model loading (lazy, cached — two modes)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_lightweight_model():
    """Load ResNet-18 lightweight model. Always works (falls back to untrained)."""
    from src import lightweight as LW
    model = LW.load_lightweight_model(task)
    ok = model is not None
    if not ok:
        num_classes = 2 if task == "pcam" else 9
        model = LW.ResNetClassifier(num_classes=num_classes, freeze_backbone=True)
        model.eval()
    return model, ok


@st.cache_resource(show_spinner=False)
def _load_vlm_model(base_id: str, adapter: str):
    """Load MedGemma + LoRA adapter. Needs GPU and 5+ GB RAM."""
    try:
        from src import model as M
        from src.model import load_trained
        from transformers import AutoProcessor
    except Exception as e:
        return None, None, False, str(e)
    adapter_path = Path(adapter)
    if not adapter_path.exists():
        return None, None, False, f"Adapter not found at {adapter_path}"
    try:
        processor = AutoProcessor.from_pretrained(adapter_path)
        model = load_trained(base_id, str(adapter_path), torch_dtype="bfloat16")
        return model, processor, True, "ok"
    except Exception as e:
        return None, None, False, str(e)


MODEL_OK = False
_is_lightweight = model_mode == "lightweight"

if _is_lightweight:
    _lw_model, MODEL_OK = _load_lightweight_model()
    _vlm_model = None
    _processor = None
else:
    _vlm_model, _processor, MODEL_OK, _load_err = _load_vlm_model(
        base_model_id, adapter_dir)
    _lw_model = None
    if not MODEL_OK:
        st.info(_load_err)


# ---------------------------------------------------------------------------
# Display sample grid + per-sample predictions
# ---------------------------------------------------------------------------

st.markdown(f"### Sample patches ({len(samples)})")

n_cols = 4
rows = [samples[i:i + n_cols] for i in range(0, len(samples), n_cols)]

for chunk in rows:
    cols = st.columns(n_cols, gap="small")
    for j, sample in enumerate(chunk):
        with cols[j]:
            label_name = (
                C.PCAM_LABEL_NAMES.get(sample["label"], str(sample["label"]))
                if sample["task"] == "pcam"
                else C.NCT_CRC_LABEL_NAMES.get(sample["label"], str(sample["label"]))
            )
            st.image(sample["image"], caption=f"GT: {label_name}",
                     use_container_width=True)
            if MODEL_OK:
                try:
                    if _is_lightweight:
                        out = _predict_lightweight(
                            _lw_model, sample["image"], sample["task"])
                    elif sample["task"] == "pcam":
                        out = _predict_pcam_vlm(
                            _vlm_model, _processor, sample["image"], use_logits)
                    else:
                        out = _predict_nct_crc_vlm(
                            _vlm_model, _processor, sample["image"])

                    if sample["task"] == "pcam":
                        pred_cls = "pv-pred-yes" if out["pred"] == 1 else "pv-pred-no"
                        st.markdown(
                            f"<span class='{pred_cls}'>{out['pred_name']}</span>",
                            unsafe_allow_html=True,
                        )
                        prob_key = "yes_prob"
                        if prob_key in out:
                            st.progress(float(out[prob_key]),
                                        text=f"P(yes)={out[prob_key]:.2f}")
                    else:
                        st.markdown(f"**{out['pred_name']}**")
                        if "raw" in out:
                            st.caption(f"raw: {out['raw'][:20]!r}")
                except Exception as e:
                    st.warning(f"inference failed: {e}")


# ---------------------------------------------------------------------------
# Held-out evaluation (only meaningful when model is loaded)
# ---------------------------------------------------------------------------

if run_eval:
    if not MODEL_OK:
        st.error("Cannot evaluate: model adapter is not loaded.")
    else:
        with st.spinner("Running held-out evaluation..."):
            eval_rows = load_samples(
                task, source, min(n_samples * 5, 60)
            )
            trues, preds = [], []
            for r in eval_rows:
                if _is_lightweight:
                    out = _predict_lightweight(_lw_model, r["image"], r["task"])
                elif r["task"] == "pcam":
                    out = _predict_pcam_vlm(_vlm_model, _processor, r["image"], use_logits)
                else:
                    out = _predict_nct_crc_vlm(_vlm_model, _processor, r["image"])
                trues.append(int(r["label"]))
                preds.append(int(out["pred"]))
            if task == "pcam":
                report = E.evaluate_binary(trues, preds)
            else:
                report = E.evaluate_multiclass(
                    trues, preds, C.NCT_CRC_LABEL_CHOICES
                )
        st.success(f"Accuracy: {report.accuracy:.4f} on {report.n} samples")
        cm = pd.DataFrame(report.confusion,
                          index=report.class_names,
                          columns=report.class_names)
        st.markdown("#### Confusion matrix")
        st.dataframe(cm, use_container_width=True)
        if report.per_class:
            rows = []
            for k, v in report.per_class.items():
                rows.append({
                    "class": v.get("name", str(k)),
                    "support": v.get("support", 0),
                    "precision": round(v.get("precision", 0.0), 4),
                    "recall": round(v.get("recall", 0.0), 4),
                    "f1": round(v.get("f1", 0.0), 4),
                })
            st.markdown("#### Per-class metrics")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Footer / metadata
# ---------------------------------------------------------------------------

st.markdown("---")
if _is_lightweight:
    st.markdown(
        "**Lightweight mode** — ResNet-18 classifier. "
        "Train with: `python train_light.py`. "
        "Switch to MedGemma VLM mode in the sidebar for higher accuracy (GPU required)."
    )
else:
    st.markdown(
        "**VLM mode** — MedGemma 1.5 4B with QLoRA. "
        "License: [Health AI Developer Foundations terms]"
        "(https://developers.google.com/health-ai-developer-foundations/terms). "
        "PCam: CC0-1.0. NCT-CRC-HE-100K: CC BY 4.0."
    )


# ---------------------------------------------------------------------------
# Prediction wrappers (lazy import to keep lightweight mode fast)
# ---------------------------------------------------------------------------

def _predict_lightweight(model, image, task_name: str):
    from src.lightweight import predict_lightweight_pcam, predict_lightweight_nct
    if task_name == "pcam":
        return predict_lightweight_pcam(model, image)
    return predict_lightweight_nct(model, image)


def _predict_pcam_vlm(model, processor, image, use_logits_flag: bool):
    from src.infer import predict_pcam  # type: ignore
    return predict_pcam(model, processor, image, use_logits=use_logits_flag)


def _predict_nct_crc_vlm(model, processor, image):
    from src.infer import predict_nct_crc  # type: ignore
    return predict_nct_crc(model, processor, image)