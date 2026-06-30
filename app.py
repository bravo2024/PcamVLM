"""PcamVLM — Interactive dashboard for histopathology patch classification.

Two model modes
---------------
1. **Lightweight CNN (default)** — ResNet-18 (ImageNet-pretrained). Runs on CPU
   in <200 MB RAM, <1 s per image. Perfect for Streamlit Cloud free tier.
2. **MedGemma VLM (GPU)** — MedGemma 1.5 4B with optional LoRA adapter. Needs
   a GPU (T4+). Best accuracy but heavy.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import core as C
from src import data as D
from src import evaluate as E


st.set_page_config(
    page_title="PcamVLM — Histopathology VLM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --pv-primary: #1e3a5f; --pv-accent: #2563eb; --pv-success: #059669; --pv-danger: #dc2626; }
    .stApp { font-family: Inter, sans-serif; }
    .pv-hero { padding:1.4rem 1.8rem; border-radius:.75rem; background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 40%,#1e40af 100%); color:#fff; margin-bottom:1.2rem; }
    .pv-hero h1 { margin:0 0 .3rem; font-size:1.8rem; font-weight:700; }
    .pv-hero .pv-sub { margin-top:.6rem; font-size:.85rem; opacity:.75; }
    .pv-badge { display:inline-block; background:rgba(255,255,255,.18); padding:.15rem .55rem; border-radius:9999px; font-size:.78rem; margin-right:.4rem; margin-top:.5rem; }
    .pv-card { background:#fff; border:1px solid #e2e8f0; border-radius:.75rem; padding:1rem 1.2rem; margin-bottom:.75rem; }
    .pv-pred-yes { color: #dc2626; font-weight: 700; }
    .pv-pred-no { color: #059669; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

is_vlm_adapter = (ROOT / "models" / "pcam-medgemma-lora").exists()
status = "✅ VLM adapter found" if is_vlm_adapter else "⚡ Lightweight mode active"
hero = f"""<div class="pv-hero">
  <h1>🔬 PcamVLM — Histopathology VLM</h1>
  <p>QLoRA fine-tune of MedGemma 1.5 4B for H&E patch classification.
     Trained on PatchCamelyon + NCT-CRC-HE-100K.</p>
  <span class="pv-badge">MedGemma 1.5 4B</span>
  <span class="pv-badge">QLoRA</span>
  <span class="pv-badge">bf16</span>
  <span class="pv-badge">PCam + NCT-CRC-HE-100K</span>
  <div class="pv-sub">{status} · Synthetic fallback enabled</div>
</div>"""
st.markdown(hero, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ⚙ Settings")
    model_mode = st.selectbox(
        "Model mode",
        options=["lightweight", "medgemma_vlm"],
        index=0,
        format_func=lambda m: "Lightweight CNN (CPU)" if m == "lightweight" else "MedGemma VLM (GPU)",
    )
    task = st.selectbox(
        "Task",
        options=["pcam", "nct_crc"],
        index=0,
        format_func=lambda t: "PCam (binary)" if t == "pcam" else "NCT-CRC (9-class)",
    )
    source = st.selectbox(
        "Data source",
        options=["synthetic", "pcam (HF Hub)", "nct_crc (Zenodo)"],
        index=0,
    )
    n_samples = st.slider("Samples", 4, 32, 12, 4)

    if model_mode == "medgemma_vlm":
        base_id = st.text_input("Base model", value="google/medgemma-1.5-4b-it")
        adapter_dir = st.text_input("Adapter dir", value=str(ROOT / "models" / "pcam-medgemma-lora"))
        use_logits = st.checkbox("Logit-level yes/no", value=True)
    else:
        base_id = "google/medgemma-1.5-4b-it"
        adapter_dir = str(ROOT / "models" / "pcam-medgemma-lora")
        use_logits = True

    st.markdown("---")
    if model_mode == "lightweight":
        st.markdown("**ResNet-18** • Train: `python train_light.py`")
    else:
        st.markdown("**MedGemma 1.5 4B** + QLoRA")


@st.cache_data(show_spinner=False)
def load_samples(task_name, source_label, n):
    sk = source_label.split(" ")[0]
    if sk == "synthetic":
        return D.make_synthetic_pcam(n, 42) if task_name == "pcam" else D.make_synthetic_nct_crc(n, 42)
    if sk == "pcam":
        return D.load_pcam("validation", n, 42)
    if sk == "nct_crc":
        return D.load_nct_crc(n, 42)
    return D.make_synthetic_pcam(n, 42)


@st.cache_resource(show_spinner=False)
def _load_lightweight_model(task_name):
    from src import lightweight as LW
    m = LW.load_lightweight_model(task_name)
    ok = m is not None
    if not ok:
        nc = 2 if task_name == "pcam" else 9
        m = LW.ResNetClassifier(num_classes=nc, freeze_backbone=True)
        m.eval()
    return m, ok


@st.cache_resource(show_spinner=False)
def _load_vlm_model(base_id, adapter):
    try:
        from src.model import load_trained
        from transformers import AutoProcessor
    except Exception as e:
        return None, None, False, str(e)
    ap = Path(adapter)
    if not ap.exists():
        return None, None, False, f"Adapter not found at {ap}"
    try:
        p = AutoProcessor.from_pretrained(str(ap))
        m = load_trained(base_id, str(ap), torch_dtype="bfloat16")
        return m, p, True, "ok"
    except Exception as e:
        return None, None, False, str(e)


def _predict_lightweight(model, image, tn):
    from src.lightweight import predict_lightweight_pcam as p, predict_lightweight_nct as n
    return p(model, image) if tn == "pcam" else n(model, image)


def _predict_pcam_vlm(model, p, img, f):
    from src.infer import predict_pcam as x
    return x(model, p, img, use_logits=f)


def _predict_nct_crc_vlm(model, p, img):
    from src.infer import predict_nct_crc as x
    return x(model, p, img)


MODEL_OK = False
_is_light = model_mode == "lightweight"
if _is_light:
    _lw_model, MODEL_OK = _load_lightweight_model(task)
    _vlm_model = None
    _processor = None
else:
    _vlm_model, _processor, MODEL_OK, err = _load_vlm_model(base_id, adapter_dir)
    _lw_model = None
    if not MODEL_OK:
        st.sidebar.info(str(err))


def get_label_name(s):
    if s["task"] == "pcam":
        return C.PCAM_LABEL_NAMES.get(s["label"], str(s["label"]))
    return C.NCT_CRC_LABEL_NAMES.get(s["label"], str(s["label"]))


def get_class_names(tn):
    return ["no", "yes"] if tn == "pcam" else C.NCT_CRC_LABEL_CHOICES


def safe_predict(s):
    if not MODEL_OK:
        return None
    try:
        if _is_light:
            return _predict_lightweight(_lw_model, s["image"], s["task"])
        if s["task"] == "pcam":
            return _predict_pcam_vlm(_vlm_model, _processor, s["image"], use_logits)
        return _predict_nct_crc_vlm(_vlm_model, _processor, s["image"])
    except Exception as e:
        return {"error": str(e)}



# ============================================================
# TABS
# ============================================================
t_explore, t_predict, t_evaluate, t_about = st.tabs(
    ["🔬 Explore", "🤖 Predict", "📊 Evaluate", "📖 About"]
)


# ============================================================
# TAB 1 — Explore
# ============================================================
with t_explore:
    st.markdown("### 📊 Dataset Explorer")
    st.markdown("Browse and understand the data. Three sources available: synthetic (offline), PCam (HF Hub), and NCT-CRC (Zenodo).")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("#### Dataset Info")
        if source.startswith("synthetic"):
            st.markdown(
                f"""<div class="pv-card">
                <b>Synthetic H&E</b> — Deterministic procedural generation.
                Tumor patches are denser and darker.<br><br>
                <b>Size:</b> {D.SYNTH_SIZE}×{D.SYNTH_SIZE} px<br>
                <b>Classes:</b> {"2 (no / yes)" if task == "pcam" else "9 (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM)"}<br>
                <b>License:</b> No restrictions (procedural)</div>""",
                unsafe_allow_html=True,
            )
        elif source.startswith("pcam"):
            st.markdown(
                """<div class="pv-card">
                <b>PatchCamelyon (PCam)</b> — H&E patches from CAMELYON16 lymph nodes.<br><br>
                <b>Size:</b> 96×96 px<br>
                <b>Patches:</b> 327,680<br>
                <b>License:</b> CC0-1.0 (public domain)<br>
                <b>Source:</b> HuggingFace Hub</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="pv-card">
                <b>NCT-CRC-HE-100K</b> — Colorectal cancer histology.<br><br>
                <b>Size:</b> 224×224 px<br>
                <b>Patches:</b> 100,000<br>
                <b>License:</b> CC BY 4.0<br>
                <b>Citation:</b> Kather et al. 2018</div>""",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("#### Class Distribution")
        ds = load_samples(task, source, min(n_samples * 4, 64))
        if ds:
            labels = [s["label"] for s in ds]
            cn = get_class_names(task)
            dist = pd.Series(labels).value_counts().sort_index()
            st.dataframe(
                pd.DataFrame({
                    "Class": [cn[i] if i < len(cn) else str(i) for i in dist.index],
                    "Count": dist.values,
                    "Pct": [f"{v / len(labels):.0%}" for v in dist.values],
                }),
                use_container_width=True,
                hide_index=True,
            )
            if task == "pcam":
                yes = sum(1 for l in labels if l == 1) / len(labels)
                st.markdown(f"**Normal (no):** {1 - yes:.0%}")
                st.progress(1 - yes)
                st.markdown(f"**Tumor (yes):** {yes:.0%}")
                st.progress(yes)

    st.markdown("---")
    st.markdown("#### Sample Gallery")
    gs = load_samples(task, source, n_samples)
    if gs:
        for i in range(0, len(gs), 4):
            cols = st.columns(4, gap="small")
            for j, s in enumerate(gs[i:i + 4]):
                with cols[j]:
                    st.image(s["image"], caption=f"GT: {get_label_name(s)}", use_container_width=True)

    st.markdown(
        """<div class="pv-card">
        <b>⚠️ Note:</b> PCam patches from the same WSI are correlated. Per-patch metrics are optimistic. For clinical use, evaluate per-slide.</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 — Predict
# ============================================================
with t_predict:
    st.markdown("### 🤖 Live Inference")

    if model_mode == "lightweight":
        if MODEL_OK:
            st.success("✅ Lightweight CNN — ResNet-18 loaded and ready.")
        else:
            st.warning("⚠️ No trained weights found. Using untrained model (random predictions).")
    else:
        if MODEL_OK:
            st.success("✅ MedGemma VLM — Model with LoRA adapter loaded.")
        else:
            st.error(f"❌ MedGemma VLM not loaded. {err}")

    col_cfg, col_pred = st.columns([1, 3])
    with col_cfg:
        st.markdown("#### Settings")
        st.caption(f"**Task:** {task}")
        st.caption(f"**Data:** {source}")
        st.caption(f"**Samples:** {n_samples}")
        if task == "pcam" and not _is_light:
            st.caption(f"**Logit scoring:** {use_logits}")
        if MODEL_OK:
            st.info("Predictions shown below each image.")
        else:
            st.warning("Load a model to see predictions.")

    with col_pred:
        st.markdown("#### Predictions")
        st.caption("Ground truth (GT) and model prediction (Pred) with confidence.")

    ps = load_samples(task, source, n_samples)
    if ps:
        for i in range(0, len(ps), 4):
            cols = st.columns(4, gap="small")
            for j, s in enumerate(ps[i:i + 4]):
                with cols[j]:
                    st.image(s["image"], caption=f"GT: {get_label_name(s)}", use_container_width=True)
                    if MODEL_OK:
                        out = safe_predict(s)
                        if out is None:
                            st.info("No prediction")
                        elif "error" in out:
                            st.warning(f"Error: {out['error']}")
                        else:
                            correct = int(out["pred"]) == int(s["label"])
                            color = "#059669" if correct else "#dc2626"
                            if s["task"] == "pcam":
                                cls = "pv-pred-yes" if out["pred"] == 1 else "pv-pred-no"
                                st.markdown(f"Pred: <span class='{cls}'>{out['pred_name']}</span>", unsafe_allow_html=True)
                                if "yes_prob" in out:
                                    prob = float(out["yes_prob"])
                                    st.progress(prob, text=f"P(yes)={prob:.2f}")
                                st.caption(f"<span style='color:{color}'>{'✓ Correct' if correct else '✗ Mismatch'}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Pred: {out['pred_name']}**")
                                st.caption(f"<span style='color:{color}'>{'✓ Correct' if correct else '✗ Mismatch'}</span>", unsafe_allow_html=True)
                    else:
                        st.info("No model loaded")

    if not MODEL_OK:
        st.info("💡 Tip: Train a model with `python train_light.py` or load a VLM adapter.")


# ============================================================
# TAB 3 — Evaluate
# ============================================================
with t_evaluate:
    st.markdown("### 📊 Evaluation Dashboard")
    st.markdown("Run held-out evaluation to measure performance. For PCam we report accuracy, sensitivity, specificity, precision, and F1. For NCT-CRC we report macro-F1 and per-class metrics.")

    eval_n = st.slider("Evaluation sample size", 10, 100, min(n_samples * 5, 60), 5)
    run_eval = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    if run_eval:
        if not MODEL_OK:
            st.error("Cannot evaluate: model is not loaded.")
        else:
            with st.spinner(f"Running evaluation on {eval_n} samples..."):
                er = load_samples(task, source, eval_n)
                trues, preds = [], []
                for r in er:
                    if _is_light:
                        out = _predict_lightweight(_lw_model, r["image"], r["task"])
                    elif r["task"] == "pcam":
                        out = _predict_pcam_vlm(_vlm_model, _processor, r["image"], use_logits)
                    else:
                        out = _predict_nct_crc_vlm(_vlm_model, _processor, r["image"])
                    trues.append(int(r["label"]))
                    preds.append(int(out["pred"]))

                report = E.evaluate_binary(trues, preds) if task == "pcam" else E.evaluate_multiclass(trues, preds, C.NCT_CRC_LABEL_CHOICES)

            st.success(f"✅ Accuracy: {report.accuracy:.2%} on {report.n} samples")

            m1, m2, m3, m4 = st.columns(4)
            if task == "pcam":
                m1.metric("Sensitivity", f"{report.per_class[1]['recall']:.1%}", help="TP / (TP + FN)")
                m2.metric("Specificity", f"{report.per_class[0]['recall']:.1%}", help="TN / (TN + FP)")
                m3.metric("Precision", f"{report.per_class[1]['precision']:.1%}", help="TP / (TP + FP)")
                m4.metric("F1 Score", f"{report.per_class[1]['f1']:.1%}")
                st.info("📌 Clinical note: For cancer screening, sensitivity is the priority — missing a tumor is worse than a false alarm.")
            else:
                f1s = [v["f1"] for v in report.per_class.values()]
                m1.metric("Macro F1", f"{np.mean(f1s):.1%}")
                m2.metric("Samples", str(report.n))
                m3.metric("Classes", str(len(report.class_names)))
                m4.metric("Accuracy", f"{report.accuracy:.1%}")

            st.markdown("#### Confusion Matrix")
            cm = pd.DataFrame(report.confusion, index=[f"A: {n}" for n in report.class_names], columns=[f"P: {n}" for n in report.class_names])
            st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.caption("Rows = Actual, Columns = Predicted. Diagonal = correct.")

            if report.per_class:
                st.markdown("#### Per-Class Metrics")
                rows = []
                for k, v in report.per_class.items():
                    rows.append({
                        "Class": v.get("name", str(k)),
                        "Support": v.get("support", 0),
                        "Precision": round(v.get("precision", 0.0), 4),
                        "Recall": round(v.get("recall", 0.0), 4),
                        "F1": round(v.get("f1", 0.0), 4),
                    })
                st.dataframe(pd.DataFrame(rows).style.background_gradient(subset=["Precision", "Recall", "F1"], cmap="Blues"), use_container_width=True, hide_index=True)

            if task == "pcam":
                sens = report.per_class[1]["recall"]
                spec = report.per_class[0]["recall"]
                st.markdown(f"**Interpretation:** Sensitivity = **{sens:.1%}** (tumor detection). Specificity = **{spec:.1%}** (normal rejection).")
    else:
        st.info("👆 Click 'Run Evaluation' to see metrics.")
        with st.expander("📖 Understanding the metrics"):
            st.markdown("""
            | Metric | What it measures | Clinical meaning |
            |---|---|---|
            | **Sensitivity** | TP / (TP + FN) | Tumours correctly identified |
            | **Specificity** | TN / (TN + FP) | Normal patches correctly identified |
            | **Precision** | TP / (TP + FP) | Trustworthiness of 'tumour' predictions |
            | **F1 Score** | Harmonic mean of P & R | Balanced accuracy measure |
            | **Macro F1** | Average F1 across classes | Overall NCT-CRC performance |
            """)


# ============================================================
# TAB 4 — About
# ============================================================
with t_about:
    st.markdown("### 📖 About PcamVLM")

    st.markdown("#### 🧠 Model Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Lightweight CNN**")
        st.markdown(
            """<div class="pv-card">
            <b>Backbone:</b> ResNet-18 (ImageNet)<br>
            <b>Params:</b> ~11M total (~0.5M trainable)<br>
            <b>Size:</b> ~45 MB<br>
            <b>Device:</b> CPU<br>
            <b>Inference:</b> &lt;1s per image<br>
            <b>Train:</b> <code>python train_light.py</code></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**MedGemma 1.5 4B VLM**")
        st.markdown(
            """<div class="pv-card">
            <b>Base:</b> <code>google/medgemma-1.5-4b-it</code><br>
            <b>Architecture:</b> SigLIP + Gemma3 decoder<br>
            <b>Params:</b> ~4B total (~30M trainable LoRA)<br>
            <b>Quant:</b> 4-bit nf4 + double quant<br>
            <b>Compute:</b> bf16 + gradient checkpointing<br>
            <b>Device:</b> GPU T4+</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 🔧 QLoRA Configuration")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("LoRA Rank (r)", "16")
        st.metric("LoRA Alpha", "32")
        st.metric("LoRA Dropout", "0.05")
    with q2:
        st.metric("Quantisation", "4-bit nf4")
        st.metric("Double Quant", "Enabled")
        st.metric("Compute dtype", "bfloat16")
    with q3:
        st.metric("Target Modules", "7")
        st.metric("Effective Batch", "8")
        st.metric("Optimiser", "adamw_torch_fused")
    st.markdown(
        """<div class="pv-card">
        <b>Why QLoRA?</b> Combines 4-bit NormalFloat quantisation with Low-Rank Adapters, enabling fine-tuning of a 4B model on a single T4 (16 GB VRAM). Only ~0.75% of parameters are trainable.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 📋 Training Records")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Lightweight Model**")
        lw_meta = ROOT / "models" / f"lightweight-{task}.json"
        if lw_meta.exists():
            meta = json.loads(lw_meta.read_text())
            for k, v in [("Task", meta.get("task")), ("Train Samples", meta.get("train_samples")), ("Val Samples", meta.get("val_samples")), ("Epochs", meta.get("epochs")), ("Final Loss", f"{meta.get('final_loss',0):.4f}"), ("Val Accuracy", f"{meta.get('accuracy',0):.2%}"), ("Timestamp", meta.get("timestamp"))]:
                st.markdown(f"**{k}:** {v}")
        else:
            st.info("No lightweight training records found.")
    with t2:
        st.markdown("**VLM Adapter**")
        vlm_meta = ROOT / "models" / "pcam-medgemma-lora" / "training_metadata.json"
        if vlm_meta.exists():
            meta = json.loads(vlm_meta.read_text())
            for k, v in [("Base Model", meta.get("base_model","").split("/")[-1]), ("Dataset", meta.get("dataset")), ("Train Size", meta.get("train_size")), ("Val Size", meta.get("val_size")), ("Epochs", meta.get("epochs")), ("LR", f"{meta.get('learning_rate',0):.0e}"), ("Final Loss", f"{meta.get('final_train_loss',0):.4f}"), ("Timestamp", meta.get("timestamp"))]:
                st.markdown(f"**{k}:** {v}")
        else:
            st.info("No VLM adapter found.")

    st.markdown("#### 🎯 Why MedGemma 1.5 4B?")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("- Medical-specialised: pre-trained on CAMELYON16/17, TCGA, histopathology, dermatology, ophthalmology, chest X-rays")
        st.markdown("- Compact for a VLM: 4B params fits on a single T4 with QLoRA")
        st.markdown("- Same lineage as PaliGemma2: well-supported in transformers + PEFT")
    with w2:
        st.markdown("- State-of-the-art on PathMCQA: 70.0% vs ~55% for generic SigLIP")
        st.markdown("- Recent & authoritative: Google Health, Jan 2026 release")
        st.markdown("- 9 fixes from earlier PaliGemma2 notebook")

    st.markdown("#### 📦 Datasets")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """<div class="pv-card">
            <b>PatchCamelyon (PCam)</b><br>
            Binary tumour/normal classification of 96×96 H&E patches from sentinel lymph nodes.<br><br>
            <b>Source:</b> HuggingFace Hub<br>
            <b>License:</b> CC0-1.0</div>""",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            """<div class="pv-card">
            <b>NCT-CRC-HE-100K</b><br>
            9-class tissue type classification of 224×224 H&E patches from colorectal cancer.<br><br>
            <b>Source:</b> Zenodo 1214456<br>
            <b>License:</b> CC BY 4.0</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 📚 Citations")
    st.code(
        """@article{sellergren2026medgemma,
  title={MedGemma 1.5 Technical Report},
  author={Sellergren, Andrew and Gao, Chufan and Mahvar, Fereshteh and others},
  journal={arXiv preprint arXiv:2604.05081},
  year={2026}
}

@inproceedings{kather2018histological,
  title={100,000 histological images of human colorectal cancer and healthy tissue},
  author={Kather, Jakob Nikolas and Halama, Niels and Marx, Alexander},
  year={2018},
  publisher={Zenodo},
  doi={10.5281/zenodo.1214456}
}""",
        language="bibtex",
    )

    st.markdown("#### ⚖️ License Information")
    st.markdown(
        """<div class="pv-card">
        • <b>MedGemma 1.5 4B:</b> Health AI Developer Foundations terms — Research use only<br>
        • <b>PCam (PatchCamelyon):</b> CC0-1.0 (public domain)<br>
        • <b>NCT-CRC-HE-100K:</b> CC BY 4.0<br>
        • <b>Project source code:</b> Apache-2.0</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.8rem;color:#64748b;text-align:center'>"
        f"PcamVLM — {'ResNet-18 Lightweight' if _is_light else 'MedGemma 1.5 4B VLM'} mode. "
        f"Built with Streamlit, PyTorch, torchvision, Transformers, PEFT, TRL. Python 3.11</div>",
        unsafe_allow_html=True,
    )

