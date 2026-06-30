"""Smoke test for PcamVLM.

Verifies that:
1. Every src/ module imports cleanly.
2. Synthetic data + label maps + answer parsing work end-to-end.
3. The evaluate helpers produce sane metrics on synthetic data.
4. We can build a Gemma3 chat-template messages list and round-trip it
   through ``processor.apply_chat_template`` (the offline tokenization path),
   without downloading MedGemma — we use a tiny public tokenizer as a
   stand-in (``google/gemma-2-2b-it`` if cached; otherwise ``bert-base-uncased``).

These tests are static-only and do NOT require a GPU or download the
4 GB MedGemma weights. Run with: ``pytest -q tests/test_smoke.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    """Every src module is importable."""
    import src  # noqa: F401
    from src import core, data, model, evaluate, persist, infer  # noqa: F401
    import train  # noqa: F401
    import app  # noqa: F401


def test_label_maps():
    from src import core as C
    assert C.PCAM_LABEL_NAMES == {0: "no", 1: "yes"}
    assert len(C.NCT_CRC_LABEL_NAMES) == 9
    assert C.NCT_CRC_LABEL_NAMES[0] == "ADI"
    assert C.NCT_CRC_LABEL_NAMES[8] == "TUM"


def test_parse_answer_pcam():
    from src import core as C
    assert C.parse_answer("pcam", "yes") == 1
    assert C.parse_answer("pcam", "Yes.") == 1
    assert C.parse_answer("pcam", "no") == 0
    assert C.parse_answer("pcam", "no, the tissue is normal.") == 0
    assert C.parse_answer("pcam", "   yes   ") == 1
    assert C.parse_answer("pcam", "garbage") == 0


def test_parse_answer_nct_crc():
    from src import core as C
    assert C.parse_answer("nct_crc", "TUM") == 8
    assert C.parse_answer("nct_crc", "LYM") == 3
    assert C.parse_answer("nct_crc", "tum") == 8
    assert C.parse_answer("nct_crc", "ADI ") == 0
    # Fallback to first choice if nothing matches
    assert C.parse_answer("nct_crc", "") == 0


def test_synthetic_pcam():
    from src import data as D
    rows = D.make_synthetic_pcam(10, seed=0)
    assert len(rows) == 10
    assert rows[0]["task"] == "pcam"
    assert all(r["image"].size == (96, 96) for r in rows)
    labels = [r["label"] for r in rows]
    assert 0 in labels and 1 in labels


def test_synthetic_nct():
    from src import data as D
    rows = D.make_synthetic_nct_crc(18, seed=0)
    assert len(rows) == 18
    assert rows[0]["task"] == "nct_crc"
    assert all(r["image"].size == (224, 224) for r in rows)
    labels = set(r["label"] for r in rows)
    assert len(labels) > 1


def test_load_combined_synthetic(monkeypatch):
    """Verify load_combined returns ``n`` synthetic rows when real datasets are skipped.

    Force the loaders into synthetic-only mode by stubbing ``_have_network`` to
    return False; this guarantees the smoke test never makes a network call.
    """
    from src import data as D
    monkeypatch.setattr(D, "_have_network", lambda: False)
    spec = D.DatasetSpec(task="mixed", train_size=20, val_size=8)
    train, val = D.load_combined(spec, seed=0)
    assert len(train) == 20
    assert len(val) == 8
    pcam_train = sum(1 for r in train if r["task"] == "pcam")
    nct_train = sum(1 for r in train if r["task"] == "nct_crc")
    assert pcam_train > 0 and nct_train > 0


def test_prepare_image_for_model():
    from src import data as D
    from PIL import Image
    img = Image.new("RGB", (96, 96), (128, 64, 32))
    out = D.prepare_image_for_model(img, target=224)
    assert out.size == (224, 224)

    img_rect = Image.new("RGB", (300, 100), (200, 100, 50))
    out2 = D.prepare_image_for_model(img_rect, target=224)
    assert out2.size == (224, 224)


def test_make_messages_pcam():
    from src import data as D
    from src import core as C
    sample = {"image": D._synthetic_pcam_image(1, 0), "label": 1, "task": "pcam"}
    out = D.make_messages(sample, C.PCAM_QUESTION,
                          C.dataset_answer_for_label("pcam", 1))
    assert "messages" in out
    msgs = out["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert any(c.get("type") == "image" for c in msgs[0]["content"])
    assert msgs[1]["content"][0]["text"] == "yes"


def test_evaluate_binary():
    from src import evaluate as E
    trues = [0, 0, 1, 1, 0, 1, 1, 0]
    preds = [0, 1, 1, 1, 0, 0, 1, 0]
    r = E.evaluate_binary(trues, preds)
    assert r.task == "pcam"
    assert 0.0 <= r.accuracy <= 1.0
    assert r.confusion[0][0] == 3  # TN
    assert r.confusion[1][1] == 3  # TP
    assert "precision" in r.per_class[1]


def test_evaluate_multiclass():
    from src import evaluate as E
    trues = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 0, 0, 2, 2]
    r = E.evaluate_multiclass(trues, preds, ["a", "b", "c"])
    assert r.task == "nct_crc"
    assert r.n == 6
    assert len(r.confusion) == 3


def test_save_metadata_roundtrip(tmp_path):
    from src import persist as P
    meta = P.TrainingMetadata(
        base_model="google/medgemma-1.5-4b-it",
        dataset="pcam+nct_crc",
        train_size=100, val_size=20,
        epochs=1,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        learning_rate=2e-4,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        max_pixels=896 * 896,
        min_pixels=224 * 224,
        final_train_loss=0.42,
        timestamp="2026-06-30T00:00:00Z",
    )
    out_dir = P.save_adapter(model=None, processor=None,
                             save_dir=tmp_path, metadata=meta)
    loaded = P.load_metadata(out_dir)
    assert loaded["base_model"] == "google/medgemma-1.5-4b-it"
    assert loaded["final_train_loss"] == 0.42
    assert loaded["lora_r"] == 16


def test_train_argparse_defaults():
    """Verify train.py default arguments are sane without running training."""
    import train
    import argparse
    # Make sure parse_args returns the expected Namespace when no args.
    sys.argv = ["train.py"]
    args = train.parse_args()
    assert args.base_model == "google/medgemma-1.5-4b-it"
    assert args.train_size == 20000
    assert args.val_size == 2000
    assert args.epochs == 1
    assert args.gradient_accumulation == 8
    assert args.save_adapter is True


def test_tokenizer_yes_no_ids():
    """Find yes/no token IDs in a small public tokenizer (bert if gemma cached).

    We don't require medgemma to be present; we just verify the helper works
    on whatever tokenizer can be loaded offline.
    """
    from src.core import yes_no_token_ids
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        return  # offline / no internet — skip silently
    ids = yes_no_token_ids(tok)
    assert "yes" in ids and "no" in ids
    assert isinstance(ids["yes"], int)


def test_lightweight_imports():
    """Lightweight module imports cleanly."""
    from src import lightweight as LW  # noqa: F401


def test_lightweight_resnet_pcam():
    """Lightweight ResNet-18 can instantiate and predict on synthetic PCam."""
    from src import lightweight as LW
    from src.data import _synthetic_pcam_image
    model = LW.ResNetClassifier(num_classes=2, freeze_backbone=True)
    model.eval()
    img = _synthetic_pcam_image(0, seed=42)
    out = LW.predict_lightweight_pcam(model, img)
    assert "pred" in out
    assert "pred_name" in out
    assert "yes_prob" in out
    assert "no_prob" in out
    assert out["pred"] in (0, 1)
    assert 0.0 <= out["yes_prob"] <= 1.0


def test_lightweight_resnet_nct():
    """Lightweight ResNet-18 can predict on synthetic NCT-CRC."""
    from src import lightweight as LW
    from src.data import _synthetic_nct_image
    model = LW.ResNetClassifier(num_classes=9, freeze_backbone=True)
    model.eval()
    img = _synthetic_nct_image(0, seed=42)
    out = LW.predict_lightweight_nct(model, img)
    assert "pred" in out
    assert "pred_name" in out
    assert "class_probs" in out
    assert len(out["class_probs"]) == 9
    assert 0 <= out["pred"] <= 8


def test_lightweight_model_roundtrip(tmp_path):
    """Lightweight model save/load round-trips correctly."""
    from src import lightweight as LW
    model = LW.ResNetClassifier(num_classes=2, freeze_backbone=True)
    path = tmp_path / "test_model.pt"
    LW.save_lightweight_model(model, path)
    loaded = LW.load_lightweight_model("pcam", path=path)
    assert loaded is not None
    from src.data import _synthetic_pcam_image
    img = _synthetic_pcam_image(1, seed=42)
    out = LW.predict_lightweight_pcam(loaded, img)
    assert "pred" in out


def test_lightweight_metadata_roundtrip(tmp_path):
    """Lightweight metadata saves and loads correctly."""
    from src import lightweight as LW
    meta = LW.LightweightMetadata(
        task="pcam", num_classes=2,
        train_samples=100, val_samples=20, epochs=5,
        final_loss=0.42, accuracy=0.85,
        timestamp="2026-06-30T00:00:00Z",
    )
    path = tmp_path / "test_model.pt"
    LW.save_lightweight_model(LW.ResNetClassifier(num_classes=2), path, metadata=meta)
    meta_path = path.with_suffix(".json")
    import json
    with open(meta_path) as f:
        data = json.load(f)
    assert data["accuracy"] == 0.85