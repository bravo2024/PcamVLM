"""PcamVLM data loaders.

Two real-world histopathology sources, plus a synthetic fallback that always
works without network access so that the Streamlit app demos on first launch.

Datasets
--------
1. PCam (PatchCamelyon) via ``1aurent/PatchCamelyon`` on HuggingFace Hub.
   - 96x96 H&E patches of sentinel lymph node sections
   - Binary labels: 0 = normal, 1 = tumor
   - License: CC0-1.0

2. NCT-CRC-HE-100K from Zenodo (https://zenodo.org/records/1214456).
   - 100k 224x224 H&E patches from colorectal cancer histology
   - 9 classes (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM)
   - License: CC BY 4.0

The Zenodo zip is large (~1.2 GB). On first run the loader auto-downloads it
into ``./.pcamvl_datasets/NCT-CRC-HE-100K`` and extracts the archive. If the
download fails (no network, Zenodo down, disk full) we fall back to PCam or
to the synthetic generator, in that order of preference.
"""
from __future__ import annotations

import hashlib
import io
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = ROOT / ".pcamvl_datasets"

NCT_CRC_ZENODO_URL = (
    "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
)
NCT_CRC_ZIP_NAME = "NCT-CRC-HE-100K.zip"

# Hard cap on Zenodo download so the loader never silently pulls 1+ GB
# during smoke tests or quick app demos. Set to None to allow the full
# archive (e.g. when the user explicitly requests the real data).
NCT_CRC_MAX_BYTES_DEFAULT = None  # No cap — full 1.2 GB archive

PCAM_HF_ID = "1aurent/PatchCamelyon"

SYNTH_SIZE = 96
SYNTH_CLASS_NAMES_PCAM = {0: "no", 1: "yes"}
SYNTH_CLASS_NAMES_NCT = {
    0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC",
    5: "MUS", 6: "NORM", 7: "STR", 8: "TUM",
}


# ---------------------------------------------------------------------------
# Synthetic fallback (works offline, deterministic per seed)
# ---------------------------------------------------------------------------

def _synthetic_pcam_image(label: int, seed: int) -> Image.Image:
    """Render a deterministic fake H&E patch.

    Pink (eosin) cytoplasm + purple (hematoxylin) nuclei. Tumor patches are
    denser and darker.
    """
    rng = random.Random(seed * 31 + label)
    base_eosin = 215 if label == 0 else 175
    img = Image.new("RGB", (SYNTH_SIZE, SYNTH_SIZE), (base_eosin, 200, 210))
    px = img.load()
    density = 12 if label == 0 else 55
    spread = 6 if label == 0 else 3
    for _ in range(density * 30):
        cx = rng.randrange(SYNTH_SIZE)
        cy = rng.randrange(SYNTH_SIZE)
        r = rng.randint(2, spread)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < SYNTH_SIZE and 0 <= y < SYNTH_SIZE:
                    px[x, y] = (60 + rng.randrange(30), 30, 90 + rng.randrange(40))
    return img


def _synthetic_nct_image(label: int, seed: int) -> Image.Image:
    """Render a deterministic fake CRC tissue patch (224x224)."""
    rng = random.Random(seed * 97 + label)
    palettes = {
        0: ((240, 220, 200), (180, 140, 130)),   # ADI  - adipose (white/yellow)
        1: ((220, 220, 220), (180, 180, 180)),   # BACK - background (gray)
        2: ((150, 100,  60), ( 80,  50,  30)),   # DEB  - debris (brown)
        3: ((180, 120, 180), (130,  60, 130)),   # LYM  - lymphocytes (purple)
        4: ((200, 230, 200), (140, 180, 140)),   # MUC  - mucus (greenish)
        5: ((230, 200, 200), (200, 150, 150)),   # MUS  - muscle (pink)
        6: ((220, 200, 200), (180, 140, 140)),   # NORM - normal mucosa
        7: ((210, 180, 170), (150, 100,  90)),   # STR  - stroma
        8: ((170, 100, 100), (100,  40,  40)),   # TUM  - tumor (deep red)
    }
    base, accent = palettes[label]
    img = Image.new("RGB", (224, 224), base)
    px = img.load()
    density = 200 + label * 30
    for _ in range(density * 10):
        cx = rng.randrange(224)
        cy = rng.randrange(224)
        r = rng.randint(1, 5)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < 224 and 0 <= y < 224:
                    px[x, y] = accent
    return img


def make_synthetic_pcam(n: int, seed: int = 0) -> List[Dict]:
    """Build ``n`` deterministic synthetic PCam-like samples (balanced)."""
    out: List[Dict] = []
    for i in range(n):
        label = i % 2
        out.append({"image": _synthetic_pcam_image(label, seed + i),
                    "label": label, "task": "pcam"})
    return out


def make_synthetic_nct_crc(n: int, seed: int = 0) -> List[Dict]:
    """Build ``n`` deterministic synthetic NCT-CRC samples (balanced)."""
    out: List[Dict] = []
    for i in range(n):
        label = i % 9
        out.append({"image": _synthetic_nct_image(label, seed + i),
                    "label": label, "task": "nct_crc"})
    return out


# ---------------------------------------------------------------------------
# PCam loader
# ---------------------------------------------------------------------------

def _have_network() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
        return True
    except Exception:
        return False


def load_pcam(split: str, n: int, seed: int = 0,
             max_iter: int = 50_000):
    """Load PCam from HuggingFace; returns list of {image,label,task}.

    Falls back to synthetic if network/Hub is unavailable. Uses
    ``streaming=True`` so we don't pull the full 327K-image dataset just to
    take ``n`` samples. ``max_iter`` caps the number of streamed rows so
    that a flaky network can't make the loader hang.
    """
    if not _have_network():
        sys.stderr.write("[data] offline; using synthetic PCam fallback\n")
        return make_synthetic_pcam(n, seed=seed)
    try:
        from datasets import load_dataset
        ds = load_dataset(PCAM_HF_ID, split=split, streaming=True)
        out: List[Dict] = []
        per = n // 2
        seen_0 = 0
        seen_1 = 0
        seen_total = 0
        for row in ds:
            seen_total += 1
            if seen_total > max_iter:
                sys.stderr.write(
                    f"[data] PCam stream cap of {max_iter} rows hit; "
                    "falling back to synthetic for the remainder.\n"
                )
                out.extend(make_synthetic_pcam(n - len(out), seed=seed + seen_total))
                break
            lbl = int(row["label"])
            if lbl == 0 and seen_0 >= per:
                if seen_1 >= per:
                    break
                continue
            if lbl == 1 and seen_1 >= per:
                if seen_0 >= per:
                    break
                continue
            img = row["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            out.append({"image": img, "label": lbl, "task": "pcam"})
            if lbl == 0:
                seen_0 += 1
            else:
                seen_1 += 1
            if seen_0 + seen_1 >= n:
                break
        if not out:
            sys.stderr.write("[data] PCam returned empty; using synthetic\n")
            return make_synthetic_pcam(n, seed=seed)
        return out
    except Exception as e:  # pragma: no cover - depends on network
        sys.stderr.write(f"[data] PCam load failed ({e}); synthetic fallback\n")
        return make_synthetic_pcam(n, seed=seed)


# ---------------------------------------------------------------------------
# NCT-CRC-HE-100K loader (with Zenodo auto-download)
# ---------------------------------------------------------------------------

def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    pct = min(100, block_num * block_size * 100 // total_size)
    sys.stderr.write(f"\r[data] downloading NCT-CRC zip: {pct}%")
    if pct >= 100:
        sys.stderr.write("\n")


def ensure_nct_crc_extracted(force: bool = False,
                             max_bytes: Optional[int] = NCT_CRC_MAX_BYTES_DEFAULT,
                             timeout: float = 10.0) -> Optional[Path]:
    """Download + extract the NCT-CRC zip if not already present.

    Returns the path to the extracted ``NCT-CRC-HE-100K`` directory, or
    ``None`` if the download could not be completed.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_CACHE / NCT_CRC_ZIP_NAME
    extract_dir = DATA_CACHE / "NCT-CRC-HE-100K"
    if extract_dir.is_dir() and any(extract_dir.iterdir()) and not force:
        return extract_dir
    if not _have_network():
        sys.stderr.write("[data] offline; skipping NCT-CRC download\n")
        return None
    try:
        sys.stderr.write(f"[data] downloading {NCT_CRC_ZENODO_URL}\n")
        if max_bytes is not None:
            sys.stderr.write(f"[data]   capping at {max_bytes/1e6:.0f} MB to keep demos fast\n")
        # Use a streaming request so we can enforce max_bytes + timeout.
        req = urllib.request.Request(NCT_CRC_ZENODO_URL)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", "0") or 0)
            if max_bytes is not None and total and total > max_bytes:
                sys.stderr.write(
                    f"[data] server reports {total/1e6:.0f} MB which exceeds the "
                    f"{max_bytes/1e6:.0f} MB cap; download aborted, falling back to "
                    "synthetic data.\n"
                )
                return None
            written = 0
            chunk = 64 * 1024
            with open(zip_path, "wb") as out:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    written += len(buf)
                    if max_bytes is not None and written > max_bytes:
                        sys.stderr.write(
                            f"[data] cap of {max_bytes/1e6:.0f} MB hit; aborting, "
                            "falling back to synthetic data.\n"
                        )
                        out.close()
                        try:
                            zip_path.unlink()
                        except Exception:
                            pass
                        return None
                    out.write(buf)
                    if total:
                        pct = written * 100 // total
                        sys.stderr.write(f"\r[data]   {pct}% ({written/1e6:.1f} MB)")
        if total:
            sys.stderr.write("\n")
        sys.stderr.write("[data] extracting zip (this takes a minute)\n")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_CACHE)
        return extract_dir if extract_dir.is_dir() else None
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[data] NCT-CRC download/extract failed: {e}\n")
        return None


def _nct_crc_dir_iter(root: Path):
    """Yield (image_path, class_index) pairs from NCT-CRC-HE-100K directory.

    Subdirectory names are the class labels (ADI, BACK, ...). We map them to
    ints according to ``core.NCT_CRC_LABEL_NAMES``.
    """
    from src.core import NCT_CRC_LABEL_NAMES
    name_to_idx = {v: k for k, v in NCT_CRC_LABEL_NAMES.items()}
    if not root.is_dir():
        return
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = name_to_idx.get(sub.name.upper())
        if label is None:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                yield f, label


def load_nct_crc(n: int, seed: int = 0,
               max_bytes: Optional[int] = NCT_CRC_MAX_BYTES_DEFAULT):
    """Load NCT-CRC-HE-100K (auto-downloading if necessary).

    Falls back to synthetic if no network or no Zenodo access.
    """
    root = ensure_nct_crc_extracted(max_bytes=max_bytes)
    if root is None:
        sys.stderr.write("[data] using synthetic NCT-CRC fallback\n")
        return make_synthetic_nct_crc(n, seed=seed)
    try:
        all_pairs = list(_nct_crc_dir_iter(root))
    except Exception as e:
        sys.stderr.write(f"[data] NCT-CRC read failed: {e}; synthetic fallback\n")
        return make_synthetic_nct_crc(n, seed=seed)
    if not all_pairs:
        sys.stderr.write("[data] NCT-CRC empty; synthetic fallback\n")
        return make_synthetic_nct_crc(n, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    per = n // 9
    out: List[Dict] = []
    counts = {i: 0 for i in range(9)}
    for path, label in all_pairs:
        if counts[label] >= per:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        out.append({"image": img, "label": label, "task": "nct_crc",
                    "path": str(path)})
        counts[label] += 1
        if sum(counts.values()) >= n:
            break
    if len(out) < n // 2:
        sys.stderr.write("[data] NCT-CRC too few samples; mixing synthetic\n")
        out.extend(make_synthetic_nct_crc(n - len(out), seed=seed + 999))
    return out


# ---------------------------------------------------------------------------
# Combined dataset assembly
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    task: str
    train_size: int
    val_size: int


def load_combined(spec: DatasetSpec, seed: int = 0,
                  nct_max_bytes: Optional[int] = NCT_CRC_MAX_BYTES_DEFAULT
                  ) -> Tuple[List[Dict], List[Dict]]:
    """Return ``(train, val)`` according to ``spec``.

    Splits the requested ``train_size`` between PCam and NCT-CRC roughly in
    proportion to the spec sizes: 2/3 PCam + 1/3 NCT-CRC for train, 1/1 for val.

    ``nct_max_bytes`` defaults to a 200 MB sample so quick demos do not pull
    the full 1.2 GB NCT-CRC archive. Pass ``None`` to allow the full download.
    """
    pcam_train_n = int(spec.train_size * 2 / 3)
    pcam_val_n = int(spec.val_size / 2)
    nct_train_n = spec.train_size - pcam_train_n
    nct_val_n = spec.val_size - pcam_val_n

    train_pcam = load_pcam("train", pcam_train_n, seed=seed)
    val_pcam = load_pcam("valid", pcam_val_n, seed=seed)
    train_nct = load_nct_crc(nct_train_n, seed=seed + 2,
                             max_bytes=nct_max_bytes)
    val_nct = load_nct_crc(nct_val_n, seed=seed + 3,
                           max_bytes=nct_max_bytes)

    train = train_pcam + train_nct
    val = val_pcam + val_nct
    rng = random.Random(seed + 4)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Image preprocessing for MedGemma 1.5 4B
# ---------------------------------------------------------------------------

def prepare_image_for_model(image: Image.Image, target: int = 224) -> Image.Image:
    """Pad to a square (preserve aspect) and resize to ``target`` on the long side.

    MedGemma's processor accepts arbitrary sizes, but a square 224x224 matches
    PCam (96->224 upscale) and NCT-CRC (224 native) consistently.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w == h:
        return image.resize((target, target), Image.LANCZOS)
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(image, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((target, target), Image.LANCZOS)


# ---------------------------------------------------------------------------
# HuggingFace ``Dataset``-friendly transforms (for TRL SFTTrainer)
# ---------------------------------------------------------------------------

def make_messages(example: Dict, question: str, answer: str) -> Dict:
    """Build a Gemma3 chat-template compatible ``messages`` field.
    
    The image is passed raw — the processor's min_pixels/max_pixels
    handles all resizing during training/inference.
    """
    img = example["image"]
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ],
    }


__all__ = [
    "DATA_CACHE",
    "PCAM_HF_ID",
    "NCT_CRC_ZENODO_URL",
    "NCT_CRC_MAX_BYTES_DEFAULT",
    "DatasetSpec",
    "load_pcam",
    "load_nct_crc",
    "load_combined",
    "make_synthetic_pcam",
    "make_synthetic_nct_crc",
    "make_messages",
    "prepare_image_for_model",
    "ensure_nct_crc_extracted",
]