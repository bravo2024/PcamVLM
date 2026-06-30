# PcamVLM — Histopathology Vision-Language Model

QLoRA fine-tune of **MedGemma 1.5 4B** (Google Health, Jan 2026) on
**PatchCamelyon (PCam)** + **NCT-CRC-HE-100K** for H&E patch classification.

```
PcamVLM/
  app.py                          # Streamlit dashboard
  train.py                        # TRL SFTTrainer QLoRA entrypoint
  pcam_medgemma_vlm_finetune.ipynb  # Colab-style walkthrough notebook
  src/
    core.py                       # prompts, label maps, logit-level yes/no
    data.py                       # PCam + NCT-CRC-HE-100K loaders + synthetic
    model.py                      # MedGemma 1.5 4B + LoRA wiring
    evaluate.py                   # accuracy / sens / spec / F1 / CM
    persist.py                    # save/load adapter + metadata
    infer.py                      # single-image prediction helpers
  tests/test_smoke.py             # static-only smoke tests
  requirements.txt
  Makefile                        # install / train / test / app
  runtime.txt                     # python-3.11
  .streamlit/config.toml
```

## What it does

1. **Loads** MedGemma 1.5 4B in 4-bit (nf4, double-quant) on bf16 compute.
2. **Attaches** LoRA adapters (r=16, alpha=32) to the language-tower attention
   and MLP projections (Gemma3 fused QKV + gate/up/down).
3. **Fine-tunes** on a stratified subset of PCam (binary tumor / normal) plus
   NCT-CRC-HE-100K (9 tissue types) using TRL `SFTTrainer`.
4. **Evaluates** accuracy / sensitivity / specificity / F1 on a held-out
   validation slice and renders a confusion matrix in the dashboard.
5. **Streams** an interactive Streamlit demo that runs in synthetic-fallback
   mode when the model adapter is absent.

## Why MedGemma 1.5 4B?

- **Compact** — fits a single Colab T4 (16 GB) with QLoRA at batch size 1 +
  gradient accumulation 8.
- **Medical-specialized** — trained on CAMELYON16/17, TCGA, histopathology
  patches, dermatology, ophthalmology, chest X-ray, and medical text.
- **Same lineage as PaliGemma2** (SigLIP + Gemma family) — keeps the swap
  mechanically simple.
- **Recent & authoritative** — Google Health, Jan 13 2026 release, official
  Colab recipe with QLoRA.

The earlier `pcam_paligemma2_vlm_finetune.ipynb` (still at the repo root
alongside `pcam_paligemma2_vlm_finetune.ipynb.bak`) used `paligemma2-3b-pt-224`.
MedGemma 1.5 4B outperforms it on PathMCQA (70.0 % vs ~55 % for the generic
SigLIP baseline) and is a better starting point for histopathology.

## Datasets

| Dataset | Source | License | Task | Train (default) | Val |
|---|---|---|---|---|---|
| PCam | `huggingface.co/datasets/1aurent/PatchCamelyon` | CC0-1.0 | binary tumor/normal | 13,334 | 1,000 |
| NCT-CRC-HE-100K | `zenodo.org/records/1214456` | CC BY 4.0 | 9-class tissue type | 6,666 | 1,000 |

NCT-CRC is auto-downloaded on first run into `./.pcamvl_datasets/`. If the
network is unavailable the loader silently falls back to a deterministic
synthetic generator so the pipeline still produces a usable (if weak)
adapter.

## Quick start

### 1. Install

```sh
make install          # pip install -r requirements.txt
```

### 2. Smoke tests (no GPU, no downloads)

```sh
make test             # pytest -q tests
```

### 3. Train (needs Colab T4 or better, ~30-60 min)

```sh
# Optional: accept MedGemma license on HuggingFace:
#   https://huggingface.co/google/medgemma-1.5-4b-it
# Then set HF_TOKEN to a write-access token in your environment.
export HF_TOKEN=...

make train
# or override defaults:
python train.py --train-size 20000 --val-size 2000 --epochs 1
```

### 4. Run the dashboard

```sh
make app              # streamlit run app.py
```

The app supports three data sources (synthetic / PCam / NCT-CRC) and
gracefully degrades to synthetic-only mode if the adapter is missing.

### 5. Notebook

```sh
make notebook
```

`pcam_medgemma_vlm_finetune.ipynb` is a self-contained Colab-style walkthrough.

## Project conventions

- All `.py` files use `from __future__ import annotations`.
- Synthetic fallback so the dashboard works offline.
- `app.py` inserts `src/` into `sys.path` at runtime (no `pip install -e .`).
- `train.py` writes models to `models/`; `app.py` loads them from there.
- 9 fixes carried over from the orphan `pcam_paligemma2_vlm_finetune.ipynb`:
  Colab try/except for HF_TOKEN + google.colab imports, `torch_dtype=torch.bfloat16`,
  `TaskType.CAUSAL_LM` (MedGemma is decoder-only), gradient checkpointing,
  fused `adamw_torch_fused`, cosine schedule, `save_total_limit=2`,
  `processing_class=processor`, `max_new_tokens=10`, thumbnail+pad for
  non-square uploads.

## License notes

- **MedGemma 1.5 4B** — [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
  Research use only. You must accept the license on HuggingFace before downloading.
- **PCam (PatchCamelyon)** — CC0-1.0 (public domain).
- **NCT-CRC-HE-100K** — CC BY 4.0; cite Kather et al. 2018 (Zenodo 1214456).
- **Project source code** — Apache-2.0 (inherits from the parent repo).

## Citation

```bibtex
@article{sellergren2026medgemma,
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
}
```