.PHONY: install install-deploy train train-light test clean app notebook help

PY ?= python

help:
	@echo "PcamVLM — MedGemma 1.5 4B QLoRA fine-tune on PCam + NCT-CRC-HE-100K"
	@echo ""
	@echo "Targets:"
	@echo "  install       pip install -r requirements.txt           # full (GPU+Colab)"
	@echo "  install-deploy pip install -r requirements-deploy.txt   # lightweight (CPU)"
	@echo "  train         python train.py              # QLoRA SFT on combined dataset"
	@echo "  train-light   python train_light.py        # ResNet-18 (CPU, synthetic data)"
	@echo "  test          pytest -q                    # static-only smoke tests"
	@echo "  app           streamlit run app.py         # interactive dashboard"
	@echo "  notebook      jupyter notebook             # open the Colab-style notebook"
	@echo "  clean         remove __pycache__, .pytest_cache, logs"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

install-deploy:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements-deploy.txt

train:
	$(PY) train.py

train-light:
	$(PY) train_light.py

test:
	$(PY) -m pytest -q tests

app:
	$(PY) -m streamlit run app.py

notebook:
	$(PY) -m jupyter notebook pcam_medgemma_vlm_finetune.ipynb

notebook-light:
	$(PY) -m jupyter notebook pcam_lightweight_train.ipynb

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__
	rm -rf logs/*.log
	rm -rf .pcamvl_datasets models/pcam-medgemma-lora