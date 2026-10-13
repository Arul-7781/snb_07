.PHONY: help install install-dev check-env smoke test lint manifest evaluate

PYTHON ?= python
CONFIG ?= configs/baseline_16k.yaml
AUDIO_ROOT ?= data/audio
MANIFEST ?= data/manifests/all.jsonl
CHECKPOINT ?=

help:
	@echo "install      Install the package after installing PyTorch"
	@echo "install-dev  Install package and development tools"
	@echo "check-env    Validate Python, FFmpeg, disk, and CUDA"
	@echo "smoke        Run a short synthetic CPU/GPU training smoke test"
	@echo "test         Run unit tests"
	@echo "lint         Run Ruff"
	@echo "manifest     Index audio under AUDIO_ROOT into MANIFEST"
	@echo "evaluate     Score CHECKPOINT on the test split and export WAV examples"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[dev,metrics]'

check-env:
	$(PYTHON) scripts/check_env.py

smoke:
	$(PYTHON) -m spectra_inn.train --config $(CONFIG) --smoke --max-steps 10

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

manifest:
	$(PYTHON) -m spectra_inn.data.manifest --root $(AUDIO_ROOT) --output $(MANIFEST)

evaluate:
	$(PYTHON) -m spectra_inn.evaluate --checkpoint $(CHECKPOINT)
