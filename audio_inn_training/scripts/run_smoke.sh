#!/usr/bin/env bash
set -euo pipefail
python scripts/preflight.py --config configs/smoke.yaml
python -m audio_inn.train --config configs/smoke.yaml --run-name smoke
pytest -q
