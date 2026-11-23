#!/usr/bin/env bash
# @author: Ngoc Luong <ngocluongg0101@gmail.com>
# Tạo .venv và cài phụ thuộc. Đổi interpreter: PYTHON=python3.12 bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -U pip
# torch từ PyPI mặc định là bản CUDA
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps

.venv/bin/python - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY
