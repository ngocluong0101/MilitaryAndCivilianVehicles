#!/usr/bin/env bash
# Tạo venv .venv cho project và cài phụ thuộc (đã dùng để tạo môi trường hiện tại).
# Dùng: bash scripts/setup_env.sh        (PYTHON=python3.12 bash scripts/setup_env.sh để đổi interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -U pip
# torch từ PyPI mặc định là bản CUDA (đã kiểm tra: torch 2.15.0+cu132 chạy với driver 595 trên RTX 4060)
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e . --no-deps

.venv/bin/python - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY
# Cập nhật bản đóng băng (chạy tay khi đổi phụ thuộc):
#   .venv/bin/python -m pip freeze --exclude-editable > requirements-lock.txt
