#!/usr/bin/env bash
# Chạy tuần tự các thí nghiệm train (qua đêm). Mỗi thí nghiệm đã có kết quả sẽ được bỏ qua.
#   bash scripts/run_experiments.sh                 # exp1, exp2, exp3a, exp3b
#   bash scripts/run_experiments.sh configs/exp4_yolo11s_v2_960.yaml   # chỉ chạy config chỉ định
# Log: logs/<tên>.log ; kết quả: reports/experiments/<tên>/ ; MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
set -u
cd "$(dirname "$0")/.."
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
mkdir -p logs

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    configs/exp1_yolov5su_legacy_640.yaml
    configs/exp2_yolov5su_v2_640.yaml
    configs/exp3a_yolo11n_v2_640.yaml
    configs/exp3b_yolo11s_v2_640.yaml
  )
fi

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  echo "=== $(date '+%F %T') bắt đầu $name"
  .venv/bin/python -m vehicledet train --config "$cfg" --skip-if-done 2>&1 | tee -a "logs/$name.log"
  echo "=== $(date '+%F %T') kết thúc $name (exit=${PIPESTATUS[0]})"
done

.venv/bin/python scripts/summarize_runs.py --roadmap ROADMAP.md
