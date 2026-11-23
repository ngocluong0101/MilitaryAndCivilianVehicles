#!/usr/bin/env bash
# @author: Ngoc Luong <ngocluongg0101@gmail.com>
# Chạy tuần tự các thí nghiệm train; thí nghiệm đã có kết quả sẽ được bỏ qua.
#   bash scripts/run_experiments.sh                                    # exp1, exp2, exp3a, exp3b
#   bash scripts/run_experiments.sh configs/exp4_yolo11s_v2_960.yaml   # chỉ config chỉ định
# COOLDOWN_SECONDS (mặc định 300): nghỉ giữa hai lượt cho laptop nguội; đặt 0 để tắt.
# Log: logs/<tên>.log ; kết quả: reports/experiments/<tên>/
set -u
cd "$(dirname "$0")/.."
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-300}"
mkdir -p logs

gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo "n/a"; }

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(
    configs/exp1_yolov5su_legacy_640.yaml
    configs/exp2_yolov5su_v2_640.yaml
    configs/exp3a_yolo11n_v2_640.yaml
    configs/exp3b_yolo11s_v2_640.yaml
  )
fi

LAST=$(( ${#CONFIGS[@]} - 1 ))
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  name="$(basename "$cfg" .yaml)"
  if [ -f "reports/experiments/$name/run_info.json" ]; then
    echo "=== bỏ qua $name (đã có kết quả)"
    continue
  fi
  echo "=== $(date '+%F %T') bắt đầu $name (nhiệt độ GPU: $(gpu_temp) °C)"
  .venv/bin/python -m vehicledet train --config "$cfg" --skip-if-done 2>&1 | tee -a "logs/$name.log"
  echo "=== $(date '+%F %T') kết thúc $name (exit=${PIPESTATUS[0]}, nhiệt độ GPU: $(gpu_temp) °C)"
  if [ "$i" -lt "$LAST" ] && [ "$COOLDOWN_SECONDS" -gt 0 ]; then
    echo "=== nghỉ ${COOLDOWN_SECONDS}s cho máy nguội..."
    sleep "$COOLDOWN_SECONDS"
  fi
done

.venv/bin/python scripts/summarize_runs.py --roadmap ROADMAP.md
