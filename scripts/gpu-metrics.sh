#!/usr/bin/env bash
# Write the GPU fields that work on GB10 to node-exporter's textfile collector.
set -euo pipefail

ROOT="${SPARKSTACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${1:-$ROOT/data/node-exporter/textfile/gpu.prom}"
TMP="${OUT}.tmp"
mkdir -p "$(dirname "$OUT")"

sample=""
up=0
if command -v nvidia-smi >/dev/null 2>&1 \
  && sample="$(nvidia-smi --query-gpu=index,utilization.gpu,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null)"; then
  up=1
fi

{
  cat <<'HEADER'
# HELP nvidia_gpu_metrics_up Whether nvidia-smi returned a sample
# TYPE nvidia_gpu_metrics_up gauge
# HELP nvidia_gpu_utilization_percent GPU utilization in percent
# TYPE nvidia_gpu_utilization_percent gauge
# HELP nvidia_gpu_temperature_celsius GPU temperature in Celsius
# TYPE nvidia_gpu_temperature_celsius gauge
# HELP nvidia_gpu_power_draw_watts GPU power draw in watts
# TYPE nvidia_gpu_power_draw_watts gauge
# HELP nvidia_gpu_sm_clock_mhz GPU SM clock in MHz
# TYPE nvidia_gpu_sm_clock_mhz gauge
# HELP nvidia_gpu_metrics_timestamp_seconds Last successful collector run
# TYPE nvidia_gpu_metrics_timestamp_seconds gauge
HEADER
  echo "nvidia_gpu_metrics_up $up"
  if [ "$up" -eq 1 ]; then
    printf '%s\n' "$sample" | awk -F', ' '
      NF >= 5 {
        gpu = $1
        gsub(/ /, "", gpu)
        if ($2 != "N/A") printf "nvidia_gpu_utilization_percent{gpu=\"%s\"} %s\n", gpu, $2
        if ($3 != "N/A") printf "nvidia_gpu_temperature_celsius{gpu=\"%s\"} %s\n", gpu, $3
        if ($4 != "N/A") printf "nvidia_gpu_power_draw_watts{gpu=\"%s\"} %s\n", gpu, $4
        if ($5 != "N/A") printf "nvidia_gpu_sm_clock_mhz{gpu=\"%s\"} %s\n", gpu, $5
      }
    '
  fi
  echo "nvidia_gpu_metrics_timestamp_seconds $(date +%s)"
} > "$TMP"
mv "$TMP" "$OUT"
