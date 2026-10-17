#!/usr/bin/env bash
# SparkStack host memory watchdog.
#
# Runs on the HOST, not in a container: on unified memory the danger is the host itself
# sliding into unreclaimable memory, and a container cannot rescue the host it is starving.
#
# Driven by watchdog.timer (~30s). Each run:
#   - read MemAvailable from /proc/meminfo
#   - if below the threshold, stop image generation first (vision, bursty and
#     restartable); if the next run is still below threshold, stop the big model (llm)
#   - the always-on services (audio, utility, status) are never touched
#   - append every action to ./data/watchdog.log
#
# The whole stack is budgeted to coexist (~112 GiB total), so this only fires when
# something outside the plan is eating memory (a 4K image run at an unusual
# resolution, a client flooding a context, host-side experiments).
set -euo pipefail

COMPOSE_DIR="${SPARKSTACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COMPOSE_DIR"

THRESHOLD_MB="${SPARKSTACK_MIN_MB:-3072}"   # 3 GiB
LOG="$COMPOSE_DIR/data/watchdog.log"
mkdir -p "$(dirname "$LOG")"

# The compose file fail-fasts on a missing HF_TOKEN (${HF_TOKEN:?}); the watchdog must
# still be able to stop services when .env is absent or incomplete, so provide a dummy.
export HF_TOKEN="${HF_TOKEN:-watchdog-no-token}"

avail_kb="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
avail_mb=$(( avail_kb / 1024 ))
ts="$(date -Is)"

# Healthy: nothing to do.
if [ "$avail_mb" -ge "$THRESHOLD_MB" ]; then
  exit 0
fi

running() { docker compose --profile "$1" ps --status running -q 2>/dev/null | grep -q .; }

if running vision; then
  echo "$ts low memory (${avail_mb} MiB < ${THRESHOLD_MB}) -> docker compose --profile vision stop" >>"$LOG"
  docker compose --profile vision stop
elif running llm; then
  echo "$ts low memory (${avail_mb} MiB < ${THRESHOLD_MB}) -> docker compose --profile llm stop" >>"$LOG"
  docker compose --profile llm stop
else
  echo "$ts low memory (${avail_mb} MiB < ${THRESHOLD_MB}) -- no heavy profile running" >>"$LOG"
fi
