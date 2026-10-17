#!/usr/bin/env bash
# SparkStack manager — install, update, download, build, run.
#
# Host orchestration only: nothing here compiles anything. The build happens inside
# Docker (./Dockerfile); this script just drives docker compose and fetches models.
#
#   ./sparkstack.sh install          # first-time setup: .env, layout, models, build, up
#   ./sparkstack.sh update           # pull fresh base images, rebuild, recreate running profiles
#   ./sparkstack.sh prepare          # create the folder layout (all inside this repo)
#   ./sparkstack.sh download [group] # fetch models via uvx (all|llm|image|utility|audio)
#   ./sparkstack.sh source           # OPTIONAL: clone llama.cpp + sd.cpp into ./vendor for pinning
#   ./sparkstack.sh build            # docker compose build
#   ./sparkstack.sh server          # print the LAN IP + every endpoint, ready to paste
#   ./sparkstack.sh up [profiles...] # start (default: the whole stack)
#   ./sparkstack.sh down             # stop everything
#   ./sparkstack.sh status           # docker compose ps
#   ./sparkstack.sh logs <service>   # follow logs for a service
#   ./sparkstack.sh test             # functional checks + first-request warmup per service
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/.env"
MODELS_DIR="$ROOT/models"
VENDOR_DIR="$ROOT/vendor"

LLAMA_URL="https://github.com/ggml-org/llama.cpp"
SD_URL="https://github.com/leejet/stable-diffusion.cpp"

# Every profile that has buildable services (so `build` covers them all).
BUILD_PROFILES=(llm vision utility tools)
# Default for bare `up`: the whole stack. The 24GB Nemotron build fits in the
# unified pool alongside image generation, so everything coexists now.
DEFAULT_UP_PROFILES=(audio utility status llm vision)

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

load_env() {
  if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
  fi
}

ensure_env() {
  if [ ! -f "$ENV_FILE" ]; then
    log "creating .env from .env.example"
    cp "$ROOT/.env.example" "$ENV_FILE"
    die "edit '$ENV_FILE' (set HF_TOKEN) and run again"
  fi
  load_env
  case "${HF_TOKEN:-}" in
    ""|hf_xxxx*) warn "HF_TOKEN is not set — the gated FLUX.3 download will fail";;
  esac
}

ensure_docker() {
  have docker || die "docker is not installed"
  docker compose version >/dev/null 2>&1 || die "docker compose v2 is not available"
  docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (is it running? are you in the docker group?)"
}

# ───────────────────────────── requirements ─────────────────────────────
# Preconditions, not installables: the DGX Spark ships Ubuntu with git and curl;
# installing system packages via sudo from a driver script is surprising behavior.
ensure_git()   { have git   || die "git is required (apt install git)"; }
ensure_curl()  { have curl  || die "curl is required (apt install curl)"; }
ensure_uv() {
  have uvx && return 0
  log "installing uv for the current user (https://docs.astral.sh/uv/)"
  ensure_curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  have uvx || die "uv installed but uvx not on PATH — add ~/.local/bin to PATH and retry"
}

hf() { uvx --from 'huggingface_hub[cli]' hf "$@"; }

# ───────────────────────────── sources (optional) ─────────────────────────────
# The Docker build clones its own copies. ./vendor is only for inspecting or pinning
# commits — never fetched during install.
clone_or_update() { # url dir ref
  local url="$1" dir="$2" ref="$3"
  if [ -d "$dir/.git" ]; then
    log "updating $(basename "$dir") -> $ref"
    git -C "$dir" fetch --depth 1 origin "$ref" 2>/dev/null || git -C "$dir" fetch --tags origin
    git -C "$dir" checkout -q FETCH_HEAD 2>/dev/null || git -C "$dir" checkout -q "$ref"
  else
    log "cloning $(basename "$dir") @ $ref"
    git clone --recursive "$url" "$dir"
    git -C "$dir" checkout -q "$ref" 2>/dev/null || true
  fi
  git -C "$dir" submodule update --init --recursive >/dev/null 2>&1 || true
}

fetch_source() {
  ensure_git
  load_env
  mkdir -p "$VENDOR_DIR"
  clone_or_update "$LLAMA_URL" "$VENDOR_DIR/llama.cpp" "${LLAMA_REF:-master}"
  clone_or_update "$SD_URL"    "$VENDOR_DIR/stable-diffusion.cpp" "${SD_REF:-master}"
  log "sources in $VENDOR_DIR (optional — the Docker build clones upstream itself)"
}

# ───────────────────────────── layout ─────────────────────────────
# Everything the stack needs lives inside the project folder. This creates the tree;
# models are downloaded into it by ./sparkstack.sh download (incl. the TTS model) and
# by WhisperX on its first run. config/tts/ (server script + voices) is committed.
prepare_layout() {
  log "creating folder layout"
  mkdir -p "$ROOT/models/nemotron" "$ROOT/models/d1-3B" \
            "$ROOT/models/embeddinggemma-2" "$ROOT/models/flux3" \
            "$ROOT/models/flux3/loras" "$ROOT/models/flux3/upscalers" \
            "$ROOT/models/tts" \
            "$ROOT/data" "$ROOT/config/tts" "$ROOT/config/sd"
  log "layout ready — models/ data/ are git-ignored"
}

# ───────────────────────────── models ─────────────────────────────
download_models() { # group: all|llm|image|utility|audio
  local group="${1:-all}"
  case "$group" in
    all|llm|image|utility|audio) ;;
    *) die "unknown download group '$group' (all|llm|image|utility|audio)" ;;
  esac
  ensure_uv
  load_env
  [ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

  dl() { log "download $1"; mkdir -p "$2"; hf download "$1" "${@:3}" --local-dir "$2"; }

  case "$group" in
    all|llm)
      # NVIDIA's 24GB build served by vLLM (llm-nemotron) + the DFlash drafter.
      # Same QAD-compressed weights as the GGUF variant; b12x kernels, 262K ctx.
      dl nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB          "$MODELS_DIR/nemotron-vllm"
      dl nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr "$MODELS_DIR/nemotron-vllm-lilicorr" ;;
  esac
  case "$group" in
    all|utility)
      dl LiquidAI/d1-3B-GGUF           "$MODELS_DIR/d1-3B"          --include '*Q8_0*'
      dl unsloth/embeddinggemma-2-GGUF "$MODELS_DIR/embeddinggemma-2" --include '*Q8_0*' ;;
  esac
  case "$group" in
    all|image)
      dl unsloth/flux-3-image-klein-GGUF   "$MODELS_DIR/flux3" --include 'flux-3-image-klein-Q8_0.gguf'
      # text encoder — exact file, so the mmproj (vision projector) is NOT pulled in
      dl Qwen/Qwen3-VL-4B-Instruct-GGUF    "$MODELS_DIR/flux3" --include 'Qwen3VL-4B-Instruct-Q8_0.gguf'
      # VAE — from the same FLUX.3 repo you license anyway; NOT unsloth (GGUF only, no AE)
      dl black-forest-labs/flux-3-image-klein "$MODELS_DIR/flux3" --include 'ae.safetensors' ;;
  esac
  case "$group" in
    all|audio)
      # downloaded into Qwen3-TTS-VoiceDesign/ — the directory name the compose mount expects
      dl Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign "$MODELS_DIR/tts/Qwen3-TTS-VoiceDesign" ;;
  esac
  verify_models "$group"
}

# Every model file the compose services reference — checked against $MODELS_DIR.
verify_models() { # group
  local group="${1:-all}" missing=0 name
  local files=()
  case "$group" in
    all|llm)
      files+=(
        nemotron-vllm/model.safetensors.index.json
        nemotron-vllm/tokenizer.json
        nemotron-vllm-lilicorr/config.json
      ) ;;
  esac
  case "$group" in
    all|utility)
      files+=(
        d1-3B/d1-3B-Q8_0.gguf
        d1-3B/mmproj-d1-3B-Q8_0.gguf
        embeddinggemma-2/embeddinggemma-2-Q8_0.gguf
        embeddinggemma-2/mmproj-Q8_0.gguf
      ) ;;
  esac
  case "$group" in
    all|image)
      files+=(
        flux3/flux-3-image-klein-Q8_0.gguf
        flux3/Qwen3VL-4B-Instruct-Q8_0.gguf
        flux3/ae.safetensors
      ) ;;
  esac
  case "$group" in
    all|audio)
      files+=(tts/Qwen3-TTS-VoiceDesign/model.safetensors) ;;
  esac
  for name in "${files[@]}"; do
    if [ ! -f "$MODELS_DIR/$name" ]; then
      warn "missing: models/$name"
      missing=1
    fi
  done
  if [ "$missing" = 1 ]; then
    die "model files missing — re-run the download for that group (downloads resume where they left off)"
  fi
  log "all model files for group '$group' present"
}

# ───────────────────────────── server info ─────────────────────────────
# The Spark is a server: clients on the LAN reach it by IP, never localhost.
server_ip() {
  ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}'
}

print_endpoints() {
  local ip; ip="$(server_ip)"
  [ -n "$ip" ] || { warn "could not detect the LAN IP"; ip="<spark-ip>"; }
  cat <<EOF

SparkStack server — $ip
Point your services and agents at these endpoints:

  chat + vision .... http://$ip:9193/v1/chat/completions
  decisions ........ http://$ip:9194/v1/systemone
  embeddings ....... http://$ip:9195/v1/embeddings
  images ............ http://$ip:9196/v1/images/generations
  speech-to-text ... http://$ip:9191/transcribe
  text-to-speech ... http://$ip:9192/v1/audio/speech
  status page ...... http://$ip:9190
  container logs ... http://$ip:9197
EOF
}

# ───────────────────────────── compose ─────────────────────────────
profile_running() { docker compose --profile "$1" ps --status running -q 2>/dev/null | grep -q .; }

build() {
  ensure_docker
  local args=(); for p in "${BUILD_PROFILES[@]}"; do args+=(--profile "$p"); done
  # --pull refreshes base images: rolling tags (vllm cu134-nightly) pick up new
  # builds, stable tags (nvidia/cuda:13.0.0-*) are a no-op.
  log "docker compose ${args[*]} build --pull"
  docker compose "${args[@]}" build --pull
}

up() {
  ensure_docker
  local profiles=("$@")
  [ ${#profiles[@]} -eq 0 ] && profiles=("${DEFAULT_UP_PROFILES[@]}")
  local args=(); for p in "${profiles[@]}"; do args+=(--profile "$p"); done
  log "docker compose ${args[*]} up -d"
  docker compose "${args[@]}" up -d
}

# Every profile, so `down` actually stops everything (bare `docker compose
# down` ignores services in non-default profiles).
DOWN_PROFILES=(audio utility status llm vision tools)

down() {
  ensure_docker
  local args=(); for p in "${DOWN_PROFILES[@]}"; do args+=(--profile "$p"); done
  log "docker compose ${args[*]} down"
  docker compose "${args[@]}" down
}
ps()      { ensure_docker; docker compose ps; }
logs()    { ensure_docker; [ $# -ge 1 ] || die "usage: ./sparkstack.sh logs <service>"; docker compose logs -f "$@"; }

# ───────────────────────────── tests ─────────────────────────────
# Functional check + warmup per endpoint. The first request on each service
# pays lazy-init costs (CUDA graph capture, first-batch kernels, JIT paths);
# after `test`, real traffic gets warm latencies. Skips services that aren't
# up (start them with: ./sparkstack.sh up [profiles...]).
TEST_FAILURES=0

post_json() { # port path json -> prints response body
  curl -fsS -m 300 -H 'Content-Type: application/json' -d "$3" "http://localhost:$1$2"
}

test_service() { # name  — runs one check; records failure
  local name="$1"
  case "$name" in
    d1-decide|embeddings|llm-nemotron|image-flux3|whisper-x|tts-voicedesign|status) ;;
    *) die "unknown service '$name' (d1-decide|embeddings|llm-nemotron|image-flux3|whisper-x|tts-voicedesign|status)" ;;
  esac
  if [ "$(compose_ps_status "$name")" != "running" ]; then
    warn "$name: not running — skipping (start it with: ./sparkstack.sh up)"
    TEST_SKIPPED=$((TEST_SKIPPED+1))
    return 0
  fi
  log "testing $name"
  "check_$name" || { warn "$name: FAILED"; TEST_FAILURES=$((TEST_FAILURES+1)); return 1; }
}

compose_ps_status() { # service -> running|other
  docker compose -p sparkstack ps --format '{{.Service}} {{.Status}}' 2>/dev/null \
    | awk -v s="$1" '$1==s {print ($2=="Up"||$2=="Restarting") ? "running" : "other"; found=1} END {if(!found) print "other"}'
}

svc_port() { # service [container_port=8080] -> mapped host port
  docker compose -p sparkstack port "$1" "${2:-8080}" 2>/dev/null | sed 's/.*://'
}

wait_healthy() { # service timeout_s — wait for compose healthcheck
  local svc="$1" timeout="${2:-300}" waited=0
  while [ "$waited" -lt "$timeout" ]; do
    [ "$(compose_ps_status "$svc")" = "running" ] || return 1
    if docker compose -p sparkstack ps --format '{{.Service}} {{.Status}}' 2>/dev/null \
        | awk -v s="$svc" -v r="0" '$1==s && $0 ~ /\(healthy\)/ {r=1} END {exit !r}'; then
      return 0
    fi
    sleep 5; waited=$((waited+5))
  done
  return 1
}

# d1-decide: health + a real completion (warms CUDA paths + the chat route)
check_d1-decide() {
  local port reply
  port="$(svc_port d1-decide)" && curl -fsS -m 10 "http://localhost:$port/health" >/dev/null \
 && reply="$(post_json "$port" /v1/chat/completions '{"model":"d1-3B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}')" \
 && echo "    d1-decide: /health ok, chat replied: $(echo "$reply" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"
}

# embeddings: health + a real embedding (warms the multimodal text path)
check_embeddings() {
  local port reply
  port="$(svc_port embeddings)" && curl -fsS -m 10 "http://localhost:$port/health" >/dev/null \
 && reply="$(post_json "$port" /v1/embeddings '{"input":"warmup","model":"embeddinggemma-2"}')" \
 && echo "    embeddings: /health ok, vector dim $(echo "$reply" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))')"
}

# llm-nemotron: health (vLLM cold load takes minutes) + a real chat completion
check_llm-nemotron() {
  local port reply
  port="$(svc_port llm-nemotron)"
  wait_healthy llm-nemotron 600 \
 && reply="$(post_json "$port" /v1/chat/completions '{"model":"nemotron-3.5-bolt","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":200}')" \
 && echo "    llm-nemotron: chat replied: $(echo "$reply" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')"
}

# image-flux3: a real 512x512 generation (warm ~7s; also warms the diffusion
# pipeline — text encoder, easycache, VAE decode)
check_image-flux3() {
  local port png_bytes
  port="$(svc_port image-flux3)" \
 && png_bytes="$(post_json "$port" /v1/images/generations '{"model":"flux-3-image-klein","prompt":"warmup: a tiny robot watering a bonsai tree","size":"512x512","steps":16}' \
      | python3 -c 'import json,sys,base64; print(len(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"])))')" \
 && echo "    image-flux3: generated a 512x512 PNG (${png_bytes} bytes)"
}

# whisper-x: health endpoint; a real transcription needs an audio file, so this
# warms the HTTP path only (model loads on first real request)
check_whisper-x() {
  wait_healthy whisper-x 300 \
 && curl -fsS -m 60 "http://localhost:$(svc_port whisper-x 8003)/health" >/dev/null \
 && echo "    whisper-x: /health ok (cuda, whisper-large-v3)"
}

# tts-voicedesign: health + a short synthesis (warms CUDA graph capture — the
# expensive first-request path this service exists to hide)
check_tts-voicedesign() {
  local port bytes
  port="$(svc_port tts-voicedesign 8000)" && curl -fsS -m 300 "http://localhost:$port/health" >/dev/null \
 && bytes="$(curl -fsS -m 300 -H 'Content-Type: application/json' -d '{"model":"voicedesign","input":"warmup","voice":"vd_british_male"}' "http://localhost:$port/v1/audio/speech" | wc -c)" \
 && echo "    tts-voicedesign: synthesized ${bytes} bytes of audio"
}

# status: gatus dashboard
check_status() {
  local port
  port="$(svc_port status)" && curl -fsS -m 10 "http://localhost:$port/" >/dev/null \
 && echo "    status: dashboard responding on :$port"
}

cmd_test() {
  ensure_docker
  TEST_SKIPPED=0
  local svc
  for svc in d1-decide embeddings llm-nemotron image-flux3 whisper-x tts-voicedesign status; do
    test_service "$svc" || true
  done
  if [ "$TEST_FAILURES" -gt 0 ]; then
    die "$TEST_FAILURES test(s) failed — check: ./sparkstack.sh logs <service>"
  fi
  log "all tests passed ($TEST_SKIPPED skipped)"
}

# ───────────────────────────── commands ─────────────────────────────
cmd_install() {
  ensure_env
  ensure_uv
  ensure_docker
  prepare_layout
  download_models all
  build
  up
  log "done"
  print_endpoints
}

cmd_update() {
  ensure_env
  build
  # recreate any profiles that are currently running
  local running=()
  for p in "${BUILD_PROFILES[@]}" audio status; do
    if profile_running "$p"; then running+=("$p"); fi
  done
  [ ${#running[@]} -gt 0 ] && up "${running[@]}" || log "nothing was running; start with: ./sparkstack.sh up"
}

usage() {
  awk 'NR>1 { if (/^#/) { sub(/^# ?/,""); print } else { exit } }' "$0"
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    install)  cmd_install ;;
    update)   cmd_update ;;
    prepare)  prepare_layout ;;
    source)   fetch_source ;;
    download) download_models "${1:-all}" ;;
    build)    build ;;
    up)       up "$@" ;;
    down)     down ;;
    status|ps) ps ;;
    server)   print_endpoints ;;
    logs)     logs "$@" ;;
    test)     cmd_test ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown command '$cmd' (try: ./sparkstack.sh help)" ;;
  esac
}

main "$@"
