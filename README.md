# SparkStack

![SparkStack — a compact 128 GB AI supercomputer projecting vision, image and audio capabilities](docs/sparkstack-banner.png)

**Ship intelligence in a 128 GB box — for fun and profit.** A complete private AI
server for the **NVIDIA DGX Spark**. The Spark is the server;
everything else on your LAN — agents, apps, other machines — are its clients:

| Port | Endpoint | Serves |
|---|---|---|
| `:9193` | `POST /v1/chat/completions` | 27B chat + vision |
| `:9194` | `POST /v1/systemone` | one-shot decisions (yes/no/score) |
| `:9195` | `POST /v1/embeddings` | text/image/video/audio embeddings |
| `:9198` | `POST /v1/rerank` | cross-encoder reranking |
| `:9196` | `POST /v1/images/generations` | FLUX.3 image generation + editing |
| `:9191` | `POST /transcribe` | speech-to-text |
| `:9192` | `POST /v1/audio/speech` | text-to-speech, 10 voices |
| `:9190` | browser | status dashboard |
| `:9197` | browser | container logs |
| `:9199` | browser | Prometheus metrics |
| `:9200` | browser | Grafana dashboards |

Prefix any endpoint with your Spark's LAN address — the installer prints it, and
`./sparkstack.sh server` prints the full endpoint map anytime. On this network the
address stays fixed (set a static DHCP lease so it never surprises you).

Service guides and Python clients are indexed in [docs/README.md](docs/README.md).

What you get: **chat with vision, a decision model, a reranker, multimodal
embeddings, image generation, speech-to-text and text-to-speech** behind one LAN
address, with Prometheus metrics and Grafana dashboards. Your own
AI backend with no API bills, no rate limits, no data leaving your LAN.
Good for private coding agents, RAG over your own documents and media, local voice
assistants, batch image generation, and routing/triage inside agent loops.

**Everything runs at once.** The chat model is coder390, a Qwen3.8-27B
post-train in NVFP4 W4A16 with text and image input. SGLang serves it with
DFlash2 speculative decoding, the full 262K context and tool calling. The
NVFP4 tier scores GPQA 170/198, MMLU 439/500 and LCB 91/100, and runs at about
53–61 tokens/s on the Spark. Decisions, embeddings and reranking run on
llama.cpp; images on stable-diffusion.cpp. The whole stack fits in the Spark's
128 GB unified memory with roughly 17 GiB to spare at idle.

## Install (on the Spark)

Needs Docker + Compose v2, git and curl, plus ~110 GB free disk (43 GB of models,
the rest Docker images). `install` checks the tooling and installs `uv` if needed.

1. Clone and configure:

   ```bash
   git clone https://github.com/malvavisc0/SparkStack.git && cd SparkStack
   cp .env.example .env    # put your Hugging Face token and LLM_API_KEY in .env
   ```

   (Token: huggingface.co → Settings → Access Tokens.)

2. **Accept the FLUX.3 license** with that same account at
   <https://huggingface.co/black-forest-labs/flux-3-image-klein> — required for the
   VAE (`ae.safetensors`). Approval is instant; without it `install` fails on the
   image model download.

3. Install:

   ```bash
   ./sparkstack.sh install
   ```

`install` downloads the models (~41 GB), builds the llama.cpp and
stable-diffusion.cpp images for the GB10 GPU, pulls the SGLang image, starts
everything, and prints the endpoint map — everything
your clients need. On this box it's `192.168.1.10`; check
<http://192.168.1.10:9190> (all cards green = healthy). After the first
`install` or `up`, run `./sparkstack.sh test` — it waits for each service to
finish loading, then fires a real request per endpoint (a chat completion, an
embedding, an actual image, a TTS clip) and prints the evidence. The first
request on every service pays lazy-init costs (CUDA graphs, JIT kernels);
`test` absorbs those so real traffic starts warm.

All services are up after install — chat, images, speech, embeddings, decisions,
status and logs, simultaneously. `up` starts everything by default; `down` stops
it. If you stopped something and want it back:

```bash
./sparkstack.sh up              # the whole stack
./sparkstack.sh up llm vision   # or individual profiles
```

## Use it (from any machine on the LAN)

Chat, embeddings, images and TTS are OpenAI-compatible — point any SDK or agent
at `http://<spark>:91xx` (speech-to-text uses its own multipart `/transcribe`
route; decisions uses `/v1/systemone`).

### Chat (vision + text) — `:9193`

Full guide: [docs/chat-api.md](docs/chat-api.md). The Python client is
[scripts/chat_client.py](scripts/chat_client.py).

```bash
curl http://192.168.1.10:9193/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "coder390",
    "messages": [{"role": "user", "content": "Explain unified memory in one paragraph."}]
  }'
```

Send images by adding an `image_url` content part. SGLang serves the full 262K
context with DFlash2 speculative decoding, tool calling and reasoning through
the model's chat template. The endpoint requires the `LLM_API_KEY` from `.env`
and returns HTTP 401 without it. With the OpenAI SDK, set
`base_url="http://<spark>:9193/v1"` and pass the same key as `api_key`.

### Speech-to-text — `:9191`

Full guide: [docs/speech-to-text-api.md](docs/speech-to-text-api.md). The Python
client is [scripts/transcribe_client.py](scripts/transcribe_client.py).

```bash
curl http://192.168.1.10:9191/transcribe -F file=@sample.wav -F language=en
```

WhisperX 3.8.6 with Whisper `large-v3`, batched on GPU, with diarization and
word timings.

### Text-to-speech — `:9192`

Full guide: [docs/text-to-speech-api.md](docs/text-to-speech-api.md). The Python
client is [scripts/tts_client.py](scripts/tts_client.py).

```bash
curl http://192.168.1.10:9192/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "SparkStack is up.", "voice": "vd_british_male"}' \
  --output hello.wav
```

`GET /speakers` lists all voices (British/German/Spanish, male/female). Any voice
can also be described inline with an `instruct` field instead of a preset id.
Formats: `wav`, `pcm`, `mp3`, or `zip` (audio + word timings).

### Embeddings — `:9195`

Full guide: [docs/embeddings-api.md](docs/embeddings-api.md). The Python client is
[scripts/embed_client.py](scripts/embed_client.py).

```bash
curl http://192.168.1.10:9195/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "hello world"}'
```

One 768-dim space shared by **text, images, video and audio** — embed a photo and
search for it with a text query.

### Reranking — `:9198`

Full guide: [docs/reranking-api.md](docs/reranking-api.md). The Python client is
[scripts/rerank_client.py](scripts/rerank_client.py).

```bash
curl http://192.168.1.10:9198/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "termination notice", "documents": ["The Labour Code requires 30 days written notice.", "Corporate tax rates for small enterprises."]}'
```

Qwen3-Reranker-0.6B scores the query and each document together. Use it after
vector search to put the best passages first. Three documents take about 30 ms.

### Decisions — `:9194`

Full guide: [docs/decisions-api.md](docs/decisions-api.md). The Python client is
[scripts/decide_client.py](scripts/decide_client.py).

```bash
curl http://192.168.1.10:9194/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}}'
```

One forward pass, zero output tokens: yes/no, best-of-N choices, or a score.
Cheap routing/triage for agents — use it to decide which tool or model to call next.

### Images — `:9196`

Full guide: [docs/image-api.md](docs/image-api.md). The Python client is
[scripts/flux_client.py](scripts/flux_client.py).

The client always expands the prompt through the chat model before rendering,
so it needs the `llm` profile. The raw API takes prompts as sent and expects a
full scene caption, not a short request.

```bash
curl -s http://192.168.1.10:9196/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Full-color outdoor photograph, a red fox stands in a snowy forest at dusk, its orange fur catching the last light, dark evergreens behind it and soft snow in the foreground."}' \
  | python3 -c 'import json,sys,base64; open("fox.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The service defaults are 24 steps with `dpm++2m` + `karras`, about 27 s per
1024² image. At 8 steps this build regularly mangled faces and hands; 24 is the
lowest step count that stayed clean in testing. Override per request with
`<sd_cpp_extra_args>{"sample_steps":16}</sd_cpp_extra_args>` (or `sample_params`
on the native API). `seed` is a top-level field, not part of `sample_params`.

**Editing** — `POST /v1/images/edits` (multipart) takes one or more `image`
fields. The first becomes the img2img init image (EDIT mode); all of them are
passed to FLUX.3 as references (up to 10). The service starts with
`--strength 0.9` (see `docker-compose.yaml`) because upstream's 0.75 default
under-edits large subjects. Override per request with
`<sd_cpp_extra_args>{"strength":0.6}</sd_cpp_extra_args>` in the prompt for a
subtler change:

```bash
curl -s http://192.168.1.10:9196/v1/images/edits \
  -F 'prompt=change the mug to bright blue, keep everything else the same' \
  -F image=@mug.png \
  | python3 -c 'import json,sys,base64; open("mug-blue.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The native async API also takes `strength` directly, per request. Caveat:
FLUX.3 Klein is img2img + multi-reference conditioning. `ref_images` alone
reproduces the reference rather than editing it, masked inpaint is unreliable,
and even at strength 0.9 the model may return the same subject in a new render
instead of following the caption. Use it for style and incidental variation,
not for replacing an object or changing identity. The Python client sends only
the extra images as references, never the init image twice.

sd-server also serves a web UI at `:9196` for interactive use.

**High-res and add-ons** — the native async API (`POST /sdcpp/v1/img_gen`,
poll `/sdcpp/v1/jobs/{id}`) unlocks everything the OpenAI route doesn't:

```bash
# two-pass generation: 1024 base → 1.5x RealESRGAN upscale → refine at base steps
# (~128 s for 1248x1824 from an 832x1216 base; latent upscalers smear faces)
curl http://192.168.1.10:9196/sdcpp/v1/img_gen \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Full-color outdoor photograph, a red fox stands in a snowy forest at dusk, orange fur catching the last light, dark evergreens behind it.",
       "width": 1024, "height": 1024,
       "hires": {"enabled": true, "upscaler": "RealESRGAN_x4",
                  "scale": 1.5, "denoising_strength": 0.3, "steps": 0}}'
```

- **LoRA**: drop FLUX-compatible LoRA files in `models/flux3/loras/` — they
  appear in `GET /sdapi/v1/loras` and attach per-request via the structured
  `lora` field (prompt-embedded `<lora:...>` tags are intentionally ignored).
- **Upscalers**: `RealESRGAN_x4` is installed in `models/flux3/upscalers/` and
  is the default for hires. If that file is missing, download
  `RealESRGAN_x4.pth` from `ai-forever/Real-ESRGAN` on Hugging Face into that
  directory. Drop other ESRGAN-family models there to use them by name instead
  of the built-in latent modes.

### Monitoring

- **<http://192.168.1.10:9190>** — Gatus: one card per service, green/red + latency.
  Red means "not running or unhealthy"; an on-demand profile that isn't started
  shows red.
- **<http://192.168.1.10:9197>** — Dozzle: live logs of every container, searchable.
  It also exposes an MCP endpoint at `/api/mcp` so agents can query the logs
  directly (requires the MCP initialize handshake).
- **<http://192.168.1.10:9199>** — Prometheus: SGLang and llama.cpp metrics, 30 day retention.
- **<http://192.168.1.10:9200>** — Grafana: the SparkStack dashboard. Anonymous
  viewers can read; the admin password comes from `GRAFANA_ADMIN_PASSWORD`.
  The GPU collector runs in the status profile. See [docs/metrics.md](docs/metrics.md).

## Operate it

Everything goes through one script:

| Command | What it does |
|---|---|
| `./sparkstack.sh install` | first-time setup: models, images, start, endpoint map |
| `./sparkstack.sh up [profiles]` | start services (default: the whole stack) |
| `./sparkstack.sh down` | stop the stack |
| `./sparkstack.sh test` | wait for loads, fire a real request per endpoint, print the evidence |
| `./sparkstack.sh status` | what's running |
| `./sparkstack.sh server` | print the LAN IP + endpoint map for your clients |
| `./sparkstack.sh logs <service>` | follow one service's logs |
| `./sparkstack.sh update` | rebuild images, restart what was running |
| `./sparkstack.sh download [group]` | re-fetch models: `all` `llm` `image` `utility` `audio` |
| `./sparkstack.sh build` | build the images only |

All profiles run at once: `audio`, `utility`, `status` (small, always-on) plus
`llm` and `vision` — the budget covers both simultaneously. A `tools` profile
(on-demand, never in `up`/`down`) provides a one-shot container with the HF CLI
for manual downloads: `docker compose --profile tools run --rm tools …`.

**Watchdog, one-time setup** — on unified memory an overrun doesn't error, it
freezes the machine. Normally idle (~35 GB available); if available memory drops
below 3 GiB it stops `vision` first, then `llm`, leaving the small services alone:

```bash
sudo cp scripts/watchdog.service scripts/watchdog.timer /etc/systemd/system/
echo "SPARKSTACK_DIR=$PWD" | sudo tee /etc/default/sparkstack
sudo systemctl enable --now watchdog.timer
```

## Reference

**Hardware**: designed and budgeted for the DGX Spark (GB10, 128 GB unified
memory, `sm_121`). Chat at NVFP4 W4A16 (about 22 GB of weights including the
DFlash2 draft) with a 262K context, the rest quantized or FP16 → ~104 GiB
resident at idle, ~17 GiB available; long prompts consume the KV reserve.

**Layout**: `models/` (weights, ~41 GB for the set the services use,
git-ignored — includes `flux3/loras/` and `flux3/upscalers/` add-on dirs,
scanned live) · `data/` (whisper cache, SGLang and FlashInfer caches, watchdog
log, GPU textfile, git-ignored) · `config/` (dashboards, TTS server + voices,
diffusion presets, prompt templates, Prometheus, Grafana, committed) ·
`.env` (your tokens, git-ignored).

**Security**: the chat endpoint requires `LLM_API_KEY` from `.env` as a bearer
token (`--api-key` on SGLang). Everything else is unauthenticated and open to
anyone who can reach the host. Dozzle mounts `docker.sock` and runs with
`DOZZLE_ENABLE_ACTIONS` + `DOZZLE_ENABLE_SHELL`, so it can start/stop containers
and `exec` a shell into them from the UI (the `:ro` on the socket mount is
cosmetic — it does not block API writes). Anyone who can reach Dozzle has the
same power as root on this host. Prometheus has no authentication and Grafana
allows anonymous viewing. Keep the box private or behind a VPN. Don't
port-forward the 919x block.

**Troubleshooting**

- First `up audio` is slow — WhisperX downloads its model then; subsequent starts are instant.
- First `up llm` takes ~2–3 minutes — SGLang loads 22 GB of weights plus the
  DFlash2 draft and captures decode CUDA graphs. Prefill graphs are disabled on
  purpose: they add ~3 minutes to every boot for no measurable gain at this
  concurrency. `test` waits for it. Don't cold-boot two heavy services at once —
  concurrent cold loads (e.g. TTS restarting while the chat model loads) can
  wedge GB10's unified memory. `up` then `test`, in that order.
- Long prompts are prefill-bound: ~1,100 tok/s, so a 128K prompt takes about
  2.5 minutes before the first token and a full 262K prompt about 4 minutes.
  Decode speed is unaffected.
- The SGLang cache lives in `data/sglang-cache`, so FlashInfer autotune results
  and compiled kernels survive restarts. Deleting that folder forces a re-tune.
- SGLang is pinned by image tag, not by a nightly digest; the model needs the
  DFlash2 and quantized-head support in the `dev-qwen38-27b-dflash2` build.
  Test a new tag before switching, and update the `image:` line in
  `docker-compose.yaml`.
- TTS 503 — model still loading; watch the status card turn green.
- Card red on :9190 — service down, or its profile isn't started (expected for `llm`/`vision` unless you started them).
- Machine feels starved during a 4K image run — the watchdog stops `vision` automatically below 3 GiB available; check `data/watchdog.log`.
