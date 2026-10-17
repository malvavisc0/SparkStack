# SparkStack

![SparkStack](docs/sparkstack-banner.png)

A complete private AI server for the **NVIDIA DGX Spark**. The Spark is the server;
everything else on your LAN — agents, apps, other machines — are its clients:

| Port | Endpoint | Serves |
|---|---|---|
| `:9193` | `POST /v1/chat/completions` | 74B chat + vision |
| `:9194` | `POST /v1/systemone` | one-shot decisions (yes/no/score) |
| `:9195` | `POST /v1/embeddings` | text/image/video/audio embeddings |
| `:9196` | `POST /v1/images/generations` | FLUX.3 image generation + editing |
| `:9191` | `POST /transcribe` | speech-to-text |
| `:9192` | `POST /v1/audio/speech` | text-to-speech, 10 voices |
| `:9190` | browser | status dashboard |
| `:9197` | browser | container logs |

Prefix any endpoint with your Spark's LAN address — the installer prints it, and
`./sparkstack.sh server` prints the full endpoint map anytime. On this network the
address stays fixed (set a static DHCP lease so it never surprises you).

What you get: **chat with vision, a decision model, multimodal embeddings, image
generation, speech-to-text and text-to-speech** on OpenAI-compatible endpoints.
Your own AI backend with no API bills, no rate limits, no data leaving your LAN.

**Everything runs at once.** The chat model is NVIDIA's own 24 GB build of
Nemotron-3.5-Bolt-74B — a compressed-and-retrained (quantization-aware
distillation) version of the 74B model, designed by NVIDIA to run alongside other
applications. It beats full-precision models of similar size on agentic benchmarks
(Terminal-Bench 51 vs Qwen3.6-35B's 44, GPQA 86 vs 83, BrowseComp 67 vs 49), served
by **vLLM** with b12x tensor-core kernels, 262K context and DFlash speculative
decoding (7 draft tokens). Everything else — decisions, embeddings — runs on
llama.cpp; images on stable-diffusion.cpp. The whole stack fits in the Spark's
128 GB unified memory with ~55 GB to spare.

## Install (on the Spark)

1. Clone and configure:

   ```bash
   git clone https://github.com/malvavisc0/SparkStack.git && cd SparkStack
   cp .env.example .env    # put your Hugging Face token in .env
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

`install` downloads the models (~50 GB), builds the images (llama.cpp,
stable-diffusion.cpp, and NVIDIA's recipe-pinned vLLM image with b12x kernels)
for the GB10 GPU, starts everything, and prints the endpoint map — everything
your clients need. On this box it's `192.168.1.10`; check
<http://192.168.1.10:9190> (all cards green = healthy). After `up`, run
`./sparkstack.sh test` — it waits for each service to finish loading, then fires
a real request per endpoint (a chat completion, an embedding, an actual image,
a TTS clip) and prints the evidence. The first request on every service pays
lazy-init costs (CUDA graphs, JIT kernels); `test` absorbs those so real
traffic starts warm.

All services are up after install — chat, images, speech, embeddings, decisions,
status and logs, simultaneously. `up` starts everything by default; `down` stops
it. If you stopped something and want it back:

```bash
./sparkstack.sh up              # the whole stack
./sparkstack.sh up llm vision   # or individual profiles
```

## Use it (from any machine on the LAN)

All endpoints are OpenAI-compatible — most agents, frameworks and clients work by
just changing the base URL to `http://<spark>:91xx`.

### Chat (vision + text) — `:9193`

```bash
curl http://192.168.1.10:9193/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3.5-bolt",
    "messages": [{"role": "user", "content": "Explain unified memory in one paragraph."}]
  }'
```

Send images by adding an `image_url` content part. 256K context, tool calling
and reasoning follow the model's chat template.

### Speech-to-text — `:9191`

```bash
curl http://192.168.1.10:9191/transcribe -F audio_file=@sample.wav
```

WhisperX `large-v3-turbo`, batched on GPU, with diarization.

### Text-to-speech — `:9192`

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

```bash
curl http://192.168.1.10:9195/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "hello world"}'
```

One 768-dim space shared by **text, images, video and audio** — embed a photo and
search for it with a text query.

### Decisions — `:9194`

```bash
curl http://192.168.1.10:9194/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"question": "Is the user asking about the weather?", "choices": ["yes", "no"]}'
```

One forward pass, zero output tokens: yes/no, best-of-N choices, or a score.
Cheap routing/triage for agents — use it to decide which tool or model to call next.

### Images — `:9196`

```bash
curl http://192.168.1.10:9196/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in a snowy forest, evening light"}' \
  --output fox.png
```

`POST /v1/images/edits` for editing with reference images (up to 10). FLUX.3
Image Klein at Q8_0: native 4K, 16 steps, tuned defaults — no parameters needed.
sd-server also serves a web UI at `:9196` for interactive use.

**High-res and add-ons** — the native async API (`POST /sdcpp/v1/img_gen`,
poll `/sdcpp/v1/jobs/{id}`) unlocks everything the OpenAI route doesn't:

```bash
# two-pass generation: 768 base → 2x latent upscale → 8-step refine (~25 s)
curl http://192.168.1.10:9196/sdcpp/v1/img_gen \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in a snowy forest, evening light",
       "width": 768, "height": 768,
       "hires": {"enabled": true, "upscaler": "Latent (nearest-exact)",
                  "scale": 2.0, "denoising_strength": 0.55, "steps": 8}}'
```

- **LoRA**: drop FLUX-compatible LoRA files in `models/flux3/loras/` — they
  appear in `GET /sdapi/v1/loras` and attach per-request via the structured
  `lora` field (prompt-embedded `<lora:...>` tags are intentionally ignored).
- **Upscalers**: drop ESRGAN-family models in `models/flux3/upscalers/` to use
  them by name as `hires.upscaler` instead of the built-in latent modes.

### Monitoring

- **<http://192.168.1.10:9190>** — Gatus: one card per service, green/red + latency.
  Red means "not running or unhealthy"; an on-demand profile that isn't started
  shows red.
- **<http://192.168.1.10:9197>** — Dozzle: live logs of every container, searchable.

## Operate it

Everything goes through one script:

| Command | What it does |
|---|---|
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
freezes the machine. Normally idle (~55 GB headroom); if free memory ever drops
below 3 GiB it stops `vision` first, then `llm`, leaving the small services alone:

```bash
sudo cp scripts/watchdog.service scripts/watchdog.timer /etc/systemd/system/
echo "SPARKSTACK_DIR=$PWD" | sudo tee /etc/default/sparkstack
sudo systemctl enable --now watchdog.timer
```

## Reference

**Hardware**: designed and budgeted for the DGX Spark (GB10, 128 GB unified
memory, `sm_121`). Chat at NVIDIA's native 2-bit (24 GB build), everything else
Q8_0 → ~60 GiB resident, ~55 GiB headroom. A full-precision Q8_0 chat fallback
lives in `models/nemotron/Q8_0/` — swap the `-m` path in `docker-compose.yaml`
to use it (needs the vision profile stopped).

**Layout**: `models/` (weights, ~96 GB, git-ignored — includes `flux3/loras/`
and `flux3/upscalers/` add-on dirs, scanned live) · `data/` (whisper cache +
watchdog log, git-ignored) · `config/` (dashboards, TTS server + voices,
diffusion presets, committed) · `.env` (your token, git-ignored).

**Security**: no auth anywhere — it's for a trusted LAN. Anyone who can reach the
host can call every model; Dozzle additionally mounts `docker.sock` read-only.
Keep the box private or behind a VPN. Don't port-forward the 919x block.

**Troubleshooting**

- First `up audio` is slow — WhisperX downloads its model then; subsequent starts are instant.
- First `up llm` takes ~7 minutes — vLLM cold-loads 24 GB of weights (plus the
  DFlash drafter); `test` waits for it. Don't cold-boot two heavy services at
  once — concurrent cold loads (e.g. TTS restarting while vLLM loads) can
  wedge GB10's unified memory. `up` then `test`, in that order.
- The vLLM image is pinned to the digest NVIDIA's recipe specifies (rolling
  nightlies segfault in the b12x MoE kernels). When NVIDIA updates the recipe,
  update the `FROM` digest in `Dockerfile` (marked with a comment).
- TTS 503 — model still loading; watch the status card turn green.
- Card red on :9190 — service down, or its profile isn't started (expected for `llm`/`vision` unless you started them).
- Machine feels starved during a 4K image run — the watchdog stops `vision` automatically below 3 GiB free; check `data/watchdog.log`.
