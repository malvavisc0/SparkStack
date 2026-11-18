# Operations

Everything runs through `./sparkstack.sh`. The script is safe to re-run: `up`
only starts what is stopped, `update` rebuilds images and recreates the profiles
that were already running.

## Commands

| Command | What it does |
|---|---|
| `./sparkstack.sh install` | first-time setup: models, images, start, endpoint map |
| `./sparkstack.sh up [profiles]` | start services (default: the whole stack) |
| `./sparkstack.sh down` | stop the stack |
| `./sparkstack.sh test` | wait for loads, fire a real request per endpoint, print the evidence |
| `./sparkstack.sh status` | what's running |
| `./sparkstack.sh server` | print the LAN address and endpoint map for your clients |
| `./sparkstack.sh logs <service>` | follow one service's logs |
| `./sparkstack.sh update` | rebuild images, restart what was running |
| `./sparkstack.sh download [group]` | re-fetch models: `all` `llm` `image` `utility` `audio` |
| `./sparkstack.sh build` | build the images only |

## Profiles

All profiles run at once: `audio`, `utility`, `status` (small, always-on) plus
`llm` and `vision`. The memory budget covers both heavy profiles
simultaneously. A `tools` profile (on-demand, never in `up`/`down`) provides a
one-shot container with the HF CLI for manual downloads:

```bash
docker compose --profile tools run --rm tools hf download <repo> --local-dir /models/<name>
```

`up` starts everything by default; pass profile names to start a subset:

```bash
./sparkstack.sh up              # the whole stack
./sparkstack.sh up llm vision   # individual profiles
```

## Startup and caches

- The first `up audio` is slow: WhisperX downloads its model, then later starts
  are instant.
- The first `up llm` takes ~2–3 minutes. SGLang loads ~22 GB of weights plus the
  DFlash2 draft and captures decode CUDA graphs. Prefill graphs are disabled on
  purpose because they add ~3 minutes to every boot for no measurable gain at
  this concurrency.
- Do not cold-boot two heavy services at once. Concurrent cold loads (for
  example TTS restarting while the chat model loads) can wedge GB10's unified
  memory. Run `up`, then `test`, in that order.
- Long prompts are prefill-bound at ~1,100 tok/s. A 128K prompt takes about 2.5
  minutes before the first token; a full 262K prompt about 4 minutes. Decode
  speed is unaffected.
- The SGLang cache lives in `data/sglang-cache`, so FlashInfer autotune results
  and compiled kernels survive restarts. Deleting that folder forces a re-tune.
- SGLang is pinned by image tag, not by a nightly digest. The model needs the
  DFlash2 and quantized-head support in the `dev-qwen38-27b-dflash2` build. Test
  a new tag before switching it in `docker-compose.yaml`.

## Watchdog

On unified memory an overrun does not error, it freezes the machine. At idle the
stack leaves about 35 GB available; if available memory drops below 3 GiB the
watchdog stops `vision` first, then `llm`, leaving the small services alone.

One-time setup:

```bash
sudo cp scripts/watchdog.service scripts/watchdog.timer /etc/systemd/system/
echo "SPARKSTACK_DIR=$PWD" | sudo tee /etc/default/sparkstack
sudo systemctl enable --now watchdog.timer
```

The log is written to `data/watchdog.log`.

## Troubleshooting

- **TTS returns 503** — the model is still loading; watch the status card turn
  green.
- **A status card is red** — the service is down, or its profile is not started
  (expected for `llm`/`vision` unless you started them).
- **First request is slow on every service** — lazy initialization (CUDA graphs,
  JIT kernels, model load) is paid once. `./sparkstack.sh test` absorbs those so
  real traffic starts warm.
- **Machine feels starved during a 4K image run** — the watchdog stops `vision`
  automatically below 3 GiB available; check `data/watchdog.log`.
- **SGLang fails after changing the image tag** — the model requires the
  DFlash2 and quantized-head support in the `dev-qwen38-27b-dflash2` build.
  Revert the tag and test changes deliberately.
- **WhisperX speaker labels fail** — accept the gated
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  terms with the HF account behind `HF_TOKEN`.
