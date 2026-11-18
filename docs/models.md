# Models and hardware

SparkStack runs a fixed model set sized for the DGX Spark's 128 GB unified
memory. `./sparkstack.sh install` downloads the weights into `models/`; WhisperX
loads its models lazily into `data/whisper-data`.

## Model stack

| Role | Model | Details | License | Source |
|---|---|---|---|---|
| Chat + vision (`llm`, :9193) | Qwen3.8-27B **Coder390** | 27B, NVFP4 W4A16, ~20 GB; 262K context, text + image, reasoning and tool calling | Apache-2.0 | [nerkyor/Qwen3.8-27B-Coder390…DFlash2](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2) |
| Speculative draft | **DFlash2-FP8** | 2.3 GB, FP8; served with the chat model from `NVFP4/W4A16/DFlash2-FP8` | Apache-2.0 | same repository |
| Decisions (`utility`, :9194) | **d1-2B** | 1.9B, Q8_0 + mmproj, ~2.0 GB; single-pass decisions, 32K context, 16 languages | LFM Open License v1.0 | [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) |
| Embeddings (`utility`, :9195) | **EmbeddingGemma 2** | 744M, Q8_0 + mmproj, ~825 MB; 768-dim joint text/image/video/audio space | Apache-2.0 | [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF) |
| Reranking (`utility`, :9198) | **Qwen3-Reranker-0.6B** | 0.6B, Q8_0, ~610 MB; cross-encoder query–document scoring | Apache-2.0 | [Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp) |
| Images (`vision`, :9196) | **FLUX.3 Image Klein** | 5.5B, Q8_0, ~6.1 GB; text-to-image and multi-reference editing | FLUX Kommunity License (gated) | [unsloth/flux-3-image-klein-GGUF](https://huggingface.co/unsloth/flux-3-image-klein-GGUF) |
| Image text encoder | **Qwen3-VL-4B-Instruct** | 4.4B, Q8_0, ~4.3 GB; conditions and expands FLUX prompts | Apache-2.0 | [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) |
| Image VAE | **FLUX.3 `ae.safetensors`** | 336 MB; decodes the FLUX latents | FLUX Kommunity License (gated) | [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) |
| Speech-to-text (`audio`, :9191) | **Whisper large-v3** | 1.55B, CTranslate2, ~2.9 GB; multilingual ASR with word timings | MIT (Systran) | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |
| Speaker diarization | **pyannote speaker-diarization-3.1** | segmentation-3.0 + WeSpeaker ResNet34, ~32 MB; speaker labels | MIT (gated) | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| Text-to-speech (`audio`, :9192) | **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | 1.9B, ~4.3 GB; 10 preset voices plus inline voice design | Apache-2.0 | [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) |
| Hires upscaler (optional) | **RealESRGAN_x4** | 64 MB; default for the `hires` route | BSD-3-Clause | [ai-forever/Real-ESRGAN](https://huggingface.co/ai-forever/Real-ESRGAN) |

Footprints are on-disk sizes. The chat target is ~20 GB and the DFlash2 draft
adds 2.3 GB, for ~22 GB of chat weights. The full download is ~41 GB.

WhisperX downloads language-specific wav2vec2 alignment models into
`data/whisper-data` on demand, so its cache grows with the languages you use
(the current cache is ~13 GB after multilingual testing). FLUX.3 and pyannote
are gated: accept their terms with the same HF account as `HF_TOKEN`. Optional
FLUX-compatible LoRAs and alternate ESRGAN upscalers go in `models/flux3/loras/`
and `models/flux3/upscalers/`; both directories are scanned live.

## Hardware budget

Designed and budgeted for the DGX Spark (GB10, 128 GB unified memory, `sm_121`).
Chat at NVFP4 W4A16 (about 22 GB of weights including the DFlash2 draft) with a
262K context, the rest quantized or FP16 → ~104 GiB resident at idle, ~17 GiB
available; long prompts consume the KV reserve. The watchdog described in
[operations.md](operations.md) stops the vision profile first, then the LLM,
when available memory falls below 3 GiB.

## Storage layout

`models/` holds the weights (~41 GB for the set the services use). It is
git-ignored and includes the `flux3/loras/` and `flux3/upscalers/` add-on
directories. `data/` holds the Whisper cache, SGLang and FlashInfer caches, the
watchdog log and the GPU textfile; it is also git-ignored. `config/` holds the
dashboards, TTS server and voices, diffusion presets, prompt templates,
Prometheus and Grafana configuration and is committed. `.env` holds your tokens
and is git-ignored.

## Download groups

```bash
./sparkstack.sh download all      # everything
./sparkstack.sh download llm      # chat + DFlash2 draft
./sparkstack.sh download image    # FLUX.3, text encoder, VAE
./sparkstack.sh download utility  # decisions, embeddings, reranker
./sparkstack.sh download audio    # Qwen3-TTS
```

`download` verifies the files each service references and fails with the missing
paths. Re-running a group only fetches what is absent or changed.
