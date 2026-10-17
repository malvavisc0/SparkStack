"""OpenAI-compatible TTS server for Qwen3-TTS-12Hz-1.7B-VoiceDesign.

Voices are described in a JSON registry: {"voice_id": {"instruct": "...", "language": "..."}}.
The instruct text fully describes the voice; no reference audio is needed.
"""

import argparse
import asyncio
import functools
import io
import json
import logging
import queue
import struct
import subprocess
import sys
import threading
import zipfile
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from pydub import AudioSegment

sys.path.insert(0, "/app")
from faster_qwen3_tts.model import FasterQwen3TTS

logger = logging.getLogger(__name__)

MODEL_LOCK = threading.Lock()
AUDIO_DONE = object()
VOICE_CREATED = 1686935002
CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "zip": "application/zip",
}


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "vd_british_male"
    response_format: str = "wav"
    speed: float = 1.0
    language: str | None = None
    instruct: str | None = None
    max_new_tokens: int | None = None


class VoiceRegistry:
    """Voice definitions, hot-reloaded when the registry file changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mtime = path.stat().st_mtime if path.exists() else 0.0
        self.voices = self.read()

    def read(self) -> dict[str, dict[str, Any]]:
        source = self.path
        if not source.exists():
            presets = source.with_name("voicedesign_voices.presets.json")
            if not presets.exists():
                raise FileNotFoundError(f"no registry at {source} and no presets at {presets}")
            logger.warning("%s missing, using presets %s", source, presets)
            source = presets
        with source.open(encoding="utf-8") as fh:
            return json.load(fh)

    def current(self) -> dict[str, dict[str, Any]]:
        mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
        if mtime > self.mtime:
            self.voices = self.read()
            self.mtime = mtime
            logger.info("Reloaded %d voices from %s", len(self.voices), self.path)
        return self.voices

    def resolve(self, name: str, has_request_instruct: bool) -> dict[str, Any]:
        cfg = self.current().get(name)
        if cfg is not None:
            return cfg
        if has_request_instruct:
            logger.info("Voice %r not registered; using the request's instruct", name)
            return {}
        raise HTTPException(
            status_code=404,
            detail=f"Voice {name!r} is not registered and the request carried no 'instruct'."
            f" Known voices: {sorted(self.current())}",
        )


def load_model(model_path: str, device: str, max_seq_len: int) -> FasterQwen3TTS:
    logger.info("Loading VoiceDesign model %s", model_path)
    model = FasterQwen3TTS.from_pretrained(
        model_path,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_seq_len=max_seq_len,
    )
    logger.info("Model ready. Sample rate: %d Hz", model.sample_rate)
    logger.info("Warming up CUDA graphs...")
    warmup = model.generate_voice_design_streaming(
        text="Warmup.", instruct="Warmup.", language="English"
    )
    for _ in warmup:
        pass
    logger.info("Warmup complete")
    return model


def generation_params(
    req: SpeechRequest, voice_cfg: dict[str, Any], max_new_tokens: int
) -> dict[str, Any]:
    # A registered voice's instruct is its identity; a request-level instruct is per-line
    # direction. Combine them, identity first, so direction never discards the character.
    identity = str(voice_cfg.get("instruct", "") or "").strip()
    direction = str(req.instruct or "").strip()
    if identity and direction and direction != identity:
        instruct: str = f"{identity} {direction}"
    else:
        instruct = direction or identity
    params: dict[str, Any] = {
        "text": req.input,
        "instruct": instruct,
        "language": req.language or voice_cfg.get("language", "English"),
        "max_new_tokens": req.max_new_tokens
        or int(voice_cfg.get("max_new_tokens", max_new_tokens)),
    }
    # Per-voice sampling overrides — generate_voice_design() accepts no seed.
    for key in ("temperature", "top_p", "top_k"):
        if voice_cfg.get(key) is not None:
            params[key] = float(voice_cfg[key]) if key != "top_k" else int(voice_cfg[key])
    return params


def pcm16(audio: np.ndarray) -> bytes:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def wav_header(sample_rate: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", 0xFFFFFFFF,
    )


def mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    segment = AudioSegment(pcm16(audio), frame_rate=sample_rate, sample_width=2, channels=1)
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def apply_tempo(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    if speed == 1.0:
        return audio
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-filter:a", f"atempo={speed}",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
        ],
        input=audio.tobytes(),
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def spawn_tempo_stream(sample_rate: int, speed: float) -> tuple[subprocess.Popen, Any, Any]:
    """Spawn ffmpeg for tempo adjustment during streaming; caller owns the process."""
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-filter:a", f"atempo={speed}",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None and ffmpeg.stdout is not None
    return ffmpeg, ffmpeg.stdin, ffmpeg.stdout


async def pcm_stream(
    model: FasterQwen3TTS,
    params: dict[str, Any],
    speed: float,
    sample_rate: int,
) -> AsyncIterator[bytes]:
    chunks: queue.Queue[Any] = queue.Queue()
    ffmpeg_stdin = None
    ffmpeg_stdout = None
    if speed != 1.0:
        _ffmpeg, ffmpeg_stdin, ffmpeg_stdout = spawn_tempo_stream(sample_rate, speed)

        def drain() -> None:
            for data in iter(lambda: ffmpeg_stdout.read(4096), b""):
                chunks.put(data)
            chunks.put(AUDIO_DONE)

        threading.Thread(target=drain, daemon=True).start()

    def produce() -> None:
        try:
            with MODEL_LOCK:
                for chunk, _sr, _timing in model.generate_voice_design_streaming(**params):
                    raw = pcm16(chunk)
                    if ffmpeg_stdin is not None:
                        ffmpeg_stdin.write(raw)
                        ffmpeg_stdin.flush()
                    else:
                        chunks.put(raw)
        # The model generator can raise anything (torch/CUDA/runtime); forward it to
        # the awaiting client instead of letting the producer thread die silently.
        except Exception as exc:
            chunks.put(exc)
        finally:
            if ffmpeg_stdin is not None:
                ffmpeg_stdin.close()
            else:
                chunks.put(AUDIO_DONE)

    threading.Thread(target=produce, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, chunks.get)
        if item is AUDIO_DONE:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def create_app(
    *,
    model: FasterQwen3TTS,
    sample_rate: int,
    registry: VoiceRegistry,
    max_new_tokens: int,
) -> FastAPI:
    app = FastAPI()

    @functools.cache
    def aligner() -> Any:
        try:
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="qwen-asr is not installed") from exc
        logger.info("Loading Qwen3-ForcedAligner-0.6B...")
        loaded = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda"
        )
        return loaded

    def voice_catalog() -> list[dict[str, Any]]:
        return [
            {"id": name, "object": "model", "created": VOICE_CREATED, "owned_by": "qwen"}
            for name in registry.current()
        ]

    def render_buffered(
        req: SpeechRequest, params: dict[str, Any], voice_cfg: dict[str, Any]
    ) -> Response:
        with MODEL_LOCK:
            audio_arrays, sr = model.generate_voice_design(**params)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        audio = apply_tempo(audio, req.speed, sr)
        if req.response_format == "zip":
            entries = aligner().align(
                audio=(audio, sr), text=req.input, language=voice_cfg.get("language", "Auto")
            )
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("audio.mp3", mp3_bytes(audio, sr))
                timer_json = json.dumps([asdict(e) for e in entries], ensure_ascii=False)
                zf.writestr("timer.json", timer_json)
            return Response(content=buffer.getvalue(), media_type="application/zip")
        return Response(content=mp3_bytes(audio, sr), media_type="audio/mpeg")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def create_speech(req: SpeechRequest) -> Response:
        req.input = req.input.strip()
        if not req.input:
            raise HTTPException(status_code=400, detail="'input' text is empty")
        voice_cfg = registry.resolve(req.voice, bool((req.instruct or "").strip()))
        params = generation_params(req, voice_cfg, max_new_tokens)
        fmt = req.response_format.lower()
        if fmt not in CONTENT_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt!r}")

        if fmt in ("mp3", "zip"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: render_buffered(req, params, voice_cfg))

        async def audio_stream() -> AsyncIterator[bytes]:
            if fmt == "wav":
                yield wav_header(sample_rate)
            async for chunk in pcm_stream(model, params, req.speed, sample_rate):
                yield chunk

        return StreamingResponse(audio_stream(), media_type=CONTENT_TYPES[fmt])

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": voice_catalog()}

    @app.get("/v1/audio/voices")
    async def list_voices() -> dict[str, Any]:
        return {"object": "list", "data": voice_catalog()}

    @app.get("/v1/audio/models")
    async def list_audio_models() -> dict[str, Any]:
        return {"object": "list", "data": voice_catalog()}

    @app.get("/speakers")
    async def speakers() -> list[str]:
        return list(registry.current())

    @app.options("/{path:path}")
    async def options(path: str) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3-TTS-VoiceDesign")
    parser.add_argument("--voices", default="/config/voicedesign_voices.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    model = load_model(args.model, args.device, args.max_seq_len)
    registry = VoiceRegistry(Path(args.voices))
    app = create_app(
        model=model,
        sample_rate=model.sample_rate,
        registry=registry,
        max_new_tokens=args.max_seq_len,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
