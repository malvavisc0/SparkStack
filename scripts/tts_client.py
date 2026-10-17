#!/usr/bin/env python3
"""Generate speech with the SparkStack Qwen3-TTS service."""

import argparse
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

from client_support import fail, request_bytes, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_TTS_URL", "http://localhost:9192")
LANGUAGES: tuple[str, ...] = (
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
)


def canonical_language(value: str) -> str | None:
    """Return the server language name, or None when unsupported."""
    name = value.strip().lower()
    if name not in LANGUAGES:
        return None
    return "auto" if name == "auto" else name.title()


def wav_info(data: bytes) -> tuple[int, int, int, float] | None:
    """Return sample rate, channels, bits and duration for WAV data."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    position = 12
    rate = channels = bits = None
    data_size = None
    while position + 8 <= len(data):
        chunk = data[position : position + 4]
        size = struct.unpack("<I", data[position + 4 : position + 8])[0]
        start = position + 8
        if chunk == b"fmt " and start + 16 <= len(data):
            _format, channels, rate, _byte_rate, _align, bits = struct.unpack(
                "<HHIIHH",
                data[start : start + 16],
            )
        elif chunk == b"data":
            if size in (0, 0xFFFFFFFF):
                data_size = len(data) - start
            else:
                data_size = min(size, len(data) - start)
            break
        if size in (0, 0xFFFFFFFF):
            break
        position = start + size + (size & 1)
    if rate and channels and bits and data_size is not None:
        duration = data_size / (rate * channels * bits / 8)
        return rate, channels, bits, duration
    return None


def default_out(response_format: str) -> Path:
    """Return a timestamped output path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(f"speech-{stamp}.{response_format}")


def main() -> None:
    """Parse options and generate the speech file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help='text to speak; use "-" for stdin')
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--voice", default="vd_british_male")
    parser.add_argument(
        "--format",
        default="wav",
        choices=("wav", "mp3", "pcm", "zip"),
        dest="response_format",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--instruct",
        help="voice direction, for example 'speak slowly and warmly'",
    )
    parser.add_argument(
        "--language",
        help="auto, Chinese, English, French, German, Italian, Japanese, "
        "Korean, Portuguese, Russian or Spanish",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    base = args.api.rstrip("/")
    if args.list_voices:
        voices: list[str] = request_json(
            base + "/speakers",
            method="GET",
            timeout=args.timeout,
        )
        for voice in voices:
            print(voice)
        return
    if args.text_file:
        try:
            text = args.text_file.read_text()
        except OSError as error:
            fail(f"cannot read {args.text_file}: {error}")
    elif args.text == "-":
        text = sys.stdin.read()
    else:
        text = args.text
    if not text or not text.strip():
        parser.error("provide text, --text-file, or text on stdin")
    language = canonical_language(args.language) if args.language else None
    if args.language and language is None:
        parser.error(f"unsupported language: {args.language}")

    payload: dict[str, Any] = {
        "input": text,
        "voice": args.voice,
        "response_format": args.response_format,
        "speed": args.speed,
    }
    if language:
        payload["language"] = language
    if args.instruct:
        payload["instruct"] = args.instruct
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens

    started = time.monotonic()
    data = request_bytes(
        base + "/v1/audio/speech",
        payload=payload,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - started

    info = wav_info(data) if args.response_format == "wav" else None
    if args.response_format == "wav" and (not info or info[3] <= 0):
        fail(
            "the service returned empty audio; check the voice name and "
            "--language (use English, German, Spanish, ...)"
        )

    out = args.out or default_out(args.response_format)
    if not out.suffix:
        out = out.with_suffix("." + args.response_format)
    out.write_bytes(data)

    details = [f"{len(data)} bytes", f"{elapsed:.2f}s render"]
    if info:
        rate, channels, bits, duration = info
        details.insert(0, f"{duration:.2f}s audio")
        details.append(f"{rate} Hz, {channels} ch, {bits} bit")
    print(f"saved {out} ({', '.join(details)})")


if __name__ == "__main__":
    main()
