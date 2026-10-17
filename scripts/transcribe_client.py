#!/usr/bin/env python3
"""Transcribe audio with the SparkStack WhisperX service."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from client_support import fail, request_multipart_json

DEFAULT_API = os.environ.get("SPARKSTACK_STT_URL", "http://localhost:9191")


def timestamp(seconds: float) -> str:
    """Format seconds as minutes and seconds."""
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds % 60:05.2f}"


def print_summary(
    result: dict[str, Any],
    elapsed: float,
    words: bool,
    text_only: bool,
) -> None:
    """Print the transcript with optional word timings."""
    segments: list[dict[str, Any]] = result.get("segments") or []
    if text_only:
        print(" ".join(segment.get("text", "").strip() for segment in segments))
        return
    print(f"language: {result.get('language', '?')}")
    if result.get("num_speakers") is not None:
        print(f"speakers: {result['num_speakers']}")
    if segments:
        print(f"duration: {max(segment['end'] for segment in segments):.2f}s")
    for segment in segments:
        speaker = f"{segment.get('speaker')}: " if segment.get("speaker") else ""
        text = segment.get("text", "").strip()
        print(
            f"[{timestamp(segment['start'])} - {timestamp(segment['end'])}] "
            f"{speaker}{text}"
        )
        if words:
            for word in segment.get("words") or []:
                score = word.get("score")
                score_text = f" score={score:.3f}" if score is not None else ""
                print(
                    f"    {timestamp(word['start'])} - "
                    f"{timestamp(word['end'])} {word.get('word', '').strip()}"
                    f"{score_text}"
                )
    print(f"transcribed in {elapsed:.2f}s")


def main() -> None:
    """Parse options and transcribe the audio file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="audio file to transcribe")
    parser.add_argument(
        "--language",
        default="auto",
        help="language code, or auto",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="exact number of speakers",
    )
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "--words",
        action="store_true",
        help="print word timings and scores",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="print the plain transcript",
    )
    parser.add_argument("--json", action="store_true", help="print the raw response")
    parser.add_argument("--out", type=Path, help="save the raw response as JSON")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    fields: dict[str, str | int | None] = {
        "language": args.language,
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
    }
    url = args.api.rstrip("/") + "/transcribe"
    started = time.monotonic()
    result: dict[str, Any] = request_multipart_json(
        url,
        fields,
        args.audio,
        args.timeout,
    )
    elapsed = time.monotonic() - started

    if result.get("status") not in (None, "success"):
        fail(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result, elapsed, args.words, args.text_only)


if __name__ == "__main__":
    main()
