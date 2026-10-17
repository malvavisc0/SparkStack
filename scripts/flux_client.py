#!/usr/bin/env python3
"""Generate, edit and upscale images with the SparkStack FLUX.3 service."""

import argparse
import base64
import json
import os
import struct
import time
import urllib.parse
from pathlib import Path
from typing import Any

from client_support import fail, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_IMAGE_URL", "http://localhost:9196")


def wait_for_job(
    api: str,
    job: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Poll a native image job until it finishes."""
    url = urllib.parse.urljoin(api.rstrip("/") + "/", str(job["poll_url"]))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state: dict[str, Any] = request_json(url, method="GET", timeout=30)
        status = state.get("status")
        if status == "completed":
            return state
        if status == "failed":
            detail = state.get("error") or json.dumps(state)[:500]
            fail(f"job failed: {detail}")
        time.sleep(1)
    fail(f"job {job.get('id', '?')} did not finish within {timeout:.0f}s")


def image_size(path: Path) -> tuple[int, int] | None:
    """Read the width and height from a PNG or JPEG file."""
    with path.open("rb") as handle:
        header = handle.read(26)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", header[16:24])
        if header[:2] != b"\xff\xd8":
            return None
        handle.seek(2)
        while True:
            byte = handle.read(1)
            while byte and byte != b"\xff":
                byte = handle.read(1)
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                return None
            code = marker[0]
            if code in (0x01, 0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) < 2:
                return None
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                return None
            if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                data = handle.read(5)
                if len(data) < 5:
                    return None
                height, width = struct.unpack(">HH", data[1:5])
                return width, height
            handle.seek(length - 2, 1)


def png_size(data: bytes) -> tuple[int, int] | None:
    """Read the width and height from PNG data."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def default_out(command: str) -> Path:
    """Return a timestamped output path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(f"flux-{command}-{stamp}.png")


def sample_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Build the optional sample settings."""
    settings: dict[str, Any] = {}
    if args.steps is not None:
        settings["sample_steps"] = args.steps
    if args.sampler:
        settings["sample_method"] = args.sampler
    if args.scheduler:
        settings["scheduler"] = args.scheduler
    if args.seed is not None:
        settings["seed"] = args.seed
    return settings


def build_payload(
    args: argparse.Namespace,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Build a native image request."""
    payload: dict[str, Any] = {
        "prompt": args.prompt,
        "width": width,
        "height": height,
    }
    if args.negative:
        payload["negative_prompt"] = args.negative
    settings = sample_settings(args)
    if settings:
        payload["sample_params"] = settings
    return payload


def run_job(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Submit a native job, wait for it and save the image."""
    started = time.monotonic()
    url = f"{args.api.rstrip('/')}/sdcpp/v1/img_gen"
    job: dict[str, Any] = request_json(url, payload=payload, timeout=60)
    state = wait_for_job(args.api, job, args.timeout)
    elapsed = time.monotonic() - started
    images = (state.get("result") or {}).get("images") or []
    if not images:
        fail(f"no image in response: {json.dumps(state)[:500]}")
    data = base64.b64decode(images[0]["b64_json"])
    out = args.out or default_out(args.command)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    size = png_size(data)
    details = f"{size[0]}x{size[1]}, " if size else ""
    print(f"saved {out} ({details}{elapsed:.1f}s)")


def cmd_generate(args: argparse.Namespace) -> None:
    """Run a text-to-image job."""
    run_job(args, build_payload(args, args.width, args.height))


def cmd_edit(args: argparse.Namespace) -> None:
    """Run an image-to-image job."""
    size = image_size(args.images[0])
    width = args.width or (size[0] if size else 1024)
    height = args.height or (size[1] if size else 1024)
    payload = build_payload(args, width, height)
    encoded = [
        base64.b64encode(path.read_bytes()).decode("ascii") for path in args.images
    ]
    payload["init_image"] = encoded[0]
    payload["ref_images"] = encoded
    if args.strength is not None:
        payload["strength"] = args.strength
    run_job(args, payload)


def cmd_hires(args: argparse.Namespace) -> None:
    """Run a two-pass hires job."""
    payload = build_payload(args, args.width, args.height)
    payload["hires"] = {
        "enabled": True,
        "upscaler": args.upscaler,
        "scale": args.scale,
        "denoising_strength": args.denoise,
        "steps": args.hires_steps,
    }
    run_job(args, payload)


def add_common(parser: argparse.ArgumentParser) -> None:
    """Add options shared by all image commands."""
    parser.add_argument("prompt", help="what the image should show")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--negative", default="", help="things to avoid")
    parser.add_argument(
        "--steps",
        type=int,
        help="sampling steps (server default: 8)",
    )
    parser.add_argument(
        "--sampler",
        help="sample method, for example dpm++2m",
    )
    parser.add_argument(
        "--scheduler",
        help="sigma scheduler, for example karras",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="fixed seed for repeatable output",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="job timeout in seconds",
    )
    parser.add_argument("--out", type=Path, help="output PNG path")


def main() -> None:
    """Parse options and run the selected image command."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="text to image")
    add_common(generate)
    generate.add_argument("--width", type=int, default=1024)
    generate.add_argument("--height", type=int, default=1024)
    generate.set_defaults(func=cmd_generate)

    edit = sub.add_parser("edit", help="img2img edit from one or more images")
    add_common(edit)
    edit.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="init image followed by optional references",
    )
    edit.add_argument(
        "--strength",
        type=float,
        help="repaint amount, 0 to 1 (server default: 0.9)",
    )
    edit.add_argument("--width", type=int)
    edit.add_argument("--height", type=int)
    edit.set_defaults(func=cmd_edit)

    hires = sub.add_parser(
        "hires",
        help="two-pass generation with a latent upscale",
    )
    add_common(hires)
    hires.add_argument("--width", type=int, default=1024)
    hires.add_argument("--height", type=int, default=1024)
    hires.add_argument("--scale", type=float, default=1.5)
    hires.add_argument("--denoise", type=float, default=0.45)
    hires.add_argument("--hires-steps", type=int, default=8)
    hires.add_argument("--upscaler", default="Latent (nearest-exact)")
    hires.set_defaults(func=cmd_hires)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
