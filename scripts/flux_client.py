#!/usr/bin/env python3
"""Generate, edit and upscale images with mandatory prompt expansion."""

import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from client_support import fail, image_part, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_IMAGE_URL", "http://localhost:9196")
DEFAULT_CHAT_API = os.environ.get("SPARKSTACK_CHAT_URL", "http://localhost:9193")


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


def load_prompt(kind: str) -> str:
    """Read a prompt template from the config directory."""
    path = (
        Path(__file__).resolve().parent.parent / "config" / "image" / f"prompt_{kind}.txt"
    )
    try:
        text = path.read_text()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    if not text.strip():
        fail(f"empty prompt template: {path}")
    return text.strip()


def upsample_prompt(
    args: argparse.Namespace,
    prompt: str,
    images: list[Path],
) -> str:
    """Expand a request into a full caption with the local chat model."""
    if args.no_expand:
        return prompt
    kind = "i2i" if images else "t2i"
    user: Any = prompt
    if kind == "t2i":
        user = f"TARGET CANVAS: {args.width}:{args.height}\n{prompt}"
    else:
        user = [{"type": "text", "text": prompt}, image_part(images[0])]
    payload: dict[str, Any] = {
        "model": args.upsample_model,
        "messages": [
            {"role": "system", "content": load_prompt(kind)},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 1200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response: dict[str, Any] = request_json(
        args.chat_api.rstrip("/") + "/v1/chat/completions",
        payload=payload,
        timeout=args.timeout,
    )
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        fail("prompt upsampler returned an empty caption")
    caption = content.strip()
    if args.show_prompt:
        print(f"upsampled prompt: {caption}", file=sys.stderr)
    return caption


def sample_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Build the optional sample settings."""
    settings: dict[str, Any] = {}
    if args.steps is not None:
        settings["sample_steps"] = args.steps
    if args.sampler:
        settings["sample_method"] = args.sampler
    if args.scheduler:
        settings["scheduler"] = args.scheduler
    return settings


def build_payload(
    args: argparse.Namespace,
    prompt: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Build a native image request."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    # sd-server only reads "seed" at the top level; inside sample_params it is
    # silently ignored, which makes runs non-repeatable even with --seed.
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.negative:
        payload["negative_prompt"] = args.negative
        print(
            "warning: negative prompts need CFG > 1; this deployment runs "
            "--cfg-scale 1.0, so the negative prompt has no effect",
            file=sys.stderr,
        )
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
    prompt = upsample_prompt(args, args.prompt, [])
    run_job(args, build_payload(args, prompt, args.width, args.height))


def cmd_edit(args: argparse.Namespace) -> None:
    """Run an image-to-image job."""
    prompt = upsample_prompt(args, args.prompt, [args.images[0]])
    size = image_size(args.images[0])
    width = args.width or (size[0] if size else 1024)
    height = args.height or (size[1] if size else 1024)
    payload = build_payload(args, prompt, width, height)
    encoded = [
        base64.b64encode(path.read_bytes()).decode("ascii") for path in args.images
    ]
    payload["init_image"] = encoded[0]
    # The init image must not also be a reference: FLUX.3 Klein is
    # reference-conditioned and reproduces a referenced image instead of editing it.
    if len(encoded) > 1:
        payload["ref_images"] = encoded[1:]
    if args.strength is not None:
        payload["strength"] = args.strength
    run_job(args, payload)


def cmd_hires(args: argparse.Namespace) -> None:
    """Run a two-pass hires job."""
    prompt = upsample_prompt(args, args.prompt, [])
    payload = build_payload(args, prompt, args.width, args.height)
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
    parser.add_argument("--api", default=DEFAULT_API, help="image service base URL")
    parser.add_argument(
        "--chat-api",
        default=DEFAULT_CHAT_API,
        help="prompt upsampler base URL",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="send the prompt as-is instead of expanding it with the chat model",
    )
    parser.add_argument(
        "--upsample-model",
        default="nemotron-3.5-bolt",
        help="model used to expand the prompt",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the upsampled prompt to stderr",
    )
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
        help="fixed seed (reproduces composition; pixel noise may still vary)",
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
        help="two-pass generation with an upscaler model",
    )
    add_common(hires)
    hires.add_argument("--width", type=int, default=1024)
    hires.add_argument("--height", type=int, default=1024)
    hires.add_argument("--scale", type=float, default=1.5)
    hires.add_argument("--denoise", type=float, default=0.3)
    hires.add_argument(
        "--hires-steps",
        type=int,
        default=0,
        help="second-pass steps (0 reuses --steps; server default)",
    )
    hires.add_argument(
        "--upscaler",
        default="RealESRGAN_x4",
        help="hires upscaler name (server default: RealESRGAN_x4)",
    )
    hires.set_defaults(func=cmd_hires)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
