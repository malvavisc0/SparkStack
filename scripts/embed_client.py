#!/usr/bin/env python3
"""Create and compare embeddings with the SparkStack embeddings service."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from client_support import image_part, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_EMBED_URL", "http://localhost:9195")


def dot(left: list[float], right: list[float]) -> float:
    """Return the dot product of two vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def print_summary(labels: list[str], vectors: list[list[float]]) -> None:
    """Print dimensions, norms and a cosine matrix."""
    norms = [math.sqrt(dot(vector, vector)) for vector in vectors]
    rows = zip(labels, vectors, norms, strict=True)
    for index, (label, vector, norm) in enumerate(rows):
        head = ", ".join(f"{value:.3f}" for value in vector[:8])
        print(f"[{index}] {label}  dim={len(vector)} norm={norm:.3f}")
        if len(labels) == 1:
            print(f"    first 8 values: [{head}]")
    if len(vectors) < 2:
        return
    print("cosine:")
    width = 8
    header = " " * width + "".join(f"{index:>{width}}" for index in range(len(vectors)))
    print(header)
    for row, left in enumerate(vectors):
        cells = []
        for column, right in enumerate(vectors):
            denominator = norms[row] * norms[column]
            value = dot(left, right) / denominator if denominator else 0.0
            cells.append(f"{value:>{width}.3f}")
        print(f"{row:<{width}}" + "".join(cells))


def main() -> None:
    """Parse options, create embeddings and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", help="text inputs")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="image input; repeatable",
    )
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--json", action="store_true", help="print the raw response")
    parser.add_argument("--out", type=Path, help="save the raw response as JSON")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    labels = [f'text "{text[:60]}"' for text in args.texts]
    inputs: list[dict[str, Any]] = [
        {"content": [{"type": "text", "text": text}]} for text in args.texts
    ]
    for path in args.image:
        labels.append(f"image {path.name}")
        inputs.append({"content": [image_part(path)]})
    if not inputs:
        parser.error("provide at least one text or --image input")

    payload = {"model": args.model, "input": inputs}
    url = args.api.rstrip("/") + "/v1/embeddings"
    response: dict[str, Any] = request_json(url, payload=payload, timeout=args.timeout)

    if args.out:
        args.out.write_text(json.dumps(response, indent=2) + "\n")
    if args.json:
        print(json.dumps(response, indent=2))
        return
    vectors: list[list[float]] = [item["embedding"] for item in response.get("data", [])]
    if len(vectors) != len(labels):
        raise SystemExit(f"expected {len(labels)} embeddings, got {len(vectors)}")
    print_summary(labels, vectors)
    usage: dict[str, Any] = response.get("usage") or {}
    print(f"{usage.get('prompt_tokens', '?')} input tokens")


if __name__ == "__main__":
    main()
