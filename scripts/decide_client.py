#!/usr/bin/env python3
"""Run one-shot decisions against the SparkStack d1-3B service."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from client_support import fail, image_part, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_DECIDE_URL", "http://localhost:9194")


def read_state(args: argparse.Namespace) -> Any:
    """Read the state from the CLI value or a text or JSON file."""
    if args.state is not None:
        return args.state
    path = Path(args.state_file)
    if not path.is_file():
        fail(f"state file not found: {path}")
    try:
        raw = path.read_text()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def read_questions(value: str) -> dict[str, Any]:
    """Read questions from a JSON file or an inline JSON object."""
    if value.lstrip().startswith("{"):
        raw = value
    else:
        path = Path(value)
        if not path.is_file():
            fail(f"questions file not found: {value}")
        try:
            raw = path.read_text()
        except OSError as error:
            fail(f"cannot read {path}: {error}")
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as error:
        fail(f"questions are not valid JSON: {error}")
    if not isinstance(questions, dict) or not questions:
        fail("questions must be a non-empty JSON object")
    return questions


def attach_images(state: Any, paths: list[Path]) -> Any:
    """Wrap a text state and its images in a user message."""
    if not paths:
        return state
    if not isinstance(state, str):
        fail("--image requires a text state; put image parts in a state file")
    content: list[dict[str, Any]] = [{"type": "text", "text": state}]
    content.extend(image_part(path) for path in paths)
    return {"messages": [{"role": "user", "content": content}]}


def print_summary(response: dict[str, Any], elapsed: float) -> None:
    """Print a compact summary of the decision answers."""
    answers: dict[str, Any] = response.get("answers") or {}
    for question, answer in answers.items():
        kind = answer.get("type")
        if kind == "choice":
            probabilities: dict[str, float] = answer.get("probabilities") or {}
            choice = answer.get("choice")
            confidence = answer.get("confidence") or 0.0
            print(f"{question}: {choice} ({confidence * 100:.1f}% confidence)")
            ranked = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            print(
                "  " + ", ".join(f"{label} {value * 100:.1f}%" for label, value in ranked)
            )
        elif kind == "score":
            probabilities = answer.get("probabilities") or {}
            score = answer.get("score") or 0.0
            confidence = answer.get("confidence") or 0.0
            top = max((int(level) for level in probabilities), default=0)
            print(f"{question}: {score:.2f} / {top} (confidence {confidence * 100:.1f}%)")
            legend: dict[str, str] = answer.get("legend") or {}
            if legend:
                print(
                    "  "
                    + ", ".join(f"{level}={label}" for level, label in legend.items())
                )
        elif kind == "noul":
            value = answer.get("noul") or 0.0
            print(f"{question}: {value * 100:.1f}% true")
        else:
            print(f"{question}: {json.dumps(answer)}")
    usage: dict[str, Any] = response.get("usage") or {}
    print(
        f"{usage.get('input_tokens', '?')} input tokens, "
        f"{usage.get('output_tokens', '?')} output tokens, {elapsed:.2f}s"
    )


def main() -> None:
    """Parse options and run the decision request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", help="state as text")
    parser.add_argument(
        "--state-file",
        type=Path,
        help="state as a text or JSON file",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="questions as a JSON file or inline JSON",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="image for a text state; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="print the raw response")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    if (args.state is None) == (args.state_file is None):
        parser.error("provide exactly one of --state or --state-file")
    state = attach_images(read_state(args), args.image)
    questions = read_questions(args.questions)
    payload = {"state": state, "questions": questions}
    url = args.api.rstrip("/") + "/v1/systemone"

    started = time.monotonic()
    response: dict[str, Any] = request_json(url, payload=payload, timeout=args.timeout)
    elapsed = time.monotonic() - started

    if args.json:
        print(json.dumps(response, indent=2))
    else:
        print_summary(response, elapsed)


if __name__ == "__main__":
    main()
