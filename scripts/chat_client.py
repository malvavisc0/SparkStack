#!/usr/bin/env python3
"""Chat with the SparkStack Nemotron service."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from client_support import fail, image_part, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_CHAT_URL", "http://localhost:9193")


def load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the messages array from the CLI options."""
    if args.messages:
        if args.prompt or args.image or args.system:
            fail("--messages cannot be combined with a prompt, --system, or --image")
        try:
            raw = Path(args.messages).read_text()
        except OSError as error:
            fail(f"cannot read {args.messages}: {error}")
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError as error:
            fail(f"--messages is not valid JSON: {error}")
        if not isinstance(messages, list) or not messages:
            fail("--messages must contain a non-empty JSON array")
        return messages
    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read()
    if not prompt.strip():
        fail("provide a prompt, a --messages file, or text on stdin")
    messages: list[dict[str, Any]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    content: Any = prompt
    if args.image:
        content = [{"type": "text", "text": prompt}, image_part(args.image)]
    messages.append({"role": "user", "content": content})
    return messages


def build_payload(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the chat completion request body."""
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.medium_effort:
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "medium_effort": True,
        }
    if args.tools:
        try:
            tools = json.loads(Path(args.tools).read_text())
        except OSError as error:
            fail(f"cannot read {args.tools}: {error}")
        except json.JSONDecodeError as error:
            fail(f"--tools is not valid JSON: {error}")
        payload["tools"] = tools
        payload["tool_choice"] = args.tool_choice or "auto"
    if args.stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def stats_line(
    elapsed: float,
    ttft: float | None,
    usage: dict[str, Any] | None,
) -> str:
    """Format timing and token counts for stderr."""
    parts = [f"elapsed {elapsed:.2f}s"]
    if ttft is not None:
        parts.append(f"ttft {ttft:.2f}s")
    if usage:
        tokens = usage.get("completion_tokens")
        if tokens:
            parts.append(f"tokens {tokens} ({tokens / elapsed:.1f}/s)")
    return " | ".join(parts)


def run_stream(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    show_reasoning: bool,
    show_stats: bool,
) -> None:
    """Stream a chat completion and print the answer."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    ttft: float | None = None
    finish: str | None = None
    usage: dict[str, Any] | None = None
    printed_content = False
    printed_reasoning = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event: dict[str, Any] = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    if ttft is None:
                        ttft = time.monotonic() - started
                    if show_reasoning:
                        sys.stderr.write(str(reasoning))
                        sys.stderr.flush()
                        printed_reasoning = True
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - started
                    sys.stdout.write(str(content))
                    sys.stdout.flush()
                    printed_content = True
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {error.code} from {url}\n{detail[:1000]}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"cannot reach {url}: {error.reason}") from error
    elapsed = time.monotonic() - started
    if printed_reasoning:
        sys.stderr.write("\n")
    if printed_content:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if finish == "length":
        print(
            "finish_reason=length: the output hit the token budget; raise --max-tokens",
            file=sys.stderr,
        )
    if show_stats:
        print(stats_line(elapsed, ttft, usage), file=sys.stderr)


def run_complete(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    show_reasoning: bool,
    show_stats: bool,
    as_json: bool,
) -> None:
    """Run a non-streaming completion and print the result."""
    started = time.monotonic()
    response: dict[str, Any] = request_json(url, payload=payload, timeout=timeout)
    elapsed = time.monotonic() - started
    if as_json:
        print(json.dumps(response, indent=2))
        if show_stats:
            print(stats_line(elapsed, None, response.get("usage")), file=sys.stderr)
        return
    choice = response["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if show_reasoning and reasoning:
        print(reasoning, file=sys.stderr)
    content = message.get("content")
    if content:
        print(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        print(f"tool call: {function.get('name')}({function.get('arguments')})")
    if choice.get("finish_reason") == "length":
        print(
            "finish_reason=length: the output hit the token budget; raise --max-tokens",
            file=sys.stderr,
        )
    sys.stdout.flush()
    if show_stats:
        print(stats_line(elapsed, None, response.get("usage")), file=sys.stderr)


def main() -> None:
    """Parse options and run the chat client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="user message; omit to read stdin")
    parser.add_argument(
        "--messages",
        type=Path,
        help="JSON file with a full messages array",
    )
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--image", type=Path, help="image to attach")
    parser.add_argument("--model", default="nemotron-3.5-bolt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="shared budget for reasoning and answer",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable reasoning for low latency",
    )
    parser.add_argument(
        "--medium-effort",
        action="store_true",
        help="enable thinking with medium effort",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        help="JSON file with an OpenAI tools array",
    )
    parser.add_argument("--tool-choice", help="auto, none or required")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream the answer as it is generated",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="print reasoning to stderr",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print timing and token counts to stderr",
    )
    parser.add_argument("--json", action="store_true", help="print the raw response")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    if args.no_thinking and args.medium_effort:
        parser.error("--no-thinking and --medium-effort cannot be combined")
    if args.tool_choice and not args.tools:
        parser.error("--tool-choice requires --tools")
    if args.tools and args.stream:
        parser.error("--tools requires non-stream mode")
    if args.json and args.stream:
        parser.error("--json requires non-stream mode")

    messages = load_messages(args)
    payload = build_payload(args, messages)
    url = args.api.rstrip("/") + "/v1/chat/completions"
    if args.stream:
        run_stream(url, payload, args.timeout, args.show_reasoning, args.stats)
    else:
        run_complete(
            url,
            payload,
            args.timeout,
            args.show_reasoning,
            args.stats,
            args.json,
        )


if __name__ == "__main__":
    main()
