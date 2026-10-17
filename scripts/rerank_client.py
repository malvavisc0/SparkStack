#!/usr/bin/env python3
"""Rerank documents with the SparkStack Qwen3 reranker."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from client_support import fail, request_json

DEFAULT_API = os.environ.get("SPARKSTACK_RERANK_URL", "http://localhost:9198")


def read_documents(args: argparse.Namespace) -> list[str]:
    """Combine positional documents with lines from --doc-file."""
    documents = list(args.documents)
    if args.doc_file:
        try:
            lines = Path(args.doc_file).read_text().splitlines()
        except OSError as error:
            fail(f"cannot read {args.doc_file}: {error}")
        documents.extend(line.strip() for line in lines if line.strip())
    if not documents:
        fail("provide at least one document or --doc-file")
    return documents


def main() -> None:
    """Parse options and rerank the documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="search query")
    parser.add_argument("documents", nargs="*", help="documents to rank")
    parser.add_argument(
        "--doc-file",
        type=Path,
        help="read one document per line",
    )
    parser.add_argument("--top-n", type=int, help="only print the best N")
    parser.add_argument("--json", action="store_true", help="print the raw response")
    parser.add_argument("--api", default=DEFAULT_API, help="service base URL")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    if args.top_n is not None and args.top_n < 1:
        parser.error("--top-n must be positive")
    documents = read_documents(args)
    payload = {"query": args.query, "documents": documents}
    url = args.api.rstrip("/") + "/v1/rerank"
    response: dict[str, Any] = request_json(url, payload=payload, timeout=args.timeout)
    results = sorted(
        response.get("results") or [],
        key=lambda result: result["relevance_score"],
        reverse=True,
    )
    if args.top_n is not None:
        results = results[: args.top_n]
    if args.json:
        print(json.dumps(response, indent=2))
        return
    for rank, result in enumerate(results, start=1):
        index = result["index"]
        score = result["relevance_score"]
        text = documents[index].replace("\n", " ").strip()
        print(f"{rank:2d}. {score:.6f}  {text[:110]}")
    usage: dict[str, Any] = response.get("usage") or {}
    print(f"{usage.get('total_tokens', '?')} tokens")


if __name__ == "__main__":
    main()
