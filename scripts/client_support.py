"""Shared HTTP, media and error helpers for the SparkStack clients."""

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn


def fail(message: str) -> NoReturn:
    """Exit with a clear error message."""
    raise SystemExit(message)


def send_request(
    url: str,
    request: urllib.request.Request,
    timeout: float,
) -> bytes:
    """Send a prepared request and return the raw response body."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {error.code} from {url}\n{detail[:1000]}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"cannot reach {url}: {error.reason}") from error


def request_bytes(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 600,
    method: str = "POST",
) -> bytes:
    """Send a JSON request or a plain request and return the raw body."""
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    return send_request(url, request, timeout)


def request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 600,
    method: str = "POST",
) -> Any:
    """Send a request and decode its JSON response."""
    body = request_bytes(url, payload=payload, timeout=timeout, method=method)
    return json.loads(body)


def image_url(path: Path) -> str:
    """Return a data URL for an image file."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SystemExit(f"cannot read {path}: {error}") from error
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_part(path: Path) -> dict[str, Any]:
    """Return an OpenAI-compatible image content part."""
    return {"type": "image_url", "image_url": {"url": image_url(path)}}


def multipart(
    fields: Mapping[str, str | int | None],
    file_path: Path,
) -> tuple[bytes, str]:
    """Build a multipart body with one file part."""
    boundary = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file_path.name) or "upload"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode()
        )
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{safe_name}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
    )
    try:
        chunks.append(file_path.read_bytes())
    except OSError as error:
        raise SystemExit(f"cannot read {file_path}: {error}") from error
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def request_multipart_json(
    url: str,
    fields: Mapping[str, str | int | None],
    file_path: Path,
    timeout: float = 600,
) -> Any:
    """Upload a file with multipart fields and decode a JSON response."""
    body, boundary = multipart(fields, file_path)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return json.loads(send_request(url, request, timeout))
