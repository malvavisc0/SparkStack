# Chat and vision

The `llm-coder390` service runs coder390, a Qwen3.8-27B post-train, on port
`:9193`. SGLang serves the NVFP4 W4A16 tier with the bundled DFlash2 draft. The
API is OpenAI-compatible, accepts text and images, supports tool calling, and
returns reasoning separately from the answer.

The model card recommends temperature 0.6, top_p 0.95 and top_k 20. The
service uses the full 262K context window (262,144 tokens) and allows two
concurrent requests, leaving memory for the image, speech and utility services.
Thinking is enabled by default. On the Spark, a short answer with thinking off
takes about 0.25 s; coding answers run at about 45–65 tokens/s. Prefill runs at
about 1,100 tokens/s, so a 128K prompt takes roughly 2.5 minutes before the
first token.

## Quick start

`scripts/chat_client.py` uses the OpenAI chat route and the recommended sampling
settings. It needs Python 3.10 or newer and no packages. It reads the bearer
token from `LLM_API_KEY`.

```bash
export LLM_API_KEY=$(sed -n 's/^LLM_API_KEY=//p' .env)
./scripts/chat_client.py "Explain unified memory in one sentence." --no-thinking
./scripts/chat_client.py "Write a short product update." --stream --stats
```

The client sends an 8,192 token budget, with 0.6 temperature, 0.95 top_p and
top_k 20. Point it at another host with `--api` or `SPARKSTACK_CHAT_URL`, and
pass the key with `--api-key` or `LLM_API_KEY`.

## Request

The endpoint is `POST /v1/chat/completions`. It requires the `LLM_API_KEY` from
`.env` as a bearer token; requests without it return HTTP 401.

```bash
curl http://192.168.1.10:9193/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "coder390",
    "messages": [{"role": "user", "content": "Explain unified memory in one sentence."}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 8192
  }'
```

The response has `choices[0].message.content` for the answer and
`choices[0].message.reasoning_content` for the thinking trace. Do not
concatenate the two fields. Some clients show the reasoning, and most should
not.

## Thinking controls

The model card sets thinking with `chat_template_kwargs`:

| Mode | Value | Use it for |
|---|---|---|
| Default | omit | Best answer quality. |
| Fast | `{"enable_thinking": false}` | Extraction, routing, tests, short answers. |

The coder390 template exposes the on/off toggle; there is no medium-effort
setting on this serving path.

Reasoning tokens share the `max_tokens` budget. A 512 token budget can fill with
reasoning and leave `content` empty, with `finish_reason: "length"`. Use 8,192
tokens for reasoning work. If a response ends with `length` and no content,
raise the budget or pass `--no-thinking`.

Measured on the Spark:

| Request | Time | Output |
|---|---|---|
| Short answer, thinking off | 0.25 s | 2 tokens |
| Python code answer, thinking off | 1.86–2.01 s | 99–123 tokens, 53–61 tokens/s |
| Bat-and-ball reasoning, thinking on | 1.23 s | 36 tokens, 107 reasoning characters |
| Tool call, thinking off | 0.93 s | 26 tokens |
| Image question, thinking off | 1.93 s | 32 tokens |

The streaming rate is the completion rate including the prompt and first token,
so it is a lower bound.

## Streaming

`--stream` prints the answer as it arrives. Reasoning goes to stderr and is
hidden unless you pass `--show-reasoning`.

```bash
./scripts/chat_client.py "Explain unified memory in one sentence." --stream --stats
```

The stream carries reasoning in `delta.reasoning_content` and the answer in
`delta.content`. The client sends `stream_options.include_usage` so token counts
arrive in the final event.

## Images

Pass one image with `--image`. The client sends it as a data URL in an
`image_url` content part. The tested path is one image per prompt.

```bash
./scripts/chat_client.py "What is in this image? One sentence." \
  --image mug.png --no-thinking --max-tokens 256
```

The test image, a 512x512 fox rendering, returned a correct description in
1.93 s with a 32-token answer.

## Tools

Pass an OpenAI tools array with `--tools`. The client rejects streaming mode for
tool calls, since it prints the completed call rather than partial arguments.

```json
[{"type": "function", "function": {
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
}}]
```

```bash
./scripts/chat_client.py "What is the weather in Paris? Use the weather tool." \
  --tools tools.json --no-thinking
```

Output:

```text
tool call: get_weather({"city": "Paris"})
```

Tool calls arrive in `message.tool_calls`. `function.arguments` is a JSON string.
To finish the loop, append the assistant message and a `tool` message with the
result, then send the history again with `--messages`.

## Multi-turn requests

`--messages history.json` sends a full messages array and skips the prompt and
image options.

```json
[
  {"role": "system", "content": "You are a terse assistant."},
  {"role": "user", "content": "Name three colors."},
  {"role": "assistant", "content": "Red, green, blue."},
  {"role": "user", "content": "Which is warmest?"}
]
```

With the OpenAI SDK, set `base_url="http://<spark>:9193/v1"` and pass the
`LLM_API_KEY` value as `api_key`. Requests with a missing or wrong key return
HTTP 401.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `model` | `coder390` | Only one chat model is served. |
| `api_key` | `$LLM_API_KEY` | Bearer token; required. |
| `temperature` | 0.6 | Recommended by the model card. Use 0 for repeatable tests. |
| `top_p` | 0.95 | Recommended by the model card. |
| `top_k` | 20 | Recommended by the model card. |
| `max_tokens` | 8192 | Shared by reasoning and answer. |
| `seed` | unset | Set it for repeatability. |
| `chat_template_kwargs` | unset | Thinking controls. |
| `tools` | unset | OpenAI tools array. |
| `stream` | false | Streaming adds usage in the final chunk. |

## Troubleshooting

- HTTP 401: send `Authorization: Bearer $LLM_API_KEY` using the value from
  `.env`.
- `content` is empty and `finish_reason` is `length`: reasoning used the whole
  budget. Raise `max_tokens` or disable thinking.
- Connection refused: start the LLM profile with `./sparkstack.sh up llm`. The
  first SGLang load takes several minutes while it captures CUDA graphs.
- A tool call never appears: the model may answer directly. Make the tool
  description explicit and keep `tool_choice` at `auto` or `required`.
- Images fail: send PNG or JPEG data URLs. The client sets the MIME type from
  the file extension.
- Long prompts: the prompt, history, images and answer share the 262K window.
  Prefill is the cost, not memory: budget roughly 1,100 tokens/s.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
