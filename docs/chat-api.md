# Chat and vision

The `llm-nemotron` service runs Nemotron 3.5 Bolt 74B on port `:9193`. It is
OpenAI-compatible, accepts text and images, and returns reasoning separately
from the answer.

The model card recommends temperature 1.0 and top_p 0.95. The service starts
with those values, a 256K context window (262,144 tokens), and thinking
enabled. A short answer with thinking off takes about 0.2 s. A 117-token answer
took 3.3 s at about 36 tokens/s. Reasoning-heavy requests take longer.

## Quick start

`scripts/chat_client.py` uses the OpenAI chat route and the recommended sampling
settings. It needs Python 3.10 or newer and no packages.

```bash
./scripts/chat_client.py "Explain unified memory in one sentence." --no-thinking
./scripts/chat_client.py "Write a short product update." --stream --stats
```

The client sends a 16,384 token budget, with 1.0 temperature and 0.95 top_p.
Point it at another host with `--api` or `SPARKSTACK_CHAT_URL`.

## Request

The endpoint is `POST /v1/chat/completions`. A minimal request:

```bash
curl http://192.168.1.10:9193/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3.5-bolt",
    "messages": [{"role": "user", "content": "Explain unified memory in one sentence."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 16384
  }'
```

The response has `choices[0].message.content` for the answer and
`choices[0].message.reasoning` for the thinking trace. Do not concatenate the
two fields. Some clients show the reasoning, and most should not.

## Thinking controls

The model card sets thinking with `chat_template_kwargs`:

| Mode | Value | Use it for |
|---|---|---|
| Default | omit | Best answer quality. |
| Fast | `{"enable_thinking": false}` | Extraction, routing, tests, short answers. |
| Medium | `{"enable_thinking": true, "medium_effort": true}` | Shorter reasoning on harder prompts. |

Reasoning tokens share the `max_tokens` budget. A 512 token budget can fill with
reasoning and leave `content` empty, with `finish_reason: "length"`. The model
card recommends 16,384 tokens for reasoning work, which is the client default.
If a response ends with `length` and no content, raise the budget or pass
`--no-thinking`.

Measured on the Spark:

| Request | Time | Output |
|---|---|---|
| Short answer, thinking off | 0.23 s | 2 tokens |
| Explanation, thinking off, streaming | 1.06 s | 38 tokens, 0.18 s to first token |
| 117-token explanation, thinking off | 3.28 s | 35.7 tokens/s |
| Image question, thinking off | 0.78 s | 18 tokens |
| Tool call, thinking off | 0.55 s | 25 tokens |
| Prime check, thinking on | 5.43 s | 85 tokens, 258 reasoning characters |
| Prime check, medium effort | 1.39 s | 63 tokens, 170 reasoning characters |

The streaming rate is the completion rate including the prompt and first token,
so it is a lower bound.

## Streaming

`--stream` prints the answer as it arrives. Reasoning goes to stderr and is
hidden unless you pass `--show-reasoning`.

```bash
./scripts/chat_client.py "Explain unified memory in one sentence." --stream --stats
```

The stream carries reasoning in `delta.reasoning` and the answer in
`delta.content`. The client sends `stream_options.include_usage` so token counts
arrive in the final event.

## Images

Pass one image with `--image`. The client sends it as a data URL in an
`image_url` content part. The service is configured for one image per prompt;
two image parts return HTTP 400.

```bash
./scripts/chat_client.py "What is in this image? One sentence." \
  --image mug.png --no-thinking --max-tokens 256
```

The test image, a 512x512 mug photo, returned a correct description in 0.78 s
with an 18-token answer.

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

With the OpenAI SDK, set `base_url="http://<spark>:9193/v1"` and use any
non-empty API key. The service does not check it.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `model` | `nemotron-3.5-bolt` | Only one chat model is served. |
| `temperature` | 1.0 | Recommended by the model card. Use 0 for repeatable tests. |
| `top_p` | 0.95 | Recommended by the model card. |
| `max_tokens` | 16384 | Shared by reasoning and answer. |
| `seed` | unset | Set it for repeatability. |
| `chat_template_kwargs` | unset | Thinking controls. |
| `tools` | unset | OpenAI tools array. |
| `stream` | false | Streaming adds usage in the final chunk. |

## Troubleshooting

- `content` is empty and `finish_reason` is `length`: reasoning used the whole
  budget. Raise `max_tokens` or disable thinking.
- Connection refused: start the LLM profile with `./sparkstack.sh up llm`. The
  first load takes several minutes.
- A tool call never appears: the model may answer directly. Make the tool
  description explicit and keep `tool_choice` at `auto` or `required`.
- Images fail: send PNG or JPEG data URLs. The client sets the MIME type from
  the file extension.
- Long prompts: the prompt, history, images and answer share the 256K window.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
