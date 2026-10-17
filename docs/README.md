# Service guides

Each guide covers one service: request shape, a Python client where it helps, measured timings and the defaults that work on the Spark. Start with the [main README](../README.md) for setup and monitoring.

| Service | Port | Guide | Client |
|---|---|---|---|
| Chat and vision | `:9193` | [chat-api.md](chat-api.md) | [chat_client.py](../scripts/chat_client.py) |
| Decisions | `:9194` | [decisions-api.md](decisions-api.md) | [decide_client.py](../scripts/decide_client.py) |
| Embeddings | `:9195` | [embeddings-api.md](embeddings-api.md) | [embed_client.py](../scripts/embed_client.py) |
| Reranking | `:9198` | [reranking-api.md](reranking-api.md) | [rerank_client.py](../scripts/rerank_client.py) |
| Image generation and editing | `:9196` | [image-api.md](image-api.md) | [flux_client.py](../scripts/flux_client.py) |
| Speech to text | `:9191` | [speech-to-text-api.md](speech-to-text-api.md) | [transcribe_client.py](../scripts/transcribe_client.py) |
| Text to speech | `:9192` | [text-to-speech-api.md](text-to-speech-api.md) | [tts_client.py](../scripts/tts_client.py) |
| Metrics and dashboards | `:9199`, `:9200` | [metrics.md](metrics.md) | n/a |

The clients use only the Python standard library and share helpers from [client_support.py](../scripts/client_support.py). They need Python 3.10 or newer.

The image client always expands prompts through the chat model before rendering. The other clients send requests as written.
