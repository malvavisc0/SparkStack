# Embeddings

The `embeddings` service runs `embeddinggemma-2` on port `:9195`. It turns text,
images and audio into 768-dimensional vectors in one shared space. The route is
OpenAI-compatible, and every vector is L2-normalized, so cosine similarity is a
dot product.

A text input takes about 0.01 s. An image takes about 0.14 s, and a short audio
clip took 0.07 s. In a test, a coffee mug photo scored 0.712 against "a ceramic
coffee mug on a wooden table" and about 0.49 against unrelated sentences.

## Quick start

`scripts/embed_client.py` accepts text and images, batches them into one request,
and prints a cosine matrix.

```bash
./scripts/embed_client.py \
  "a ceramic coffee mug on a wooden table" \
  "a red fox in the snow" \
  --image mug.png
```

```text
[0] text "a ceramic coffee mug on a wooden table"  dim=768 norm=1.000
[1] text "a red fox in the snow"  dim=768 norm=1.000
[2] image mug.png  dim=768 norm=1.000
cosine:
               0       1       2
0          1.000   0.495   0.712
1          0.495   1.000   0.489
2          0.712   0.489   1.000
151 input tokens
```

For a single input, the client prints the dimension, norm and the first eight
values. Use `--json` for the raw response and `--out vectors.json` to save it.

## Request

Plain strings are the simplest input:

```bash
curl http://192.168.1.10:9195/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello world", "a second sentence"]}'
```

For images, audio or video, send content parts:

```json
{
  "input": [{
    "content": [
      {"type": "text", "text": "a ceramic coffee mug on a wooden table"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

The server accepts these content part types:

| Part | Field |
|---|---|
| Text | `{"type": "text", "text": "..."}` |
| Image | `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` |
| Audio | `{"type": "input_audio", "input_audio": {"data": "<base64>"}}` |
| Video | `{"type": "input_video", "input_video": {"data": "<base64>"}}` |

Text, image and audio inputs were tested here. Video uses the same shape and is
accepted by the server, but it was not part of this test run.

Each entry in `input` produces one vector. A batch returns vectors in the same
order, with `usage.prompt_tokens` for the total. Batch related items when you
can, since one request avoids repeated HTTP overhead.

## Output format

The default `encoding_format` is `float`, which returns an array of 768 numbers.
Set `encoding_format` to `base64` for a compact string, which is 4,096 characters
for a 768-dimension vector. Decode it as 32-bit floats before
storing it.

Vectors are already normalized. Store them as they arrive and use cosine
similarity or a dot product. Most vector databases can take the float array
directly.

Related pairs in this model are not tightly clustered by absolute score. The mug
photo and its text description scored 0.71, while two unrelated sentences scored
about 0.5. Calibrate a threshold on your own data instead of assuming a fixed
cutoff.

Pair retrieval with the [reranker](reranking-api.md) to put the best passages
first.

## Context and chunking

The model has an 8,192 token context. Long documents must be split before
embedding. Chunk by section or paragraph, keep each chunk well under the limit,
and store the source text with the vector. For retrieval, smaller chunks usually
give better matches than one vector for a whole document.

## Troubleshooting

- `400 "input" or "content" must be provided`: the request body is empty or uses
  the wrong field.
- An empty content array is accepted and produces a vector from special tokens
  only. Send text or media when you need a useful vector.
- Image requests fail: send a data URL with a supported image type. The service
  needs its vision projector, which this stack starts.
- Long text returns an error or a weak vector: the input exceeded the 8K context.
  Split it into smaller chunks.
- The client reports a count mismatch: the service returned fewer vectors than
  inputs. Check the raw response with `--json`.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
