# One-shot decisions

The `d1-decide` service runs LiquidAI d1-3B on port `:9194`. It reads a state and
returns probabilities for a set of questions. It is not a chat model: output
tokens are always zero, and a request takes about 0.1 s for text or 0.2 s when
an image is attached. Use it for routing, triage, labels and scores when a full
chat model is more than you need.

## Quick start

Save a questions file:

```json
{
  "route": {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
      "billing": "payments and refunds",
      "shipping": "deliveries",
      "technical": "software problems"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["can wait", "this week", "today", "right now"]
  },
  "angry": {
    "type": "noul",
    "instructions": "Is the customer angry?"
  }
}
```

Run it:

```bash
./scripts/decide_client.py \
  --state "I was charged twice and nobody replied. I want a refund now." \
  --questions questions.json
```

Output:

```text
route: billing (97.8% confidence)
  billing 98.6%, technical 0.8%, shipping 0.7%
urgency: 2.24 / 3 (confidence 24.1%)
  0=can wait, 1=this week, 2=today, 3=right now
angry: 54.7% true
147 input tokens, 0 output tokens, 0.09s
```

The client prints a summary. Use `--json` for the raw response.

## Question types

| Type | `criteria` | Response |
|---|---|---|
| `choice` | Object of labels. Descriptions can be `null`. | `choice`, `probabilities`, `confidence` |
| `score` | Array of 2 to 10 ordered levels. | `score`, `legend`, `probabilities`, `confidence` |
| `noul` | Optional `true` and `false` descriptions. | `noul`, the probability that the answer is true |

A `choice` question returns one probability per label, and the probabilities add
to 1. One label is accepted and returns probability 1.0, but it does not classify
anything. `confidence` is the model's own 0 to 1 score, not the top probability.

A `score` question maps levels to the numbers 0 through N-1. The `score` is the
expected value. In the example, 2.24 falls between "today" and "right now".

A `noul` question returns `P(true)`. In a test, a calm thank-you message scored
0.009 and an angry complaint scored 0.992.

## State

The state can be text, a JSON object, or a messages array with images. An object
is serialized as JSON for the model.

```json
{"ticket": "I was charged twice.", "plan": "pro"}
```

The client accepts text with `--state` or a text or JSON file with
`--state-file`. A JSON file is parsed; any other file is sent as text.

For images, pass `--image` with a text state:

```bash
./scripts/decide_client.py \
  --state "Classify the image." \
  --image mug.png \
  --questions '{"object":{"type":"choice","instructions":"What is shown?","criteria":{"mug":"a coffee mug","fox":"a fox","car":"a car"}}}'
```

```text
object: mug (98.5% confidence)
  mug 99.0%, fox 0.6%, car 0.4%
344 input tokens, 0 output tokens, 0.17s
```

The client wraps the text and image in a user message. For full control, use a
state file with this shape:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Classify the image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

The endpoint rejects a top-level `videos` field. Image input works because the
service starts the model with its vision projector.

## Recommended use

Ask several questions in one request to reuse the same state. More questions add
tokens and some time, but they avoid repeating the state in separate calls.

Use `noul` for binary decisions, `choice` for unordered labels, and `score` for
ordered levels. Give descriptions to labels that could be confused, and keep
score questions to 10 levels or fewer.

The service has a 32,768 token context. Keep the state compact, send only the
fields the questions need, and avoid dumping whole documents when a summary will
do.

Probabilities can vary slightly between identical runs. In a repeated test the
top choice stayed the same but moved by a few tenths of a percent. Set thresholds
with margin, and compare changes across many examples instead of one request.

Measured timings on the Spark:

| Input | Time | Tokens |
|---|---|---|
| Text with three questions | 0.09 to 0.11 s | 147 to 165 |
| Image with one question | 0.17 to 0.24 s | 344 to 349 |

## Troubleshooting

- `400 "state" must be provided`: send a state, even an empty string or object.
- `400 "questions" must be a non-empty object`: the questions payload is missing
  or empty.
- `score` fails: provide between 2 and 10 levels.
- Low confidence: the state is ambiguous or the labels overlap. Add criteria
  descriptions or more context.
- Image requests fail: this stack starts the model with `--mmproj`. Check the
  service logs if the projector is missing.

The service definition lives in `docker-compose.yaml`. General stack setup is
covered in the [README](../README.md).
