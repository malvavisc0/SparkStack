# Reranking

The `reranker` service runs Qwen3-Reranker-0.6B on port `:9198`. It is a cross-encoder: it reads the query and each document together and returns a relevance score. Use it after vector search to put the best passages first.

A three-document request took about 30 ms on the Spark. The model uses Q8_0 weights, an 8K context and the rank pooling head from a llama.cpp conversion of the official Qwen3-Reranker model.

## Quick start

`scripts/rerank_client.py` sends a query and any number of documents, sorts the results and prints the scores.

```bash
./scripts/rerank_client.py "employment termination notice period" \
  "The Labour Code requires 30 calendar days written notice." \
  "Corporate tax rates for small enterprises." \
  "Notice must be given in writing before termination."
```

```text
 1. 0.913154  The Labour Code requires 30 calendar days written notice.
 2. 0.152765  Notice must be given in writing before termination.
 3. 0.000070  Corporate tax rates for small enterprises.
256 tokens
```

Use `--doc-file` to read one document per line, `--top-n` to keep only the best results, and `--json` for the raw response.

## Request

```bash
curl http://192.168.1.10:9198/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "employment termination notice period",
    "documents": [
      "The Labour Code requires 30 calendar days written notice.",
      "Corporate tax rates for small enterprises."
    ]
  }'
```

The response includes the model name, token usage and results with `index` and `relevance_score`. The client sorts by score, then prints the original document text.

Scores are model outputs, not probabilities. They are useful for ordering documents for one query. Do not compare absolute scores across different queries without calibration.

## Use with retrieval

The usual pipeline is:

1. Embed the query and retrieve 30 to 50 candidates with the embeddings service.
2. Rerank those candidates with the original query.
3. Keep the top 3 to 8 passages for the chat model, depending on context budget.

Reranking costs one forward pass per document, so keep the candidate set bounded. A 0.6B model is fast enough for interactive search; a 4B model is available if quality matters more than latency.

## Model choice

The default is `Qwen3-Reranker-0.6B` in Q8_0. On MTEB AskUbuntuDupQuestions, the Q8_0 build scored 0.6677 NDCG@10, within 0.2% of the F16 model. The 4B model scored about 0.70 and costs several times the compute and uses 2.3 GB in Q4_K_M. The service loads one model; switch by changing the compose command and downloading the larger file.

The context is 8,192 tokens for the query plus one document. Longer documents are truncated. Split long passages before reranking.

## Troubleshooting

- Connection refused: start the utility profile with `./sparkstack.sh up utility`.
- Empty results: send at least one string in `documents`.
- All scores near zero: check that the query and documents use the same language and that the document text is present.
- Scores look compressed: the rank head is trained for ordering, not absolute thresholds. Use the top-k cutoff rather than a fixed score.
- Long documents lose detail: lower the chunk size in the retrieval step.

The service definition lives in `docker-compose.yaml`. Stack setup is covered in the [README](../README.md).
