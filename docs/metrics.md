# Metrics and dashboards

The status profile runs Prometheus, Grafana, node-exporter and a small GPU collector alongside the model services. The model services expose their own metrics where the upstream server supports it.

| Service | Port | What it provides |
|---|---|---|
| Prometheus | `:9199` | Metrics storage and queries, 30 day retention |
| Grafana | `:9200` | SparkStack dashboard and ad hoc queries |
| node-exporter | internal `:9100` | Host CPU, memory, disk and network |
| vLLM | internal `:8080/metrics` | Chat tokens, queue, latency and KV cache |
| llama.cpp | internal `:8080/metrics` | Embedding and reranking throughput and queue |
| GPU collector | textfile | GPU utilization, power, temperature and SM clock |

Grafana has anonymous viewer access on the LAN. The admin user is `admin`; the password comes from `GRAFANA_ADMIN_PASSWORD` in `.env`, with `sparkstack` as the fallback. Change it before exposing the box beyond a trusted network.

## Quick start

```bash
./sparkstack.sh up status
```

Open <http://192.168.1.10:9200> and choose the **SparkStack** dashboard. Prometheus is at <http://192.168.1.10:9199>.

The dashboard shows GPU utilization, power, temperature and sample age; host CPU and memory; vLLM running and waiting requests; KV cache use; vLLM generated and prompt tokens per second; vLLM p95 end-to-end and inter-token latency; and llama.cpp throughput for embeddings and reranking.

## GPU collector

The status profile runs a small CUDA container that calls `nvidia-smi` every 15 seconds and writes the supported fields to node-exporter's textfile collector. No host setup is needed.

The sample file is `data/node-exporter/textfile/gpu.prom`. The `GPU sample age` panel should stay under 30 seconds. If it grows, check `docker logs sparkstack-gpu-exporter-1` and confirm the container is healthy.

## What GB10 reports

These fields work and are collected:

- GPU utilization
- GPU temperature
- Board power draw
- SM clock

These do not work on GB10 in the usual way:

- Framebuffer used and free: there is no separate framebuffer. Use host memory and the vLLM KV cache metric instead.
- DCGM profiling metrics: the profiling module is not loaded on this hardware.
- Memory copy utilization: the copy engine metric reads zero.

The vLLM `vllm:kv_cache_usage_perc` metric is the useful signal for chat memory pressure. Host memory is available from node-exporter.

## Useful queries

```promql
rate(vllm:generation_tokens_total[5m])
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))
sum by (instance) (rate(llamacpp:prompt_tokens_total[5m]))
node_memory_MemAvailable_bytes
```

Prometheus keeps 30 days in the `prometheus_data` Docker volume. Grafana state lives in `grafana_data`.

## Troubleshooting

- A target is down: the matching profile is not running. `./sparkstack.sh up` starts the full stack.
- GPU panels are empty: check `docker logs sparkstack-gpu-exporter-1` and `cat data/node-exporter/textfile/gpu.prom`.
- GPU sample age keeps growing: the exporter container is unhealthy. Restart it with `docker compose --profile status restart gpu-exporter`.
- Grafana shows no dashboard: check the container logs and the provisioning paths under `config/grafana`.
- Prometheus restarts with a permission error: use the named volumes in Compose, not a host bind mount for `/prometheus`.
- Host memory looks full: this is shared with the GPU. Compare `node_memory_MemAvailable_bytes` with the README's idle budget.

Metrics are for a trusted LAN. Prometheus has no authentication and Grafana allows anonymous viewing.
