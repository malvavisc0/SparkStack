# Security

SparkStack is designed for a trusted private network. Treat every port in the
919x block as a LAN-only service and do not port-forward it to the internet.

- **Chat (`:9193`)** requires `LLM_API_KEY` from `.env` as a bearer token
  (`--api-key` on SGLang) and returns HTTP 401 without it.
- **Everything else is unauthenticated.** Decisions, embeddings, reranking,
  images, speech-to-text and text-to-speech are open to anyone who can reach the
  host.
- **Dozzle (`:9197`)** mounts `docker.sock` and runs with
  `DOZZLE_ENABLE_ACTIONS` and `DOZZLE_ENABLE_SHELL`, so it can start/stop
  containers and `exec` a shell into them from the UI. The `:ro` on the socket
  mount is cosmetic; it does not block API writes. Anyone who can reach Dozzle
  has the same power as root on this host.
- **Prometheus (`:9199`)** has no authentication.
- **Grafana (`:9200`)** allows anonymous viewing. Change the admin password
  (`GRAFANA_ADMIN_PASSWORD` in `.env`) and restrict network access before
  exposing it beyond the trusted LAN.
- **Gatus (`:9190`)** is a status dashboard with no authentication and no
  control actions.

The safe deployment is a private LAN, a firewall, or a VPN. If you need remote
access, put the Spark behind a VPN or an authenticating reverse proxy rather than
opening the 919x ports directly.
