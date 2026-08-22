# Deploying Castwright with the Ollama → NInfer gateway

This directory bundles vanilla Castwright with a local adapter that lets
Castwright drive a **llama-swap / NInfer Qwen3.8-27B** service through the
standard Ollama API. Clone this branch and everything needed is in one repo.

## Architecture

```
Castwright (UI + server, :8080)
    │  Ollama API (OLLAMA_URL)
    ▼
gateway.mjs (127.0.0.1:11435)          ← this adapter, plain Node.js, no deps
    │  OpenAI-compatible /v1/chat/completions
    ▼
llama-swap (127.0.0.1:5001)            ← pre-existing service
    ▼
NInfer · Qwen3.8-27B (NVFP4)           ← pre-existing inference backend
```

| Component     | Port  | Notes                                                        |
| ------------- | ----- | ------------------------------------------------------------ |
| Castwright    | 8080  | Web UI + app server                                          |
| Gateway       | 11435 | Loopback-only (`127.0.0.1`); change via `GATEWAY_PORT`       |
| llama-swap    | 5001  | OpenAI-compatible front end over NInfer                      |

The gateway does not download or duplicate model weights. It translates
protocol only: Ollama requests in, OpenAI requests out, and it maps
`keep_alive: 0` to llama-swap `/unload` so Castwright can release the GPU
before TTS starts.

## Prerequisites

- Node.js ≥ 22.22.0
- ffmpeg ≥ 6.0 on PATH
- Python 3.12 (for the Castwright TTS sidecar)
- A running llama-swap instance at `127.0.0.1:5001` serving NInfer
  Qwen3.8-27B, with an OpenAI-compatible `/v1/chat/completions` endpoint
- Recommended: a dedicated llama-swap alias named **`castwright`** pointing
  at the same backend, configured with thinking disabled and a low-temperature
  sampling profile (so other clients keep using the shared `qwen` alias
  unchanged). If you don't create such an alias, point `UPSTREAM_MODEL` at
  whichever model name your llama-swap actually serves.

## Step 1 — Clone this branch

```bash
git clone -b feat/ollama-gateway-bundle https://github.com/<you>/Castwright.git
cd Castwright
```

## Step 2 — Install Castwright

Follow [INSTALL.md](../INSTALL.md) → *Install — Linux*:

```bash
npm ci
npm --prefix server ci

cd server/tts-sidecar
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
cd ../..

bash scripts/install-kokoro.sh        # or install Kokoro in-app later

cp server/.env.example server/.env    # then edit WORKSPACE_DIR
```

## Step 3 — Point Castwright at the gateway

In `server/.env`:

```dotenv
OLLAMA_URL=http://127.0.0.1:11435
OLLAMA_MODEL=qwen3.8:27b
```

Then start the app once (`npm run start:prod`) and in the UI go to
**Account → Defaults for new books → Analysis model** and set it to
`qwen3.8:27b`. The UI picker takes precedence over `OLLAMA_MODEL`; the tag
must contain `:` (Ollama shape) for it to be used as the model name.

Alternative: skip the `.env` edit entirely and run the gateway on the
default Ollama port instead — `GATEWAY_PORT=11434 node castwright-ollama-gateway/gateway.mjs`
(only if nothing else occupies 11434).

## Step 4 — Run the gateway

Foreground (for testing):

```bash
node castwright-ollama-gateway/gateway.mjs
# → Castwright Ollama gateway listening on http://127.0.0.1:11435; upstream http://127.0.0.1:5001/castwright
```

As a systemd service (example, adjust paths/user):

```ini
# /etc/systemd/system/castwright-gateway.service
[Unit]
Description=Castwright Ollama gateway (llama-swap adapter)
After=network.target

[Service]
WorkingDirectory=/opt/castwright/castwright-ollama-gateway
ExecStart=/usr/bin/env node gateway.mjs
Environment=UPSTREAM_MODEL=castwright
Restart=on-failure
User=castwright

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now castwright-gateway
```

## Verify

```bash
curl -s http://127.0.0.1:11435/api/version
# {"version":"castwright-llama-swap-gateway/1.0.0"}

curl -s http://127.0.0.1:11435/api/tags
# single model entry named "qwen3.8:27b"

curl -s http://127.0.0.1:11435/health
# {"status":"ok"}

curl -s http://127.0.0.1:11435/api/ps
# {"models":[]} until the model has been loaded by a request

# cheap end-to-end round trip (loads the model, ~1 token):
curl -s http://127.0.0.1:11435/api/generate \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8:27b","prompt":"Reply only with OK.","stream":false}'
```

After that, `/api/ps` reports the model as loaded, and Castwright's Model
Manager shows `qwen3.8:27b` as available. From the UI you can now run
analysis normally; when a job finishes, Castwright sends `keep_alive: 0`,
which the gateway turns into llama-swap `/unload` to free VRAM for TTS.

## Gateway environment reference

| Variable             | Default                     | Purpose                                            |
| -------------------- | --------------------------- | -------------------------------------------------- |
| `GATEWAY_HOST`       | `127.0.0.1`                 | Bind address (keep loopback)                       |
| `GATEWAY_PORT`       | `11435`                     | Port Castwright talks to                           |
| `LLAMA_SWAP_URL`     | `http://127.0.0.1:5001`     | Upstream llama-swap base URL                       |
| `GATEWAY_MODEL`      | `qwen3.8:27b`               | Public model name exposed to Castwright            |
| `UPSTREAM_MODEL`     | `qwen`                      | Model name sent to llama-swap (`castwright` alias) |
| `RUNNING_MODEL`      | `qwen`                      | Canonical key llama-swap reports running aliases under; used by `/api/ps` |
| `MODEL_SIZE_BYTES`   | `21492695040`               | Reported model size                                |
| `MODEL_VRAM_BYTES`   | `27250393088`               | Reported VRAM usage                                |
| `MAX_OUTPUT_TOKENS`  | `8192`                      | Cap for `num_predict`                              |
| `REQUEST_TIMEOUT_MS` | `1800000`                   | Upstream request timeout (30 min)                  |

## Notes

- llama-swap reports a running alias under its canonical model key (`qwen`),
  so leave `RUNNING_MODEL=qwen` even when `UPSTREAM_MODEL=castwright` —
  otherwise `/api/ps` would claim the model is unloaded.
- Ollama `format` (JSON schema / `"json"`) is enforced via an injected strict
  system prompt, since llama-swap's OpenAI endpoint is what actually serves
  the completion.
- The gateway exits cleanly on SIGINT/SIGTERM; logs are plain stdout/stderr.