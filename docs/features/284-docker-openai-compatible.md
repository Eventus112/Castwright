---
status: active
shipped: null
owner: Eventus112 fork
---

# 284 — Docker deployment and OpenAI-compatible analyzer

> Status: active
> Key files: `Dockerfile`, `compose.yaml`, `compose.nvidia.yaml`, `docker/entrypoint.sh`, `server/src/analyzer/openai-compatible.ts`, `server/src/analyzer/index.ts`, `server/src/routes/user-settings.ts`, `src/views/analysing.tsx`
> URL surface: `http://127.0.0.1:8080/` by default
> OpenAPI ops: existing analysis operations; no new HTTP API
> Tracking: extends upstream `ops-2` ([#433](https://github.com/dudarenok-maker/Castwright/issues/433)) and the fork owner's OpenAI-compatible endpoint requirement.

## Benefit / Rationale

- **User:** a Linux host can start the complete web app and local TTS sidecar with `docker compose up --build -d` and keep books across rebuilds.
- **User:** analysis can use any service implementing OpenAI Chat Completions, including self-hosted vLLM, llama.cpp, LM Studio, or a hosted provider.
- **Technical:** Node, Python 3.12, ffmpeg 6+, and the sidecar environment become one reproducible image; large model weights remain replaceable host-mounted data.
- **Architectural:** the analyzer interface stays provider-neutral; OpenAI compatibility is a transport implementation, not a special case inside Gemini or Ollama.

## Architectural impact

### Container model

One container preserves Castwright's existing process-ownership model: the Node server serves the built frontend and supervises the Python TTS sidecar as its child. Splitting those processes into independent Compose services would break the current install/repair, restart, ownership, and model-lifecycle paths or require container-control plumbing that does not exist.

The image is multi-stage:

1. a Node build stage installs locked dependencies and builds frontend + server;
2. a Python 3.12 stage builds the sidecar venv for a selected accelerator profile without downloading model weights;
3. an Ubuntu 24.04 runtime carries ffmpeg 6+, the built Node app, the venv, and only runtime files.

`compose.yaml` is CPU-safe and binds the published port to host loopback by default. `compose.nvidia.yaml` is an explicit NVIDIA override that requests the GPU and builds the CUDA profile. This avoids making a missing NVIDIA runtime prevent the portable default from starting.

Host state lives under `${CASTWRIGHT_DATA_DIR:-./data}` and is mounted at `/data`; run tokens, handoff files, audio, and logs are below that root. `CASTWRIGHT_WORKSPACE_DIR` is a separate nested mount so an existing native workspace (books, backups, and designed voices) can be reused in place. `CASTWRIGHT_SETTINGS_DIR` similarly preserves the native `~/.castwright` settings directory. The existing Hugging Face cache is mounted separately through `CASTWRIGHT_HF_CACHE_DIR`, so rebuilding for another accelerator does not copy or download weights. The app image itself is replaceable. The entrypoint maps the container account to `PUID`/`PGID` without recursively changing existing book or cache ownership, and it leaves existing user-owned mount entries unchanged.

The bundled Python environment is immutable image content. Hugging Face model
weights persist in the configured host cache; workspace-scoped voices persist
under `/data`. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` provide a strict
no-download mode. Changing Python packages or repairing the baseline is an image
rebuild concern. Container deployment does not promise that an in-place venv
mutation survives replacing the image.

### Analyzer provider

`ANALYZER_PROVIDER=openai` is an ops-level override. It selects a new `OpenAICompatibleAnalyzer` regardless of the saved local/Gemini account default, so a Compose deployment is deterministic from environment alone. Required variables are:

- `OPENAI_BASE_URL` — API root such as `http://host.docker.internal:8000/v1`; a full `/chat/completions` URL is also accepted.
- `OPENAI_MODEL` — model id sent to the endpoint.
- `OPENAI_API_KEY` — optional; when blank, no `Authorization` header is sent, which supports local unauthenticated endpoints.

The analyzer uses the standard `POST /v1/chat/completions` messages contract, preserves the existing system/user/assistant validation-retry conversation, shares Castwright's Zod validation/repair functions, honours abort signals, and reports response bytes through the existing progress callback. It intentionally uses a non-streaming response for the broadest compatibility across local servers; the existing waiting timer still reports liveness while the request runs.

The public analysis metadata remains `engine: 'gemini' | 'local'` in this first slice. `openai` is a provider selection behind the existing remote/cloud engine envelope, avoiding a repository-wide OpenAPI/UI migration unrelated to endpoint connectivity. The effective model id is still reported verbatim.

`GET /api/user/settings` adds only safe read-only effective-provider metadata
for the browser: provider `openai` and the effective model id. This prevents the
analysis screen and setup checker from probing or waiting for Ollama when the
deployment override is active, including model ids containing `:`. The API key
and endpoint URL are never returned. This is runtime visibility, not a third
editable engine or a saved-settings schema migration.

### Migration and reversibility

No workspace or persisted user-settings schema changes. The settings response
gains only optional, read-only effective-provider metadata. Unset
`ANALYZER_PROVIDER` preserves the existing Ollama/Gemini selection. Removing
the Docker files, response overlay, and OpenAI selector branch restores the
prior deployment model without touching data.

## Invariants to preserve

1. `server/src/index.ts` must start with `import './load-env.js'`, and the container must launch with `server/` as its working directory.
2. The Node server remains the owner/supervisor of the TTS sidecar; `SIDECAR_VENV_DIR` points at the image venv.
3. `/data` is the persistent application root for books, settings, logs,
   handoff files, generated voice samples, and runtime model caches. The
   baseline venv is immutable image content; model caches are host-mounted.
4. The default Compose port is published on `127.0.0.1`, not every LAN interface. Remote exposure requires an explicit bind-address change and a trusted reverse proxy/VPN policy.
5. OpenAI credentials are runtime environment values only and never enter image layers, logs, fixtures, or version control.
6. An OpenAI-compatible response must pass the same Zod validation and one corrective retry as Gemini before it is persisted.
7. Ollama and Gemini behaviour is unchanged when `ANALYZER_PROVIDER` is unset.

## Test plan

### Automated coverage

- Vitest server (`server/src/analyzer/openai-compatible.test.ts`) — endpoint normalization, optional bearer auth, request messages, successful validation, corrective retry, malformed response, HTTP error, and abort propagation.
- Vitest server (`server/src/analyzer/select-analyzer.test.ts`) — explicit OpenAI provider precedence, required config errors, model metadata, and unchanged Ollama/Gemini paths.
- Focused server/frontend Vitest — safe resolved-provider metadata, no secret
  exposure, setup readiness without an Ollama/Gemini dependency, and an
  OpenAI model id containing `:` bypassing the local-model readiness gate.
- Node test (`scripts/tests/docker-config.test.mjs`) — Compose safety defaults, persistence mounts, healthcheck, Dockerfile runtime/ffmpeg/Python/sidecar wiring, and NVIDIA override.
- Container smoke — build/start with the NVIDIA override, wait for `/api/health`, assert CUDA and the mounted Qwen cache inside the container, and send a real Chat Completions request to the configured local endpoint.

### Manual acceptance walkthrough

1. Copy `.env.docker.example` to `.env`, set `ANALYZER_PROVIDER=openai`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.
2. Run `docker compose up --build -d`; open `http://127.0.0.1:8080/`; the setup/library UI loads and `/api/health` reports the env as loaded.
3. Import the canonical Coalfall fixture and run analysis against a test OpenAI-compatible endpoint; the selected model id appears in analysis progress and the cast is produced.
4. Restart/rebuild the container and confirm the imported book and user settings remain under the configured data directory.
5. On an NVIDIA Linux host with Container Toolkit installed, confirm readiness reports CUDA and the configured Hugging Face cache is visible without network access.

## Out of scope

- Publishing to GHCR or changing upstream `release.yml`; the fork can add registry credentials/repository naming after the local image contract is stable.
- OpenAI Responses API, Anthropic Messages, or provider-specific extensions.
- A third editable analysis-engine choice in Account/Setup UI; this slice is
  env-driven for reproducible headless deployment and exposes only read-only
  effective-provider status to those screens.
- Automatic public-LAN exposure, TLS termination, router/firewall changes, or reverse-proxy provisioning.
- Bundling Ollama; Docker deployments use the configured OpenAI-compatible endpoint, an external Ollama instance, or Gemini.

## Ship notes

Implemented on `feat/ops-docker-openai` in the Eventus112 fork worktree.

- Full NVIDIA image built from CUDA 12.8 and started on an RTX 5090; container
  Python reported `torch 2.11.0+cu128`, CUDA available, and the correct device.
- Existing native workspace, settings, and 78 GB Hugging Face cache were mounted
  in place. The library exposed both existing books; Qwen Base/1.7B/VoiceDesign
  snapshots were present with `HF_HUB_OFFLINE=1` and no model download.
- The configured llama-swap endpoint returned HTTP 200 for a real minimal Chat
  Completions request using model alias `castwright`; readiness and settings
  exposed only the safe effective provider/model metadata.
- Verification: Docker contract 5/5, frontend 4,823/4,823, server
  7,913/7,913 (9 skipped), production build/typecheck, Compose config/build
  check, and live container health/readiness. Node 26 frontend verification
  requires an isolated `--localstorage-file`; without it, 19 unrelated legacy
  tests fail because Node disables the global Web Storage object.

Publishing the branch/image and a full canonical-book analysis acceptance run
remain outside this local implementation pass.
