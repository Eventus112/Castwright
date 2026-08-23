# syntax=docker/dockerfile:1
# Castwright web server. Topology: the TTS sidecar runs on THIS machine (the
# `tts` compose service); the Ollama LLM daemon runs on ANOTHER tailnet
# machine (100.97.134.52:11435).
#
# URL wiring — NO app config needed: the server resolves sidecarUrl/ollamaUrl
# to factory localhost defaults (localhost:9000 / localhost:11434) because a
# missing user-settings.json is seeded in-process from defaults at boot, which
# shadows the LOCAL_TTS_URL / OLLAMA_URL env vars. Instead of fighting that,
# a tiny TCP relay (Dockerfile.proxy, the `proxy` compose service) parks on
# THIS container's loopback (shared network namespace) and forwards:
#   localhost:9000  -> tts:9000             (local sidecar)
#   localhost:11434 -> 100.97.134.52:11435  (remote Ollama)
# so the app's defaults just work. Zero changes to the Castwright source.
#
# Base is ubuntu:24.04 — the platform INSTALL.md validates on (its ffmpeg
# 6.1.1 meets the >= 6.0 support floor used for chapter MP3 encoding).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    HUSKY=0

# Node 24 (repo target; engines floor 22.22) + ffmpeg >= 6.0.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layers (cached until the lockfiles change).
COPY package.json package-lock.json ./
COPY server/package.json server/package-lock.json server/
RUN npm ci --no-audit --no-fund \
    && npm --prefix server ci --no-audit --no-fund

COPY . .

# Frontend bundle (dist/) + compiled server (server/dist/), then strip devDeps.
RUN npm run build \
    && npm prune --omit=dev --no-audit --no-fund \
    && npm --prefix server prune --omit=dev --no-audit --no-fund

# --- Runtime configuration ---------------------------------------------------
# LAN_HTTPS=0          prod defaults to mkcert HTTPS on :8443; with no certs
#                      the server degrades to a LOOPBACK bind and is
#                      unreachable from outside the container. Opt out => HTTP.
# BIND_HOST=0.0.0.0    plain-HTTP mode binds 127.0.0.1 by default (srv-19);
#                      Tailscale access needs all interfaces.
# WORKSPACE_DIR        books / voices — back it with a volume.
# USER_SETTINGS_FILE   web-UI account settings default to
#                      ~/.castwright/user-settings.json (homedir) — an
#                      ephemeral container layer wiped on every recreate.
#                      Pin into the /workspace volume so they persist.
# CASTWRIGHT_PROMPTS_DIR  user-forked analyzer prompts default to
#                      ~/.castwright/prompts — same persistence problem;
#                      pin into the volume too.
# DISABLE_AUTOSTART_SIDECAR=1  this image ships no Python runtime; do NOT try
#                      to spawn a local sidecar. All TTS traffic goes to
#                      LOCAL_TTS_URL instead.
# LOCAL_TTS_URL        remote TTS endpoint. The hostname MUST be the bare
#                      alias `tts` (the srv-21 guard rejects dotted hosts
#                      outside the private ranges above, incl. 100.64.0.0/10).
ENV NODE_ENV=production \
    PORT=8080 \
    LAN_HTTPS=0 \
    BIND_HOST=0.0.0.0 \
    WORKSPACE_DIR=/workspace \
    USER_SETTINGS_FILE=/workspace/user-settings.json \
    CASTWRIGHT_PROMPTS_DIR=/workspace/prompts \
    DISABLE_AUTOSTART_SIDECAR=1 \
    LOCAL_TTS_URL=http://tts:9000

# The server loads server/.env relative to its cwd (process.loadEnvFile never
# overrides real env vars, so the ENV block above wins either way). Shipping
# the file keeps the "running on DEFAULTS" boot warning out of the logs.
RUN printf '%s\n' \
      'PORT=8080' \
      'LAN_HTTPS=0' \
      'BIND_HOST=0.0.0.0' \
      'WORKSPACE_DIR=/workspace' \
      'USER_SETTINGS_FILE=/workspace/user-settings.json' \
      'CASTWRIGHT_PROMPTS_DIR=/workspace/prompts' \
      'DISABLE_AUTOSTART_SIDECAR=1' \
      'LOCAL_TTS_URL=http://tts:9000' \
      > server/.env

VOLUME ["/workspace"]
EXPOSE 8080

WORKDIR /app/server

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["node", "dist/index.js"]