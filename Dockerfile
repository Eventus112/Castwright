# syntax=docker/dockerfile:1

ARG NODE_IMAGE=node:24-bookworm-slim
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${NODE_IMAGE} AS node-base

FROM node-base AS app-build
WORKDIR /build

COPY package.json package-lock.json ./
COPY server/package.json server/package-lock.json ./server/
RUN HUSKY=0 npm ci \
    && HUSKY=0 npm --prefix server ci

COPY . .
RUN npm run build \
    && npm --prefix server prune --omit=dev

FROM ${RUNTIME_BASE} AS sidecar-build
ARG ACCELERATOR=cpu

COPY --from=node-base /usr/local/ /usr/local/
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      python3 \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY package.json ./
COPY scripts/lib/is-main-module.mjs ./scripts/lib/is-main-module.mjs
COPY server/tts-sidecar ./server/tts-sidecar

ENV ACCELERATOR=${ACCELERATOR} \
    SIDECAR_VENV_DIR=/opt/castwright/sidecar-venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build the Python runtime only. Model weights are mutable, large assets and
# belong in host-mounted caches; baking them into an image would download them
# again for each accelerator/base-image variant.
RUN node server/tts-sidecar/scripts/bootstrap-venv.mjs python3

FROM ${RUNTIME_BASE} AS runtime-base

COPY --from=node-base /usr/local/ /usr/local/
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      ffmpeg \
      libgomp1 \
      libsndfile1 \
      lsof \
      python3 \
      util-linux \
    && existing_group="$(getent group 1000 | cut -d: -f1 || true)" \
    && if [ -n "$existing_group" ]; then groupmod --new-name castwright "$existing_group"; else groupadd --gid 1000 castwright; fi \
    && existing_user="$(getent passwd 1000 | cut -d: -f1 || true)" \
    && if [ -n "$existing_user" ]; then usermod --login castwright --home /nonexistent --groups '' --shell /usr/sbin/nologin "$existing_user"; else useradd --uid 1000 --gid castwright --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin castwright; fi \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS runtime
ARG ACCELERATOR=cpu

WORKDIR /app
COPY --from=app-build --chown=castwright:castwright /build/dist ./dist
COPY --from=app-build --chown=castwright:castwright /build/server/dist ./server/dist
COPY --from=app-build --chown=castwright:castwright /build/server/node_modules ./server/node_modules
COPY --from=app-build --chown=castwright:castwright /build/server/package.json ./server/package.json
COPY --chown=castwright:castwright docker/server.env ./server/.env
COPY --from=app-build --chown=castwright:castwright /build/package.json ./package.json
COPY --from=app-build --chown=castwright:castwright /build/build-manifest.json ./build-manifest.json
COPY --from=app-build --chown=castwright:castwright /build/RELEASE_NOTES.md ./RELEASE_NOTES.md
COPY --from=app-build --chown=castwright:castwright /build/skills ./skills
COPY --from=app-build --chown=castwright:castwright /build/samples ./samples
COPY --from=app-build --chown=castwright:castwright /build/scripts/lib/is-main-module.mjs ./scripts/lib/is-main-module.mjs
COPY --from=sidecar-build --chown=castwright:castwright /build/server/tts-sidecar ./server/tts-sidecar
COPY --from=sidecar-build --chown=castwright:castwright /opt/castwright/sidecar-venv /opt/castwright/sidecar-venv

RUN rm -rf /app/server/handoff \
    && ln -s /data/handoff /app/server/handoff \
    && ln -s /data/audio /app/server/audio \
    && ln -s /opt/castwright/sidecar-venv /app/server/tts-sidecar/.venv

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/castwright-entrypoint

ENV ACCELERATOR=${ACCELERATOR} \
    APP_LOG_DIR=/data/logs \
    APP_RUN_DIR=/data/run \
    BIND_HOST=0.0.0.0 \
    HF_HOME=/data/cache/huggingface \
    HOME=/data/home \
    LAN_HTTPS=0 \
    LOCAL_TTS_HOST=127.0.0.1 \
    NODE_ENV=production \
    PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIDECAR_NEVER_ADOPT=1 \
    SIDECAR_VENV_DIR=/opt/castwright/sidecar-venv \
    TORCH_HOME=/data/cache/torch \
    USER_SETTINGS_FILE=/data/settings/user-settings.json \
    WORKSPACE_DIR=/data/workspace \
    XDG_CACHE_HOME=/data/cache \
    XDG_DATA_HOME=/data/share

VOLUME ["/data"]
EXPOSE 8080
WORKDIR /app/server
ENTRYPOINT ["castwright-entrypoint"]
CMD ["node", "dist/index.js"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD node -e "fetch('http://127.0.0.1:8080/api/health').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"
