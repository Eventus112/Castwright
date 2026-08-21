import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path) => readFileSync(resolve(root, path), 'utf8');

const dockerfile = read('Dockerfile');
const compose = read('compose.yaml');
const nvidia = read('compose.nvidia.yaml');
const entrypoint = read('docker/entrypoint.sh');

test('base Compose is CPU-safe, loopback-only, and keeps all mutable state under /data', () => {
  assert.match(compose, /ACCELERATOR: cpu/);
  assert.match(compose, /RUNTIME_BASE: ubuntu:24\.04/);
  assert.match(
    compose,
    /\$\{CASTWRIGHT_BIND_ADDRESS:-127\.0\.0\.1\}:\$\{CASTWRIGHT_PORT:-8080\}:8080/,
  );
  assert.match(compose, /source: \$\{CASTWRIGHT_DATA_DIR:-\.\/data\}[\s\S]*target: \/data/);
  assert.match(
    compose,
    /source: \$\{CASTWRIGHT_WORKSPACE_DIR:-\.\/data\/workspace\}[\s\S]*target: \/data\/workspace/,
  );
  assert.match(
    compose,
    /source: \$\{CASTWRIGHT_SETTINGS_DIR:-\.\/data\/settings\}[\s\S]*target: \/data\/settings/,
  );
  assert.doesNotMatch(compose, /capabilities:\s*\n\s*- gpu/);

  for (const invariant of [
    'NODE_ENV: production',
    "LAN_HTTPS: '0'",
    'BIND_HOST: 0.0.0.0',
    'SIDECAR_VENV_DIR: /opt/castwright/sidecar-venv',
    'WORKSPACE_DIR: /data/workspace',
    'USER_SETTINGS_FILE: /data/settings/user-settings.json',
    'APP_LOG_DIR: /data/logs',
    'APP_RUN_DIR: /data/run',
  ]) {
    assert.ok(compose.includes(invariant), `missing Compose invariant: ${invariant}`);
  }
  assert.match(compose, /\/api\/health/);
  assert.match(compose, /cap_drop:[\s\S]*- ALL/);
  assert.match(
    compose,
    /cap_add:[\s\S]*- CHOWN[\s\S]*- DAC_OVERRIDE[\s\S]*- FOWNER[\s\S]*- SETGID[\s\S]*- SETUID/,
  );
  assert.match(compose, /security_opt:[\s\S]*- no-new-privileges:true/);
  assert.match(compose, /tmpfs:[\s\S]*\/tmp:rw,nosuid,nodev,mode=1777/);
  assert.match(compose, /stop_grace_period: 3m30s/);
});

test('OpenAI-compatible settings remain runtime-only Compose values', () => {
  for (const key of ['ANALYZER_PROVIDER', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY']) {
    assert.ok(compose.includes(`${key}: ` + '${' + `${key}:-}`));
    assert.doesNotMatch(dockerfile, new RegExp(`(?:ARG|ENV)\\s+${key}(?:=|\\s)`));
  }
});

test('Dockerfile builds on Node 24 and Ubuntu 24.04 with the repository sidecar bootstrap', () => {
  assert.match(dockerfile, /ARG NODE_IMAGE=node:24-/);
  assert.match(dockerfile, /ARG RUNTIME_BASE=ubuntu:24\.04/);
  assert.match(dockerfile, /ARG ACCELERATOR=cpu/);
  assert.match(dockerfile, /apt-get install[\s\S]*ffmpeg/);
  assert.match(dockerfile, /apt-get install[\s\S]*lsof/);
  assert.match(dockerfile, /getent group 1000[\s\S]*groupmod --new-name castwright/);
  assert.match(dockerfile, /getent passwd 1000[\s\S]*usermod --login castwright/);
  assert.match(dockerfile, /usermod --login castwright[\s\S]*--groups ''/);
  assert.match(dockerfile, /--home \/nonexistent/);
  assert.match(dockerfile, /bootstrap-venv\.mjs python3/);
  assert.doesNotMatch(dockerfile, /RUN[^\n]*install-kokoro\.mjs/);
  assert.match(dockerfile, /docker\/server\.env \.\/server\/\.env/);
  assert.match(dockerfile, /SIDECAR_VENV_DIR=\/opt\/castwright\/sidecar-venv/);
  assert.match(
    dockerfile,
    /ln -s \/opt\/castwright\/sidecar-venv \/app\/server\/tts-sidecar\/\.venv/,
  );
  assert.match(dockerfile, /ln -s \/data\/audio \/app\/server\/audio/);
  assert.match(
    dockerfile,
    /--chown=castwright:castwright \/opt\/castwright\/sidecar-venv \/opt\/castwright\/sidecar-venv/,
  );
  assert.match(dockerfile, /WORKDIR \/app\/server/);
  assert.match(dockerfile, /CMD \["node", "dist\/index\.js"\]/);
});

test('NVIDIA support is an explicit CUDA override with a GPU reservation', () => {
  assert.match(nvidia, /ACCELERATOR: nvidia/);
  assert.match(nvidia, /nvidia\/cuda:12\.8\.1-cudnn-runtime-ubuntu24\.04/);
  assert.match(nvidia, /reservations:[\s\S]*devices:[\s\S]*driver: nvidia/);
  assert.match(nvidia, /capabilities:[\s\S]*- gpu/);
  assert.match(
    nvidia,
    /source: \$\{CASTWRIGHT_HF_CACHE_DIR:\?[^}]+\}[\s\S]*target: \/data\/cache\/huggingface/,
  );
});

test('entrypoint maps PUID/PGID without recursively changing persistent data', () => {
  assert.match(entrypoint, /PUID:-1000/);
  assert.match(entrypoint, /PGID:-1000/);
  assert.match(entrypoint, /usermod --uid/);
  assert.match(entrypoint, /groupmod --gid/);
  assert.match(entrypoint, /exec setpriv/);
  assert.match(entrypoint, /stat -c '%u'/);
  assert.match(entrypoint, /umask 0027/);
  assert.match(entrypoint, /\/data\/audio/);
  assert.doesNotMatch(entrypoint, /chown\s+(?:-[^\s]*R[^\s]*\s+|--recursive)/);
});
