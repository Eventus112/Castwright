#!/usr/bin/env bash
set -euo pipefail
umask 0027

readonly target_uid="${PUID:-1000}"
readonly target_gid="${PGID:-1000}"

if [[ ! "$target_uid" =~ ^[0-9]+$ ]] || [[ "$target_uid" == "0" ]]; then
  echo "[entrypoint] PUID must be a positive numeric user id" >&2
  exit 64
fi
if [[ ! "$target_gid" =~ ^[0-9]+$ ]] || [[ "$target_gid" == "0" ]]; then
  echo "[entrypoint] PGID must be a positive numeric group id" >&2
  exit 64
fi

if [[ "$(id -u)" != "0" ]]; then
  echo "[entrypoint] start the container as root; PUID/PGID are applied before dropping privileges" >&2
  exit 77
fi

current_uid="$(id -u castwright)"
current_gid="$(id -g castwright)"

if [[ "$target_gid" != "$current_gid" ]]; then
  existing_group="$(getent group "$target_gid" | cut -d: -f1 || true)"
  if [[ -n "$existing_group" ]]; then
    usermod --gid "$existing_group" castwright
  else
    groupmod --gid "$target_gid" castwright
  fi
fi

if [[ "$target_uid" != "$current_uid" ]]; then
  if getent passwd "$target_uid" >/dev/null; then
    echo "[entrypoint] PUID $target_uid is already assigned inside the image" >&2
    exit 65
  fi
  usermod --uid "$target_uid" castwright
fi

readonly runtime_gid="$(id -g castwright)"

# Update only the application-owned directory entries. Existing books and
# caches keep their ownership; startup never walks a potentially large bind.
for dir in \
  /data \
  /data/audio \
  /data/audio/voices \
  /data/cache \
  /data/cache/torch \
  /data/handoff \
  /data/handoff/cache \
  /data/handoff/inbox \
  /data/handoff/outbox \
  /data/home \
  /data/logs \
  /data/run \
  /data/settings \
  /data/share \
  /data/workspace
do
  if [[ ! -d "$dir" ]]; then
    install -d -o "$target_uid" -g "$runtime_gid" -m 0750 "$dir"
  elif [[ "$(stat -c '%u' "$dir")" == "0" ]]; then
    # Compose creates a missing bind source as a root-owned directory. Change
    # only that directory entry; existing user-owned mounts retain their mode
    # and ownership, and large trees are never traversed.
    chown "$target_uid:$runtime_gid" "$dir"
  fi
done

exec setpriv \
  --reuid="$target_uid" \
  --regid="$runtime_gid" \
  --init-groups \
  "$@"
