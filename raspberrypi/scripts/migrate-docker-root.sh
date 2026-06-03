#!/usr/bin/env bash
# Migrate Docker + containerd root directory to a new location.
# Usage: sudo ./migrate-docker-root.sh /new_dir_structure
set -euo pipefail

NEW_BASE="${1:-/home/yisheng/Documents}"
NEW_DOCKER_ROOT="$NEW_BASE/docker"
NEW_CONTAINERD_ROOT="$NEW_BASE/containerd"
DAEMON_JSON="/etc/docker/daemon.json"
CONTAINERD_CONF="/etc/containerd/config.toml"

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (use sudo)." >&2
    exit 1
fi

# Set "data-root" in daemon.json without clobbering existing keys
# (the file may hold an nvidia runtime block we must preserve).
# Args: $1 = path to daemon.json, $2 = new data-root path
update_daemon_json() {
    local file="$1" root="$2"
    command -v jq >/dev/null || { echo "jq required: apt install jq" >&2; exit 1; }
    mkdir -p "$(dirname "$file")"
    local existing="{}"
    [[ -f "$file" ]] && existing="$(cat "$file")"
    jq --arg root "$root" '.["data-root"] = $root' <<<"$existing" > "$file"
}

echo "==> Target base directory: $NEW_BASE"
mkdir -p "$NEW_BASE"

echo "==> 1. Stopping docker + containerd services"
systemctl stop docker docker.socket containerd

echo "==> 2. Moving data directories"
if [[ -d /var/lib/docker && ! -d "$NEW_DOCKER_ROOT" ]]; then
    mv /var/lib/docker "$NEW_DOCKER_ROOT"
else
    echo "    skip docker move (source missing or target exists)"
fi
if [[ -d /var/lib/containerd && ! -d "$NEW_CONTAINERD_ROOT" ]]; then
    mv /var/lib/containerd "$NEW_CONTAINERD_ROOT"
else
    echo "    skip containerd move (source missing or target exists)"
fi

echo "==> 3.1 Updating $DAEMON_JSON"
update_daemon_json "$DAEMON_JSON" "$NEW_DOCKER_ROOT"

echo "==> 3.2 Updating $CONTAINERD_CONF"
# Set/replace the top-level `root = "..."` line, append if absent.
if [[ -f "$CONTAINERD_CONF" ]] && grep -qE '^\s*root\s*=' "$CONTAINERD_CONF"; then
    sed -i -E "s|^\s*root\s*=.*|root = \"$NEW_CONTAINERD_ROOT\"|" "$CONTAINERD_CONF"
else
    echo "root = \"$NEW_CONTAINERD_ROOT\"" >> "$CONTAINERD_CONF"
fi

echo "==> 4. Reloading systemd and starting services"
systemctl daemon-reload
systemctl start containerd docker

echo "==> 5. Validating new Docker root"
docker info -f '{{ .DockerRootDir }}'
