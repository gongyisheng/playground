#!/usr/bin/env bash
# Move Docker + containerd data dirs. Run with sudo.
set -euo pipefail

DOCKER_DST="/home/yisheng/docker"
CONTAINERD_DST="/home/yisheng/containerd"
DOCKER_SRC="/var/lib/docker"
CONTAINERD_SRC="/var/lib/containerd"

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }

echo "==> Stopping services"
systemctl stop docker docker.socket containerd

echo "==> Copying data (preserving ownership/perms)"
mkdir -p "$DOCKER_DST" "$CONTAINERD_DST"
rsync -aP "$DOCKER_SRC/"     "$DOCKER_DST/"
rsync -aP "$CONTAINERD_SRC/" "$CONTAINERD_DST/"

echo "==> Updating /etc/docker/daemon.json"
mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
  cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
  # set/replace data-root, requires jq
  if command -v jq >/dev/null; then
    jq --arg p "$DOCKER_DST" '."data-root"=$p' /etc/docker/daemon.json.bak > /etc/docker/daemon.json
  else
    echo "!! jq not found and daemon.json exists. Edit it manually:"
    echo '   add  "data-root": "'"$DOCKER_DST"'"'
    exit 1
  fi
else
  printf '{\n  "data-root": "%s"\n}\n' "$DOCKER_DST" > /etc/docker/daemon.json
fi

echo "==> Updating /etc/containerd/config.toml"
mkdir -p /etc/containerd
[[ -f /etc/containerd/config.toml ]] || containerd config default > /etc/containerd/config.toml
cp /etc/containerd/config.toml /etc/containerd/config.toml.bak
if grep -qE '^\s*root\s*=' /etc/containerd/config.toml; then
  # active root line -> replace value
  sed -i -E "s|^(\s*root\s*=).*|\1 \"$CONTAINERD_DST\"|" /etc/containerd/config.toml
elif grep -qE '^\s*#\s*root\s*=' /etc/containerd/config.toml; then
  # commented root line -> uncomment and set value
  sed -i -E "s|^\s*#\s*root\s*=.*|root = \"$CONTAINERD_DST\"|" /etc/containerd/config.toml
else
  # no root line at all -> append as top-level key
  printf '\nroot = "%s"\n' "$CONTAINERD_DST" >> /etc/containerd/config.toml
fi

echo "==> Restarting services"
systemctl start containerd docker

echo "==> Verifying"
docker info 2>/dev/null | grep "Docker Root Dir"
echo "containerd root -> $(grep -E '^\s*root\s*=' /etc/containerd/config.toml)"

cat <<EOF

Done. Verify your containers/images look right, then reclaim old space:
  sudo mv $DOCKER_SRC ${DOCKER_SRC}.old
  sudo mv $CONTAINERD_SRC ${CONTAINERD_SRC}.old
  # delete the .old dirs after a few days of stable operation