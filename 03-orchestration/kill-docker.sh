#!/usr/bin/env bash
#
# docker-cleanup.sh — stop containers, remove images and volumes.
# Usage:
#   ./docker-cleanup.sh            # interactive, asks before each step
#   ./docker-cleanup.sh --yes      # skip prompts, do everything
#   ./docker-cleanup.sh --compose  # only act on docker compose project in CWD
#   ./docker-cleanup.sh --all      # nuclear: includes images, networks, build cache

set -euo pipefail

ASSUME_YES=false
COMPOSE_ONLY=false
NUCLEAR=false

for arg in "$@"; do
  case "$arg" in
    --yes|-y)     ASSUME_YES=true ;;
    --compose|-c) COMPOSE_ONLY=true ;;
    --all|-a)     NUCLEAR=true ;;
    --help|-h)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

confirm() {
  local prompt="$1"
  if $ASSUME_YES; then return 0; fi
  read -r -p "$prompt [y/N] " response
  [[ "$response" =~ ^[Yy]$ ]]
}

# ---------- Compose-only mode ----------
if $COMPOSE_ONLY; then
  if [[ ! -f docker-compose.yml && ! -f compose.yml && ! -f docker-compose.yaml && ! -f compose.yaml ]]; then
    echo "No docker-compose file found in $(pwd)." >&2
    exit 1
  fi
  echo "==> docker compose project in $(pwd)"
  if confirm "Stop and remove containers, networks, and volumes for this project?"; then
    docker compose down --volumes --remove-orphans --rmi local
    echo "Done."
  fi
  exit 0
fi

# ---------- Global cleanup ----------
echo "==> Docker cleanup (global)"
docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' | head -20
echo

# 1. Containers
if [[ -n "$(docker ps -aq)" ]]; then
  if confirm "Stop and remove ALL containers ($(docker ps -aq | wc -l | tr -d ' '))?"; then
    docker ps -aq | xargs -r docker stop
    docker ps -aq | xargs -r docker rm -v
    echo "Containers removed."
  fi
else
  echo "No containers to remove."
fi

# 2. Volumes
if [[ -n "$(docker volume ls -q)" ]]; then
  if confirm "Remove ALL volumes ($(docker volume ls -q | wc -l | tr -d ' '))? This deletes data."; then
    docker volume ls -q | xargs -r docker volume rm
    echo "Volumes removed."
  fi
else
  echo "No volumes to remove."
fi

# 3. Images
if [[ -n "$(docker images -q)" ]]; then
  if confirm "Remove ALL images ($(docker images -q | wc -l | tr -d ' '))? Will re-download on next pull."; then
    docker images -q | xargs -r docker rmi -f
    echo "Images removed."
  fi
else
  echo "No images to remove."
fi

# 4. Nuclear option — networks and build cache too
if $NUCLEAR; then
  if confirm "Also prune custom networks and build cache?"; then
    docker network prune -f
    docker builder prune -af
    echo "Networks and build cache pruned."
  fi
fi

echo
echo "==> Done. Disk usage:"
docker system df
