#!/usr/bin/env bash
# Bring up the full stack in the right build order.
#
# Build order matters because docker/animation-service.Dockerfile does
# FROM kimodo:1.0, and that image is built by upstream's text-encoder
# service.

set -euo pipefail

cd "$(dirname "$0")/.."

# upstream's Dockerfile COPYs kimodo-viser/ from its build context.
# kimodo-viser is a sibling repo to nv-tlabs/kimodo, not a submodule of it.
# Git won't let us nest a submodule inside our existing upstream/ submodule,
# so we plain-clone it here and keep it in sync. The directory is .gitignored.
if [ ! -d upstream/kimodo-viser ]; then
  echo "[0/3] First-time clone of nv-tlabs/kimodo-viser into upstream/..."
  git clone https://github.com/nv-tlabs/kimodo-viser.git upstream/kimodo-viser
fi

if [ ! -f .env ]; then
  echo "[!] No .env found. Copying .env.example → .env."
  echo "    Edit .env and set HF_TOKEN before continuing if text-encoder needs to download gated models."
  cp .env.example .env
fi

echo "[1/3] Building kimodo:1.0 (upstream base image)..."
docker compose build text-encoder

echo "[2/3] Building animation-service..."
docker compose build animation-service

echo "[3/3] Starting all services..."
docker compose up
