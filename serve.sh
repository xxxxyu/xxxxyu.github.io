#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

uv run --locked python -m scripts.prepare_assets
uv run --locked python -m scripts.prepare_assets --watch &
asset_watcher_pid=$!

cleanup() {
  if kill -0 "$asset_watcher_pid" 2>/dev/null; then
    kill "$asset_watcher_pid" 2>/dev/null || true
    wait "$asset_watcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

zola serve --extra-watch-path data "$@"
