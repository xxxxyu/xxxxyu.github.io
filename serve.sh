#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

uv run --locked python scripts/build_font_subsets.py
uv run --locked python scripts/build_font_subsets.py --watch &
font_watcher_pid=$!

cleanup() {
  if kill -0 "$font_watcher_pid" 2>/dev/null; then
    kill "$font_watcher_pid" 2>/dev/null || true
    wait "$font_watcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

zola serve --extra-watch-path data "$@"
