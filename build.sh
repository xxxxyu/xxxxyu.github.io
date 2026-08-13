#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

uv run --locked python scripts/build_font_subsets.py
zola build "$@"
uv run --locked python scripts/build_font_subsets.py --check-rendered public
