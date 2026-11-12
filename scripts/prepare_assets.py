"""Prepare search and fonts in order: python -m scripts.prepare_assets."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

from scripts import build_font_subsets as fonts
from scripts import build_zh_search_index as search


def prepare_assets() -> None:
    if search.ensure_index():
        print("Chinese search index updated.", flush=True)
    required = fonts.source_characters()
    changed, cached = fonts.ensure_subsets(required)
    fonts.print_summary(required, changed, cached=cached)


def source_fingerprint() -> tuple[tuple[Path, bytes], ...]:
    fingerprint = []
    for path in fonts.source_text_files():
        # Watch the font inputs, but never read or react to generated outputs.
        if path == search.OUTPUT_PATH:
            continue
        try:
            fingerprint.append((path, hashlib.sha256(path.read_bytes()).digest()))
        except FileNotFoundError:
            # Editors can rename/delete a file between discovery and reading.
            continue
    return tuple(fingerprint)


def watch(interval: float) -> int:
    previous = None
    pending = None
    last_error = None
    print("Watching site sources for search and font updates…", flush=True)
    try:
        while True:
            try:
                current = source_fingerprint()
                if current != pending:
                    pending = current
                elif current != previous:
                    # Wait for two matching snapshots to coalesce rapid saves.
                    prepare_assets()
                    previous = current
                    last_error = None
            except Exception as error:  # retry partial saves without stopping Zola
                message = str(error)
                if message != last_error:
                    print(f"Asset preparation failed: {message}", file=sys.stderr, flush=True)
                    last_error = message
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true", help="prepare assets after source edits settle")
    mode.add_argument(
        "--print-font-cache-key", action="store_true",
        help="prepare the search index and print the font cache key before restoring CI caches",
    )
    parser.add_argument("--interval", type=float, default=0.75, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

    try:
        if args.watch:
            return watch(args.interval)
        if args.print_font_cache_key:
            search.ensure_index()
            print(fonts.cache_key(fonts.source_characters()))
        else:
            prepare_assets()
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
