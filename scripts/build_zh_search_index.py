"""Build the lightweight Chinese search document store used by search.js."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content"
OUTPUT_PATH = REPO_ROOT / "static" / "search_index.zh.json"


def split_content(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("+++", maxsplit=2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Invalid TOML front matter in {path.relative_to(REPO_ROOT)}")
    return tomllib.loads(parts[1]), parts[2]


def output_path(source: Path, metadata: dict[str, Any]) -> str:
    explicit = metadata.get("path")
    if isinstance(explicit, str) and explicit:
        return "/" + explicit.strip("/") + "/"

    relative = source.relative_to(CONTENT_ROOT)
    if source.name.startswith("_index.") or source.name.startswith("index."):
        parts = relative.parent.parts
    else:
        filename = source.name.removesuffix(".zh.md").removesuffix(".md")
        parts = (*relative.parent.parts, filename)
    suffix = "/".join(parts)
    prefix = "/zh/" if source.name.endswith(".zh.md") else "/"
    return prefix + (suffix + "/" if suffix else "")


def plain_text(markdown: str) -> str:
    text = re.sub(r"```[^\n]*\n|```", " ", markdown)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`*_>#~-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_index() -> dict[str, list[dict[str, str]]]:
    config = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))
    base_url = config["base_url"].rstrip("/")
    documents = []
    default_sources = [
        path for path in sorted(CONTENT_ROOT.rglob("*.md"))
        if not path.name.endswith(".zh.md")
    ]
    for default_source in default_sources:
        default_metadata, _ = split_content(default_source)
        if default_metadata.get("draft") is True or default_metadata.get("in_search_index") is False:
            continue
        translated_source = default_source.with_name(
            default_source.name.removesuffix(".md") + ".zh.md"
        )
        source = translated_source if translated_source.is_file() else default_source
        metadata, body = split_content(source)
        if source == translated_source and (
            metadata.get("draft") is True or metadata.get("in_search_index") is False
        ):
            source = default_source
            metadata, body = split_content(source)
        path = output_path(source, metadata)
        documents.append(
            {
                "id": base_url + path,
                "path": path,
                "title": str(metadata.get("title", "")),
                "description": str(metadata.get("description", "")),
                "body": plain_text(body),
            }
        )
    return {"documents": documents}


def rendered_index() -> str:
    return json.dumps(build_index(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed index is stale")
    args = parser.parse_args()
    rendered = rendered_index()
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale; run this script without --check", file=sys.stderr)
            return 1
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
