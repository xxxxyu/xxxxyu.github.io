"""Build the lightweight Chinese search document store used by search.js."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = Path("static/search_index.zh.json")
OUTPUT_PATH = REPO_ROOT / INDEX_PATH


def split_content(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("+++", maxsplit=2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Invalid TOML front matter in {path}")
    return tomllib.loads(parts[1]), parts[2]


def output_path(source: Path, metadata: dict[str, Any], content_root: Path) -> str:
    explicit = metadata.get("path")
    if isinstance(explicit, str) and explicit:
        return "/" + explicit.strip("/") + "/"

    relative = source.relative_to(content_root)
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


def is_searchable(metadata: dict[str, Any]) -> bool:
    return metadata.get("draft") is not True and metadata.get("in_search_index") is not False


def build_index(repo_root: Path = REPO_ROOT) -> dict[str, list[dict[str, str]]]:
    config = tomllib.loads((repo_root / "config.toml").read_text(encoding="utf-8"))
    content_root = repo_root / "content"
    base_url = config["base_url"].rstrip("/")
    documents = []
    default_sources = [
        path for path in sorted(content_root.rglob("*.md"))
        if not path.name.endswith(".zh.md")
    ]
    for default_source in default_sources:
        metadata, body = split_content(default_source)
        if not is_searchable(metadata):
            continue
        source = default_source
        translated_source = default_source.with_name(
            default_source.name.removesuffix(".md") + ".zh.md"
        )
        if translated_source.is_file():
            translated_metadata, translated_body = split_content(translated_source)
            if is_searchable(translated_metadata):
                source, metadata, body = translated_source, translated_metadata, translated_body
        path = output_path(source, metadata, content_root)
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


def rendered_index(repo_root: Path = REPO_ROOT) -> str:
    return json.dumps(build_index(repo_root), ensure_ascii=False, indent=2) + "\n"


def ensure_index(repo_root: Path = REPO_ROOT) -> bool:
    """Publish a complete index only when its contents change."""
    rendered = rendered_index(repo_root)
    output = repo_root / INDEX_PATH
    if output.is_file() and output.read_text(encoding="utf-8") == rendered:
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(rendered, encoding="utf-8", newline="\n")
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the generated index is missing or stale")
    args = parser.parse_args()
    if args.check:
        rendered = rendered_index()
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale; run this script without --check", file=sys.stderr)
            return 1
        return 0
    ensure_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
