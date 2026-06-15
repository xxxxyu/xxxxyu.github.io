#!/usr/bin/env python3
"""Optimize local source images for an Outside collection.

This script keeps camera originals outside git. Run it on any machine that has
some source images; commit only the generated files under static/img/outside/
and data/outside/.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FORMATS = ("jpg", "webp")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create web-ready images and a gallery manifest for /outside/."
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection slug, e.g. city-light-studies.",
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Local directory containing source images. This directory is not committed.",
    )
    parser.add_argument(
        "--cover",
        help="Source filename to use as cover. Defaults to the first image.",
    )
    parser.add_argument(
        "--mode",
        choices=("replace", "append"),
        default="replace",
        help="Replace or append to data/outside/<collection>.toml.",
    )
    parser.add_argument(
        "--name-mode",
        choices=("stem", "sequence"),
        default="stem",
        help="Use source filename stems or frame-01 style output names.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First index for --name-mode sequence.",
    )
    parser.add_argument(
        "--include-cover-in-gallery",
        action="store_true",
        help="Also include the cover source image as a gallery frame.",
    )
    parser.add_argument(
        "--gallery-layout",
        choices=("pair", "full"),
        default="pair",
        help="Default layout written for generated gallery items.",
    )
    parser.add_argument(
        "--cover-max",
        type=int,
        default=1800,
        help="Maximum cover image edge in pixels.",
    )
    parser.add_argument(
        "--gallery-max",
        type=int,
        default=2200,
        help="Maximum gallery image edge in pixels.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=84,
        help="JPEG/WebP quality, 1-95.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated output formats: jpg,webp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing optimized images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write data/outside/<collection>.toml.",
    )
    return parser.parse_args()


def require_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError:
        print(
            "Pillow is required for image optimization.\n"
            "Install it with: python3 -m pip install Pillow",
            file=sys.stderr,
        )
        sys.exit(2)
    return Image, ImageOps


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "image"


def frame_base_from_stem(stem: str) -> str:
    slug = slugify(stem)
    return slug if slug.startswith("frame-") else f"frame-{slug}"


def find_sources(src_dir: Path) -> list[Path]:
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")
    images = [
        path
        for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def choose_cover(images: list[Path], cover_name: str | None) -> Path:
    if not images:
        raise SystemExit("No source images found.")
    if not cover_name:
        return images[0]
    matches = [path for path in images if path.name == cover_name or path.stem == cover_name]
    if not matches:
        raise SystemExit(f"Cover source not found in --src: {cover_name}")
    return matches[0]


def output_paths(collection: str, base: str, formats: set[str]) -> dict[str, Path]:
    out_dir = ROOT / "static" / "img" / "outside" / collection
    paths: dict[str, Path] = {}
    if "jpg" in formats or "jpeg" in formats:
        paths["jpg"] = out_dir / f"{base}.jpg"
    if "webp" in formats:
        paths["webp"] = out_dir / f"{base}.webp"
    return paths


def resize_and_save(src: Path, outputs: dict[str, Path], max_edge: int, quality: int, force: bool, dry_run: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing file(s): {names}. Use --force.")

    for path in outputs.values():
        print(f"write {path.relative_to(ROOT)}")

    if dry_run:
        return

    Image, ImageOps = require_pillow()
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        for fmt, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "jpg":
                image.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
            elif fmt == "webp":
                image.save(path, "WEBP", quality=quality, method=6)


def public_path(path: Path) -> str:
    rel = path.relative_to(ROOT / "static")
    return "/" + rel.as_posix()


def manifest_block(image_paths: dict[str, Path], label: str, layout: str) -> str:
    lines = ["[[gallery]]"]
    if "jpg" in image_paths:
        lines.append(f'image = "{public_path(image_paths["jpg"])}"')
    if "webp" in image_paths:
        lines.append(f'image_webp = "{public_path(image_paths["webp"])}"')
    lines.append(f'layout = "{layout}"')
    lines.append(f'label = "{label}"')
    lines.append('caption = ""')
    return "\n".join(lines) + "\n"


def write_manifest(collection: str, blocks: list[str], mode: str, dry_run: bool) -> Path:
    manifest = ROOT / "data" / "outside" / f"{collection}.toml"
    text = "\n".join(blocks)
    if mode == "append" and manifest.exists():
        existing = manifest.read_text()
        text = existing.rstrip() + "\n\n" + text

    print(f"write {manifest.relative_to(ROOT)}")
    if not dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(text, encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    collection = slugify(args.collection)
    src_dir = args.src.expanduser().resolve()
    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    unsupported = formats - {"jpg", "jpeg", "webp"}
    if unsupported:
        raise SystemExit(f"Unsupported format(s): {', '.join(sorted(unsupported))}")
    if not formats:
        formats = set(DEFAULT_FORMATS)

    images = find_sources(src_dir)
    cover = choose_cover(images, args.cover)

    cover_outputs = output_paths(collection, "cover", formats)
    resize_and_save(cover, cover_outputs, args.cover_max, args.quality, args.force, args.dry_run)

    gallery_sources = images if args.include_cover_in_gallery else [p for p in images if p != cover]
    blocks: list[str] = []
    for offset, image in enumerate(gallery_sources):
        if args.name_mode == "sequence":
            base = f"frame-{args.start_index + offset:02d}"
            label = f"Frame {args.start_index + offset:02d}"
        else:
            base = frame_base_from_stem(image.stem)
            label = image.stem
        paths = output_paths(collection, base, formats)
        resize_and_save(image, paths, args.gallery_max, args.quality, args.force, args.dry_run)
        blocks.append(manifest_block(paths, label, args.gallery_layout))

    if not args.no_manifest:
        write_manifest(collection, blocks, args.mode, args.dry_run)

    print("\nFront matter fields:")
    if "jpg" in cover_outputs:
        print(f'cover = "{public_path(cover_outputs["jpg"])}"')
    if "webp" in cover_outputs:
        print(f'cover_webp = "{public_path(cover_outputs["webp"])}"')
    if not args.no_manifest:
        print(f'gallery_manifest = "data/outside/{collection}.toml"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
