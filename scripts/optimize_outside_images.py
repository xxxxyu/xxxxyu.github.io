#!/usr/bin/env python3
"""Optimize local source images for an Outside collection.

This script keeps camera originals outside git. Run it on any machine that has
some source images; commit only the generated files under static/img/outside/
and data/outside/.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FORMATS = ("jpg", "webp")
DEFAULT_IMAGE_MAX = 2160
DEFAULT_WATERMARK = ROOT / "static" / "img" / "identity" / "logo-libre-baskerville-bold-dark.svg"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


@dataclass(frozen=True)
class WatermarkOptions:
    path: Path
    opacity: float
    width_ratio: float
    margin_ratio: float
    bottom_margin_ratio: float
    shadow_opacity: float
    shadow_blur_ratio: float
    shadow_offset_ratio: float


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
        "--order",
        help=(
            "Comma-separated source filenames or stems for custom gallery order. "
            "Unlisted images are appended in filename order."
        ),
    )
    parser.add_argument(
        "--include-cover-in-gallery",
        action="store_true",
        help="Deprecated: cover handling is automatic.",
    )
    parser.add_argument(
        "--exclude-cover-from-gallery",
        action="store_true",
        help="Use the cover source only for the unwatermarked cover image.",
    )
    parser.add_argument(
        "--gallery-layout",
        choices=("pair", "center", "full"),
        default="pair",
        help="Default layout written for generated gallery items.",
    )
    parser.add_argument(
        "--cover-max",
        type=int,
        default=DEFAULT_IMAGE_MAX,
        help="Maximum cover image edge in pixels.",
    )
    parser.add_argument(
        "--gallery-max",
        type=int,
        default=DEFAULT_IMAGE_MAX,
        help="Maximum gallery image edge in pixels.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=88,
        help="JPEG/WebP quality, 1-95.",
    )
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Place the site logo watermark in the lower-right corner.",
    )
    parser.add_argument(
        "--watermark-path",
        type=Path,
        default=DEFAULT_WATERMARK,
        help="SVG watermark path used with --watermark.",
    )
    parser.add_argument(
        "--watermark-opacity",
        type=float,
        default=0.38,
        help="Watermark opacity, 0-1.",
    )
    parser.add_argument(
        "--watermark-width-ratio",
        type=float,
        default=0.032,
        help="Watermark width as a ratio of the output image width.",
    )
    parser.add_argument(
        "--watermark-margin-ratio",
        type=float,
        default=0.010,
        help="Watermark margin as a ratio of the output image width.",
    )
    parser.add_argument(
        "--watermark-bottom-margin-ratio",
        type=float,
        default=0.004,
        help="Watermark bottom margin as a ratio of the output image width.",
    )
    parser.add_argument(
        "--watermark-shadow-opacity",
        type=float,
        default=0.18,
        help="Dark watermark shadow opacity, 0-1.",
    )
    parser.add_argument(
        "--watermark-shadow-blur-ratio",
        type=float,
        default=0.018,
        help="Watermark shadow blur radius as a ratio of watermark width.",
    )
    parser.add_argument(
        "--watermark-shadow-offset-ratio",
        type=float,
        default=0.010,
        help="Watermark shadow offset as a ratio of watermark width.",
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
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        print(
            "Pillow is required for image optimization.\n"
            "Install dependencies with: uv sync",
            file=sys.stderr,
        )
        sys.exit(2)
    return Image, ImageEnhance, ImageFilter, ImageOps


def require_cairosvg():
    try:
        import cairosvg
    except ImportError:
        print(
            "CairoSVG is required for SVG watermarks.\n"
            "Install dependencies with: uv sync",
            file=sys.stderr,
        )
        sys.exit(2)
    return cairosvg


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "image"


def image_base_from_stem(stem: str) -> str:
    slug = slugify(stem)
    return f"{slug}-image" if slug == "cover" else slug


def find_sources(src_dir: Path) -> list[Path]:
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")
    images = [
        path
        for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images, key=lambda p: p.name.lower())


def apply_custom_order(images: list[Path], order: str | None) -> list[Path]:
    if not order:
        return images

    by_name = {path.name.lower(): path for path in images}
    by_stem = {path.stem.lower(): path for path in images}
    ordered: list[Path] = []
    seen: set[Path] = set()
    missing: list[str] = []

    for item in [part.strip() for part in order.split(",") if part.strip()]:
        key = item.lower()
        path = by_name.get(key) or by_stem.get(key)
        if not path:
            missing.append(item)
            continue
        if path not in seen:
            ordered.append(path)
            seen.add(path)

    if missing:
        raise SystemExit(f"Ordered source image(s) not found: {', '.join(missing)}")

    ordered.extend(path for path in images if path not in seen)
    return ordered


def choose_cover(images: list[Path], cover_name: str | None) -> Path:
    if not images:
        raise SystemExit("No source images found.")
    if not cover_name:
        return images[0]
    matches = [path for path in images if path.name == cover_name or path.stem == cover_name]
    if not matches:
        raise SystemExit(f"Cover source not found in --src: {cover_name}")
    return matches[0]


def gallery_sources_after_detail_image(images: list[Path], cover: Path) -> list[Path]:
    return [path for path in images if path != cover]


def output_paths(collection: str, base: str, formats: set[str]) -> dict[str, Path]:
    out_dir = ROOT / "static" / "img" / "outside" / collection
    paths: dict[str, Path] = {}
    if "jpg" in formats or "jpeg" in formats:
        paths["jpg"] = out_dir / f"{base}.jpg"
    if "webp" in formats:
        paths["webp"] = out_dir / f"{base}.webp"
    return paths


def validate_watermark_args(args: argparse.Namespace) -> WatermarkOptions | None:
    if not args.watermark:
        return None

    path = args.watermark_path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Watermark does not exist: {path}")
    if not 0 < args.watermark_opacity <= 1:
        raise SystemExit("--watermark-opacity must be greater than 0 and no more than 1.")
    if not 0 < args.watermark_width_ratio <= 0.5:
        raise SystemExit("--watermark-width-ratio must be greater than 0 and no more than 0.5.")
    if not 0 <= args.watermark_margin_ratio <= 0.2:
        raise SystemExit("--watermark-margin-ratio must be between 0 and 0.2.")
    if not 0 <= args.watermark_bottom_margin_ratio <= 0.2:
        raise SystemExit("--watermark-bottom-margin-ratio must be between 0 and 0.2.")
    if not 0 <= args.watermark_shadow_opacity <= 1:
        raise SystemExit("--watermark-shadow-opacity must be between 0 and 1.")
    if not 0 <= args.watermark_shadow_blur_ratio <= 0.2:
        raise SystemExit("--watermark-shadow-blur-ratio must be between 0 and 0.2.")
    if not 0 <= args.watermark_shadow_offset_ratio <= 0.2:
        raise SystemExit("--watermark-shadow-offset-ratio must be between 0 and 0.2.")

    return WatermarkOptions(
        path=path,
        opacity=args.watermark_opacity,
        width_ratio=args.watermark_width_ratio,
        margin_ratio=args.watermark_margin_ratio,
        bottom_margin_ratio=args.watermark_bottom_margin_ratio,
        shadow_opacity=args.watermark_shadow_opacity,
        shadow_blur_ratio=args.watermark_shadow_blur_ratio,
        shadow_offset_ratio=args.watermark_shadow_offset_ratio,
    )


def apply_watermark(image, options: WatermarkOptions):
    if options.opacity <= 0:
        return image

    Image, ImageEnhance, ImageFilter, _ImageOps = require_pillow()
    cairosvg = require_cairosvg()

    target_width = max(1, round(image.width * options.width_ratio))
    png = cairosvg.svg2png(url=str(options.path), output_width=target_width)

    with Image.open(io.BytesIO(png)) as rendered:
        watermark = rendered.convert("RGBA")

    margin = round(image.width * options.margin_ratio)
    bottom_margin = round(image.width * options.bottom_margin_ratio)
    position = (
        max(0, image.width - watermark.width - margin),
        max(0, image.height - watermark.height - bottom_margin),
    )
    base = image.convert("RGBA")

    if options.shadow_opacity > 0:
        shadow_alpha = ImageEnhance.Brightness(watermark.getchannel("A")).enhance(options.shadow_opacity)
        shadow = Image.new("RGBA", watermark.size, (0, 0, 0, 0))
        shadow.putalpha(shadow_alpha)
        blur = max(0, round(watermark.width * options.shadow_blur_ratio))
        if blur:
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
        offset = round(watermark.width * options.shadow_offset_ratio)
        shadow_position = (
            min(max(0, image.width - shadow.width), position[0] + offset),
            min(max(0, image.height - shadow.height), position[1] + offset),
        )
        base.alpha_composite(shadow, shadow_position)

    alpha = watermark.getchannel("A")
    watermark.putalpha(ImageEnhance.Brightness(alpha).enhance(options.opacity))
    base.alpha_composite(watermark, position)
    return base.convert("RGB")


def resize_and_save(
    src: Path,
    outputs: dict[str, Path],
    max_edge: int,
    quality: int,
    watermark: WatermarkOptions | None,
    force: bool,
    dry_run: bool,
) -> tuple[int, int] | None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not force:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing file(s): {names}. Use --force.")

    for path in outputs.values():
        print(f"write {path.relative_to(ROOT)}")

    if dry_run:
        return None

    Image, _ImageEnhance, _ImageFilter, ImageOps = require_pillow()
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        if watermark:
            image = apply_watermark(image, watermark)
        size = image.size

        for fmt, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "jpg":
                image.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
            elif fmt == "webp":
                image.save(path, "WEBP", quality=quality, method=6)

        return size


def public_path(path: Path) -> str:
    rel = path.relative_to(ROOT / "static")
    return "/" + rel.as_posix()


def manifest_block(image_paths: dict[str, Path], label: str, layout: str, size: tuple[int, int] | None) -> str:
    lines = ["[[gallery]]"]
    if "jpg" in image_paths:
        lines.append(f'image = "{public_path(image_paths["jpg"])}"')
    if "webp" in image_paths:
        lines.append(f'image_webp = "{public_path(image_paths["webp"])}"')
    if size:
        lines.append(f"width = {size[0]}")
        lines.append(f"height = {size[1]}")
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
    watermark = validate_watermark_args(args)
    formats = {fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()}
    unsupported = formats - {"jpg", "jpeg", "webp"}
    if unsupported:
        raise SystemExit(f"Unsupported format(s): {', '.join(sorted(unsupported))}")
    if not formats:
        formats = set(DEFAULT_FORMATS)

    images = apply_custom_order(find_sources(src_dir), args.order)
    cover = choose_cover(images, args.cover)

    cover_outputs = output_paths(collection, "cover", formats)
    resize_and_save(cover, cover_outputs, args.cover_max, args.quality, None, args.force, args.dry_run)

    detail_outputs = None
    gallery_index_start = args.start_index
    if not args.exclude_cover_from_gallery:
        if args.name_mode == "sequence":
            detail_base = f"frame-{args.start_index:02d}"
        else:
            detail_base = image_base_from_stem(cover.stem)
        detail_outputs = output_paths(collection, detail_base, formats)
        resize_and_save(cover, detail_outputs, args.gallery_max, args.quality, watermark, args.force, args.dry_run)
        gallery_index_start = args.start_index + 1

    gallery_sources = gallery_sources_after_detail_image(images, cover)
    blocks: list[str] = []
    for offset, image in enumerate(gallery_sources):
        if args.name_mode == "sequence":
            index = gallery_index_start + offset
            base = f"frame-{index:02d}"
            label = f"Frame {index:02d}"
        else:
            base = image_base_from_stem(image.stem)
            label = image.stem
        paths = output_paths(collection, base, formats)
        size = resize_and_save(image, paths, args.gallery_max, args.quality, watermark, args.force, args.dry_run)
        blocks.append(manifest_block(paths, label, args.gallery_layout, size))

    if not args.no_manifest:
        write_manifest(collection, blocks, args.mode, args.dry_run)

    print("\nFront matter fields:")
    if "jpg" in cover_outputs:
        print(f'cover = "{public_path(cover_outputs["jpg"])}"')
    if "webp" in cover_outputs:
        print(f'cover_webp = "{public_path(cover_outputs["webp"])}"')
    if detail_outputs:
        if "jpg" in detail_outputs:
            print(f'detail_image = "{public_path(detail_outputs["jpg"])}"')
        if "webp" in detail_outputs:
            print(f'detail_image_webp = "{public_path(detail_outputs["webp"])}"')
    if not args.no_manifest:
        print(f'gallery_manifest = "data/outside/{collection}.toml"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
