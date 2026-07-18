# Asset and gallery workflow

This document covers repository asset policy and the image-processing workflow
for the site's Outside collections.

## Asset policy

Commit only web-ready assets that the site serves. Do not commit camera
originals, export folders, or other large source files. `assets-src/` is ignored
by Git and is the preferred location for local originals.

Recommended layout:

```text
static/files/
  cv/
  blog/<post-slug>/
  papers/
  slides/

static/img/
  identity/
  papers/
  blog/<post-slug>/
  outside/<collection-slug>/

assets-src/
  outside/<collection-slug>/   # local originals; never committed
```

Files under `static/` keep the same URL path in the built site. For example,
`static/img/papers/example.webp` is served as `/img/papers/example.webp`.

## Create an Outside photography collection

Create a local source directory and install the image tooling once per machine:

```bash
mkdir -p assets-src/outside/tokyo-2026
uv sync
```

Place the source images in that directory, then process them from the repository
root:

```bash
uv run scripts/optimize_outside_images.py \
  --collection tokyo-2026 \
  --src assets-src/outside/tokyo-2026 \
  --cover IMG_0001.JPG
```

The script writes:

- optimized images to `static/img/outside/<collection>/`;
- a gallery manifest to `data/outside/<collection>.toml`.

Commit those generated files, not the originals. A different machine can
process a separate collection without having access to earlier originals. To
add images to an existing collection, use `--mode append` so the current
manifest is preserved.

Run with `--dry-run` to inspect planned outputs, or `--force` when existing
generated images should be overwritten. The script's complete and current
option list is available through:

```bash
uv run scripts/optimize_outside_images.py --help
```

## Image output and optional watermark

By default, cover and gallery images:

- retain their original aspect ratio and are not cropped;
- have a maximum edge of 2160 px;
- are written as progressive JPEG and WebP at quality 88.

Page layout, rather than the generated file, controls visual framing.

Visible watermarks are not used by default: they offer limited protection for
web-sized images and can distract from the composition. Copyright and reuse
terms are communicated by the page metadata and the repository licensing file.

When a particular collection does require visible branding, `--watermark`
places the current site logo in the lower-right corner. Its defaults are
3.2% of image width, 1% right margin, 0.4% bottom margin, and 38% opacity. These
can be adjusted with `--watermark-width-ratio`, `--watermark-margin-ratio`,
`--watermark-bottom-margin-ratio`, and `--watermark-opacity`; shadow controls
are also exposed in `--help`.

The cover source creates `cover.*` files for cards. Unless
`--exclude-cover-from-gallery` is used, that source also becomes a detail image
named from its source stem, such as `img-0001.*`. The manifest starts with the
remaining images, avoiding an immediate repeat below the page introduction.

## Cover, naming, and ordering rules

- Sources are discovered under `--src` and sorted case-insensitively by
  filename.
- `--cover IMG_0001.JPG` selects a source by exact filename or stem. Without it,
  the first sorted source is the cover.
- `--order IMG_0003.JPG,IMG_0001.JPG,IMG_0002.JPG` sets an explicit order by
  filename or stem; unlisted sources follow in filename order.
- Output names use normalized source stems by default. Use
  `--name-mode sequence` only when `frame-01.*`-style names are desired.
- `--exclude-cover-from-gallery` prevents generation of the cover's detail
  image.

## Gallery layout

Each generated manifest item can be edited to control its layout and caption:

```toml
[[gallery]]
image = "/img/outside/tokyo-2026/img-0002.jpg"
image_webp = "/img/outside/tokyo-2026/img-0002.webp"
layout = "center" # pair | center | full
label = "IMG_0002"
caption = "..."
```

Use `pair` for the normal two-column rhythm, `center` for a pair-sized image on
its own centered row, and `full` for a highlight spanning the gallery width.
`--gallery-layout` changes the default written for a processing run.

Finally, reference the manifest from the collection page's front matter:

```toml
[extra]
gallery_manifest = "data/outside/tokyo-2026.toml"
```

Review the collection in the local development server before committing the
optimized files and manifest.
