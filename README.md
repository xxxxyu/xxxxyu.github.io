# xxxxyu.github.io

Personal academic homepage and blog for Xiangyu Li, built with [Zola](https://www.getzola.org/).

## Development

```bash
# Install Zola (macOS)
brew install zola

# Local dev server with live reload
zola serve

# Production build
zola build
```

The site builds to `public/`.

## Deployment

Pushes to `main` automatically deploy to GitHub Pages via the workflow in `.github/workflows/deploy.yml`.

To enable: go to repo Settings > Pages > Source > GitHub Actions.

## Structure

```
content/          # Markdown content (homepage, about, blog posts, outside pages)
data/             # Structured data and generated manifests
data/gs/          # Google Scholar badge JSON, loaded at build time
data/outside/     # Gallery manifests for Outside collections
templates/        # Zola Tera templates
sass/             # SCSS stylesheets
static/           # Public static assets copied as-is
static/files/     # Public PDFs and downloadable files
static/icons/     # SVG icons
static/img/       # Web-optimized images used by the site
assets-src/       # Local source assets, ignored by git
```

## Assets

Commit only web-ready assets that the site actually serves. Do not commit camera
originals, export folders, or other large source files.

Recommended layout:

```
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
  outside/<collection-slug>/   # local originals, ignored by git
```

Outside photography workflow:

```bash
# Optional: create a local source folder. This folder is ignored by git.
mkdir -p assets-src/outside/tokyo-2026

# Install the image backend once on each machine.
uv sync

# Process whatever source images exist on this machine.
uv run scripts/optimize_outside_images.py \
  --collection tokyo-2026 \
  --src assets-src/outside/tokyo-2026 \
  --cover IMG_0001.JPG \
  --watermark
```

The script writes optimized files to `static/img/outside/<collection>/` and a
gallery manifest to `data/outside/<collection>.toml`. Commit those generated
files, not the originals. Another machine can process another collection, or use
`--mode append` to add more images to an existing manifest without having the
earlier originals locally.

By default, cover and gallery images are resized proportionally to a maximum
edge of 2160px and encoded as progressive JPEG plus WebP at quality 88. The
script does not crop image files; page layout decides how images are framed.
Use `--watermark` to place the site logo in the lower-right corner, with
optional `--watermark-opacity`, `--watermark-width-ratio`, and
`--watermark-margin-ratio` adjustments. The default watermark is small and
close to the corner: 3.2% of the image width, 1% right margin, 0.4% bottom
margin, and 38% opacity. The script writes unwatermarked `cover.*` files from
the cover source for cards, then writes the same source as a watermarked
source-stem image, such as `img-0001.*`, for the detail image. The gallery
manifest starts after that detail image, so the first photo is not repeated
below the page intro.

Cover and ordering rules:

- Sources are discovered from `--src` and sorted by filename, case-insensitively.
- `--cover IMG_0001.JPG` selects the cover source by exact filename or stem.
- If `--cover` is omitted, the first sorted source is used.
- The cover source is promoted to the watermarked detail image unless
  `--exclude-cover-from-gallery` is passed.
- `--order IMG_0003.JPG,IMG_0001.JPG,IMG_0002.JPG` customizes gallery order by
  filename or stem; unlisted images are appended in filename order.
- Default output names come from source filename stems. Pass
  `--name-mode sequence` only when you explicitly want `frame-01.*` naming.

Gallery items support a small amount of layout control:

```toml
[[gallery]]
image = "/img/outside/tokyo-2026/img-0002.jpg"
image_webp = "/img/outside/tokyo-2026/img-0002.webp"
layout = "center" # pair | center | full
label = "IMG_0002"
caption = "..."
```

Use `pair` for the normal two-column rhythm, `center` for a pair-sized image
centered on its own row, and `full` for highlight images that should span the
full gallery width.

## Credits

- [Zola](https://www.getzola.org/) — static site generator
- Design inspired by [Minimal Light](https://github.com/yaoyao-liu/minimal-light) (academic layout) and [Tabi](https://github.com/welpo/tabi) (blog layout)
