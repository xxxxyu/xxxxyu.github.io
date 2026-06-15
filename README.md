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
python3 -m pip install Pillow

# Process whatever source images exist on this machine.
python3 scripts/optimize_outside_images.py \
  --collection tokyo-2026 \
  --src assets-src/outside/tokyo-2026 \
  --cover IMG_0001.JPG
```

The script writes optimized files to `static/img/outside/<collection>/` and a
gallery manifest to `data/outside/<collection>.toml`. Commit those generated
files, not the originals. Another machine can process another collection, or use
`--mode append` to add more images to an existing manifest without having the
earlier originals locally.

Gallery items support a small amount of layout control:

```toml
[[gallery]]
image = "/img/outside/tokyo-2026/frame-01.jpg"
image_webp = "/img/outside/tokyo-2026/frame-01.webp"
layout = "full" # pair | full
label = "Frame 01"
caption = "..."
```

Use `pair` for the normal two-column rhythm and `full` for highlight images
that should span the full gallery width.

## Credits

- [Zola](https://www.getzola.org/) — static site generator
- Design inspired by [Minimal Light](https://github.com/yaoyao-liu/minimal-light) (academic layout) and [Tabi](https://github.com/welpo/tabi) (blog layout)
