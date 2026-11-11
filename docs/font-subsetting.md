# Web-font subsetting and loading

`scripts/build_font_subsets.py` generates Noto Sans/Serif SC, Libre Baskerville,
Source Sans 3, and Source Serif 4 subsets from site-owned text. The original
variable fonts stay in their existing source directories, with their license
metadata intact. Zola excludes those build inputs from deployment; only the
generated subsets and the small Site Symbols font are served.

## Automatic coverage and cache

The generator scans `content/`, `data/`, `templates/`, `sass/`, runtime JavaScript,
`config.toml`, and the Chinese search index. It normalizes HTML entities and
Unicode. CJK faces include the Chinese characters in these sources; Latin faces
include supported non-CJK characters plus printable ASCII for search input,
dates, and dynamic numbers. Characters outside the generated coverage use the
visitor's system fonts.

Each run collects one character set and partitions it for each font family;
generation, cache keys, and validation all use that same snapshot. The generated
CSS and preload catalog share the font definitions in `FONT_TARGETS`. File writes
use atomic replacement, and generated URLs always use web path separators.

Generated fonts, CSS, and a preload catalog live in `static/fonts/generated/`.
This directory and `.font-subsets-manifest.json` are ignored by Git. The manifest
checks source-font hashes, both character sets, the generator, FontTools version,
and all output hashes before reusing a build. The CSS declares each face's actual
Unicode range and gives its URL a content hash. Changes invalidate that font's
browser cache; unchanged fonts retain their URL across pages and builds.

The three upright Latin faces are preloaded. Chinese pages also preload the two
CJK faces. Italics and CJK characters on English pages load only when CSS needs
them. Font failures fall back to system text, so reading does not depend on a
JavaScript reveal step. CSS and page scripts also use Zola's content-based
`cachebust` URLs. HTTP caching owns asset reuse; localStorage only stores reader
preferences, and no video or font blobs are persisted there.

## Entry points

- `serve.sh` / `serve.ps1` generate fonts before startup and watch source text.
- `build.sh` / `build.ps1` generate fonts, build Zola, and check rendered coverage.
- GitHub Actions caches the generated directory and incremental manifest.
- Tests check coverage, hashed URLs, incremental reuse, and deployment wiring.

The script keeps its existing name and CLI for compatibility. After changing the
generator itself, restart the development wrapper at the next normal startup so
its watcher uses the new code; ordinary content edits are handled automatically.

```bash
uv run --locked python scripts/build_font_subsets.py
uv run --locked python scripts/build_font_subsets.py --check
./build.sh
```
