# CJK font subsetting

The site serves generated Noto Sans SC and Noto Serif SC subsets so that pages
do not need to download the complete 7–10 MB CJK variable fonts.

## Automatic coverage

`scripts/build_font_subsets.py` scans all site-owned text sources that can
contribute rendered characters: `content/`, `data/`, `templates/`, `sass/`, the
runtime JavaScript, `config.toml`, and the generated Chinese search index. HTML
entities and canonically equivalent Unicode text are normalized before the CJK
character set is collected.

The generated files are:

- `static/fonts/noto-sans-sc/NotoSansSC-SiteSubset.woff2`
- `static/fonts/noto-serif-sc/NotoSerifSC-SiteSubset.woff2`
- `.font-subsets-manifest.json` (the local incremental-build manifest)

These files are build artifacts and are intentionally excluded from Git. A new
clone generates them on the first development or production build, which may
take about a minute. Later runs reuse the manifest and skip font conversion
whenever the source character set, source fonts, generator, and FontTools
version are unchanged.

The normal workflow does not require a character list or a manual font step:

- `serve.sh` and `serve.ps1` generate the subsets before Zola starts and watch
  for newly introduced characters while the development server runs.
- `build.sh` and `build.ps1` generate and validate the subsets around every
  production build;
- the Pages workflow restores a cache of all three generated files and falls
  back to the same cold generation path on a cache miss;
- after Zola renders the site, CI checks the generated HTML, JSON, and XML to
  ensure every rendered CJK character available in the Noto source fonts is
  covered;
- repository tests verify source coverage and the loading/fallback wiring.

The generator records hashes of the character set, source fonts, generator
script, FontTools version, generation format, and outputs in the ignored
`.font-subsets-manifest.json`. When those inputs are unchanged, subsequent
development-server starts and production builds skip the expensive font
conversion and proceed immediately. A real rebuild prints a progress message
before processing the large source fonts.

The full Noto fonts remain behind the subsets in the CSS font stack as a
defensive fallback. They are not preloaded or requested during a correctly
generated build, but they prevent a missed or dynamically introduced character
from becoming a missing glyph. Characters absent from the Noto source files
(including their excluded full-width punctuation) continue through the stack
to an installed system font.

## Manual checks

These commands are useful when debugging the pipeline itself:

```bash
uv run --locked python scripts/build_font_subsets.py
uv run --locked python scripts/build_font_subsets.py --check
./build.sh
```
