# xxxxyu.github.io

Source for [Xiangyu Li's personal website](https://xxxxyu.github.io), an academic
homepage and long-form blog built with [Zola](https://www.getzola.org/).

The site brings together:

- a research profile, experience, news, and selected skills;
- publications with Google Scholar citation badges;
- technical writing on on-device AI, efficient inference, and embodied AI,
  including ordered post series;
- an "Outside" section for photography, personal writing, and music;
- full-site search, light/dark/system themes, Atom feeds, and responsive layouts.

The repository intentionally keeps the stack small: Markdown and TOML hold the
content, Tera templates render it, and Sass provides the presentation. Zola
produces a fully static site in `public/`.

## Quick start

Install Zola before starting local development; see the
[Zola installation guide](https://www.getzola.org/documentation/getting-started/installation/).

### Unix (Linux and macOS)

```bash
# From the repository root
chmod +x serve.sh
./serve.sh
```

### Windows (PowerShell)

Make sure `zola.exe` is available on `PATH`, then run:

```powershell
# From the repository root
.\serve.ps1
```

Both scripts enable live reload, watch `data/` in addition to the normal Zola
paths, and forward extra arguments to Zola. For example:

```bash
./serve.sh
```

```powershell
.\serve.ps1
```

Open the URL printed by Zola (normally `http://127.0.0.1:1111`). To create a
production build, run `zola build`; generated output is written to `public/`.

## Repository checks

Run the lightweight content and feature checks through
[uv](https://docs.astral.sh/uv/):

```bash
uv run python -m unittest discover -s tests -v
```

The checks currently validate the post-series structure and are intended to
grow alongside repository features. Run `zola build` as a separate production
rendering check when templates, styles, configuration, or content structure
change.

## Project layout

```text
content/       Markdown pages and section front matter
data/          Structured content and generated manifests
templates/     Zola Tera templates and shortcodes
sass/          Site styles
static/        Files copied directly into the built site
scripts/       Content and asset maintenance utilities
tests/         Extensible repository and content checks
docs/          Maintainer documentation
config.toml    Zola and site-wide configuration
serve.sh       Unix development entry point
serve.ps1      Windows development entry point
```

See the [post series convention](docs/post-series.md) before creating or
reordering a series, and the
[asset and gallery workflow](docs/assets-and-galleries.md) before adding large
files or publishing an Outside photography collection.

## Deployment

Pushes to `main` are built and deployed to GitHub Pages by
`.github/workflows/deploy.yml`. The same workflow refreshes Google Scholar data
on its scheduled and manual runs, falling back to committed values if the
external lookup is unavailable.

For a new fork, select **GitHub Actions** as the Pages source under
**Settings > Pages**.

## License

Site code is available under the [MIT License](licenses/MIT.txt). Original Blog
and Outside content is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless a
page states otherwise. Other personal content is not included in that grant.

See [LICENSE.md](LICENSE.md) for the exact scope and
[third-party notices](THIRD_PARTY_NOTICES.md) for bundled fonts and icons.

## Credits

- [Zola](https://www.getzola.org/) — static site generator
- Design references: [Minimal Light](https://github.com/yaoyao-liu/minimal-light)
  for the academic layout and [Tabi](https://github.com/welpo/tabi) for the blog
  layout
