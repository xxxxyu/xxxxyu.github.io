# Multilingual content

The site uses Zola's native multilingual content model. English (`en`) is the
default language and keeps the existing unprefixed URLs. Chinese (`zh`) uses
the `/zh/` prefix.

## Translation pairs

Keep translations beside their source and add the language code before the
Markdown extension:

```text
content/blog/example/index.md
content/blog/example/index.zh.md
```

Section translations follow the same rule (`_index.md` and `_index.zh.md`).
Zola does not fall back to a default-language section, so every translated
page must also have translated `_index.zh.md` files for the sections that own
it.

Pages in a transparent series retain the default section pointer in every
language:

```toml
[extra]
series = "blog/device-setup/_index.md"
```

Templates resolve that pointer with the page language. For pages with an
explicit stable English `path`, set the Chinese path explicitly to
`zh/<english-path>`; Zola does not add the language prefix when `path` is set.

## Synchronized metadata

Translation pairs must keep `date`, `updated`, `weight`, `taxonomies.tags`, and
`extra.series` identical. Titles and descriptions are translated. Assets stay
shared unless a language-specific figure is necessary.

For a translation produced by AI, record the source language and the generation
setup on the translated page:

```toml
[extra]
ai_translation_source = "en"
ai_translation_harness = "OpenCode"
ai_translation_model = "GLM-5.2"
ai_translation_effort = "max"
```

Use `"zh"` when an English page was translated from a Chinese original. The
`ai_translation_harness`, `ai_translation_model`, and
`ai_translation_effort` fields preserve the full generation setup for future
writing-quality comparisons. Keep `ai_translation_effort` as an empty string
when the original effort cannot be confirmed. The post template currently
discloses only the model and links the original title to its source page. Human
translations omit all four fields.

All Blog posts currently have both English and Chinese versions. New posts
should normally add the paired file at publication time; if one version must
ship first, the default-content fallback remains available until its pair is
ready.

The language preference is site-wide and changes only when the visitor clicks
the language switcher. The switcher opens the equivalent translation when Zola
exposes one. If the target translation is missing, the current URL and default
content remain in place while the site chrome and search adopt the selected
language. Following later links prefers that language where a translation
exists and otherwise continues to use the default content without clearing the
preference. Links in the Chinese navigation whose sections are not translated
yet point to the English section and are marked `lang="en"`.

## Generated language surfaces

Both languages generate an Atom feed. Zola writes the feeds as `/atom.xml` and
`/zh/atom.xml`, and generates a compact Fuse-format document list at
`/search_index.en.json` for English. The frontend uses its own substring scoring,
so it does not download an unused Elasticlunr inverted index. The standard
Zola binary is not built with its optional Chinese tokenizer, so the repository
generates the Chinese document store at `/search_index.zh.json`; the search UI
performs substring matching for it. Sitemap entries include both language URLs.
Page and section templates emit a self canonical plus reciprocal `hreflang`
links for available translations, with English as `x-default`.

The Chinese index is an ignored build artifact at `static/search_index.zh.json`.
`scripts.prepare_assets` generates it before collecting font characters, and is
shared by the Unix/Windows build and development wrappers and GitHub Actions.
CI prepares the index before computing the font cache key; it no longer requires
a committed copy of the index.

The search generator owns content selection and JSON output; the font generator
owns font subsetting and coverage checks. `scripts.prepare_assets` owns ordering
and watching. Keep those responsibilities separate when extending the workflow.

The generator reads Markdown titles, descriptions, and bodies, preferring a
published Chinese translation and otherwise using the English source. Drafts
and pages excluded from search are not published in the index. Content added by
templates, such as data-driven news and paper lists, is outside this source-based
index. Each content file is parsed once per index build. Results are written by
atomic replacement only when the JSON changes.

Development uses one asset watcher. Two matching source snapshots, 0.75 seconds
apart, coalesce rapid saves; invalid partial edits retain the previous index and
are retried. Generated outputs do not trigger another preparation cycle. Restart
the development wrapper at the next normal startup after changing the generator
or watcher itself; content edits need no restart.

After adding or changing a translation, run:

```bash
uv run --locked python -m unittest discover -s tests -v
./build.sh
```

On Windows, use `.\build.ps1` for the final production-build check.

Tests validate generated document coverage and the generator's translation
fallback, exclusions, update behavior, and failure recovery without depending on
a checked-in index. For a standalone refresh, run
`uv run --locked python scripts/build_zh_search_index.py`; add `--check` to verify
an existing output without changing it.
