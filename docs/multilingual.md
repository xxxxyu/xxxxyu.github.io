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

For a translation produced by AI, record the source language on the translated
page:

```toml
[extra]
ai_translation_source = "en"
```

Use `"zh"` when an English page was translated from a Chinese original. The
post template then adds a localized disclosure at the beginning of the article
and links the original title to its source page. Human translations omit this
field.

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
`/zh/atom.xml`, and generates `/search_index.en.json` for English. The standard
Zola binary is not built with its optional Chinese tokenizer, so the repository
generates the Chinese document store at `/search_index.zh.json`; the search UI
performs substring matching for it. Sitemap entries include both language URLs.
Page and section templates emit a self canonical plus reciprocal `hreflang`
links for available translations, with English as `x-default`.

After adding or changing a translation, run:

```bash
uv run python scripts/build_zh_search_index.py
uv run python -m unittest discover -s tests -v
zola build
```
