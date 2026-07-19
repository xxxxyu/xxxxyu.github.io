# Post series

Post series use a transparent Zola subsection so their members remain in the
main chronological Blog flow.

## Series section

Create the series below `content/blog/<series-slug>/` and add `_index.md` with:

```toml
+++
title = "Series title"
description = "Short description."
template = "series.html"
page_template = "blog-page.html"
sort_by = "weight"
transparent = true

[extra]
series = true
+++
```

The section title, description, URL, and ordered pages are the source of truth
for the overview and series metadata. Member posts use the same chronological
previous/next navigation as the rest of the Blog.

## Member posts

Place each member below the series directory. Give it a stable output path, a
weight with room for insertions, and a pointer to the series section:

```toml
path = "blog/existing-post-slug"
weight = 10

[extra]
series = "blog/<series-slug>/_index.md"
```

Use weights such as `10`, `20`, and `30`. The explicit `path` is required when
an existing post is moved into a series and its public URL must remain stable.
The `extra.series` pointer is also explicit because Zola omits transparent
sections from a rendered page's ancestor list.

Do not duplicate the series title in member front matter; templates derive it
from the referenced section. If a series later needs internal navigation or
progress text, add it through a series intro/outro rather than the site-wide
Blog navigation component.

After adding or reordering members, run:

```bash
uv run python -m unittest discover -s tests -v
zola build
```

The repository tests validate the section convention, membership pointers,
weights, and stable output paths. Also check the series overview, Blog
navigation, Blog, Archive, Atom feed, search index, sitemap, and a narrow
viewport when the related templates or styles change.
