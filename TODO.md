# TODO

Personal roadmap for the site. This file is repository-only and is not
published by Zola.

## Site features

- [ ] Add first-class post series, using [tabi's series
  design](https://welpo.github.io/tabi/blog/series/) as a reference.
  - [ ] Define a small front-matter/section convention for series membership
    and explicit ordering.
  - [ ] Add a series overview page with its description and ordered post list.
  - [ ] Show the series name and `part N of M` on each member post.
  - [ ] Add previous/next navigation within the series without removing posts
    from the main chronological Blog flow.
  - [ ] Ensure series pages work with feeds, search, tags, archives, and mobile
    layouts.

- [ ] Add multilingual site and post support.
  - [ ] Decide the default language, URL scheme, and translation-pairing
    convention before translating content.
  - [ ] Add a compact language switcher that opens the equivalent translation
    of the current page when available and falls back predictably otherwise.
  - [ ] Add English and Chinese versions of existing posts while keeping dates,
    tags, series membership, and assets in sync.
  - [ ] Add appropriate `lang`, `hreflang`, canonical, feed, search-index, and
    sitemap behavior for each language.
  - [ ] Verify navigation length, typography, code blocks, ToC, and responsive
    behavior in both languages.

- [ ] Enrich the Outside photography collection layout.
  - [ ] Support more deliberate rhythms than the current `pair`, `center`, and
    `full` primitives, including asymmetric sequences where useful.
  - [ ] Improve captions and collection-level context without competing with
    the photographs.
  - [ ] Review portrait/landscape transitions, spacing, and mobile presentation
    on real collections.
  - [ ] Consider an accessible lightbox only if it materially improves viewing
    at full size.

## Content

- [ ] Write about vibe coding: workflow, useful patterns, limitations, and what
  remains worth doing manually.
- [ ] Write an ongoing-research update focused on current questions and
  direction rather than unpublished claims that may age poorly.
- [ ] Write about the personal knowledge base: information flow, tools,
  maintenance cost, and lessons learned.
- [ ] Keep a backlog of shorter technical notes that can later become posts or
  members of a series.
