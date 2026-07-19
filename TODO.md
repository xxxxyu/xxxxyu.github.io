# TODO

Personal roadmap for the site. This file is repository-only and is not
published by Zola.

## Site features

- [x] Add first-class post series, using [tabi's series
  design](https://welpo.github.io/tabi/blog/series/) as a reference.
  - [x] Define a small front-matter/section convention for series membership
    and explicit ordering.
  - [x] Add a series overview page with its description and ordered post list.
  - [x] Show the series name on each member post and post listing.
  - [x] Keep member posts in the main chronological Blog navigation flow.
  - [ ] Add optional series intro/outro content if a series needs internal
    progress or previous/next links.
  - [x] Ensure series pages work with feeds, search, tags, archives, and mobile
    layouts.

- [x] Add multilingual infrastructure MVP.
  - [x] Decide the default language, URL scheme, and translation-pairing
    convention before translating content.
  - [x] Add a compact language switcher that opens the equivalent translation
    of the current page when available and falls back predictably otherwise.
  - [x] Establish the first English/Chinese Cheatsheet pair while keeping dates,
    tags, series membership, and assets in sync.
  - [x] Add appropriate `lang`, `hreflang`, canonical, feed, search-index, and
    sitemap behavior for each language.
  - [x] Verify navigation length, typography, code blocks, ToC, and responsive
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

- [x] Translate all existing posts, preserving the multilingual
  metadata and URL conventions in `docs/multilingual.md`.
- [ ] Publish English and Chinese versions of new posts together when practical.

- [ ] Write about vibe coding: workflow, useful patterns, limitations, and what
  remains worth doing manually.
- [ ] Write an ongoing-research update focused on current questions and
  direction rather than unpublished claims that may age poorly.
- [ ] Write about the personal knowledge base: information flow, tools,
  maintenance cost, and lessons learned.
- [ ] Keep a backlog of shorter technical notes that can later become posts or
  members of a series.
