# Research map

The homepage links to `/research/` and `/zh/research/`. The research page presents
an interactive thematic landscape for the publications and projects in
`data/research.toml`, with the canonical publication records kept in
`data/papers.toml`.

## Editorial data

`data/research.toml` is the source for the map. Each work has a stable `id`, a
short label, role, contribution weight, topics, taxonomy terms, localized
summaries, and destination links. The English taxonomy lives in
`data/research-taxonomy.en.toml`; the concrete research terms remain English in
both page versions so they stay recognizable in the literature.

The broad fields are `systems`, `embodied`, and `agents`. `Interaction Memory`
is kept distinct from Agent Memory. Field membership is editorial and does not
claim a formal clustering result. Shared taxonomy terms create thematic links;
the default view shows a curated subset to keep the landscape readable.

Roles use `author` for first/co-first/sole author, `section` for an explicitly
led survey section, `collaborator` for co-authorship, and `project` for an
open-source project or contribution. The map presents `author` and `section`
with the same primary visual treatment, while the source data keeps the roles
distinct for filtering and future edits.

## Interaction and layout

Tera emits the complete SVG and the accessible publication list. Page-local
vanilla JavaScript adds field, role, type, and text filters; hover/focus detail
cards; direct links; and the animated landscape. The map starts each refresh or
filter from a randomized compact cloud and uses locally bundled `d3-force` to
open it into a fixed 820 × 400 landscape frame. A small float continues after
settling. The page intentionally keeps this motion even when the browser
reports reduced-motion preferences because animation is the selected
presentation for this visualization.

The `physics` object in `static/js/research.js` centralizes force parameters.
Contribution weight controls center attraction and collision mobility, while
shared terms control link strength and length. A soft group center and edge
field prevent the cloud from drifting into corners. Collision resolution uses
label measurements and boundaries, and does not precompute final coordinates.

Very light convex envelopes follow the currently visible members of each broad
field. They are background orientation cues, not strict set boundaries, and do
not participate in the force simulation. The map is wide on desktop while the
intro, filters, and supporting copy follow the site's normal reading width.
On small screens only the figure scrolls horizontally.

## Maintenance

Run the development wrapper (`./serve.sh` or `./serve.ps1`) for live preview.
Run `zola build` for a production check and
`uv run python -m unittest discover -s tests -v` after changing the research
data or templates. `tests/test_research.py` checks publication coverage,
coordinates, supported roles, taxonomy assignments, and destination links.

The D3 vendor files are locally bundled and documented in
`THIRD_PARTY_NOTICES.md`. The homepage uses a lightweight static illustration
and does not load the research page's force simulation.

## References

- [D3 force](https://github.com/d3/d3-force)
- [D3 force-directed graph component](https://observablehq.com/@d3/force-directed-graph-component)
- [Connected Papers](https://www.connectedpapers.com/)
