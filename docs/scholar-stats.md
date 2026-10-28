# Scholar citation workflow

`data/scholar.json` is the single citation cache. It contains the Scholar
`author_id`, the author's `total` citation count, and a `papers` object mapping
stable `gs_id` values from `data/papers.toml` to integer citation counts. Paper
titles and other publication metadata remain in `data/papers.toml`.

Cards display citation badges only for counts of at least 10. Lower counts are
still refreshed, so a paper appears automatically after crossing the threshold.
Missing entries do not display a badge. The author total comes from Scholar;
it is not computed by summing the listed papers.

## Refreshing

Run `python scripts/update_scholar_stats.py` with `SERPAPI_KEY` set. Each successful
refresh uses one SerpAPI author-list request (`num=100`) for both the total and
all paper counts, without per-paper detail requests. A failed refresh is retried
once. Without the key, the script uses scholarly's author listing (install
`scholarly` in the runner environment).

Titles are matched ignoring case, punctuation and spacing. Add an optional
`gs_title` alias in `data/papers.toml` when Scholar uses a different title; both
the display title and alias are tried. Give each new paper a unique `gs_id`.
Unmatched papers keep their previous count; papers removed from the tracked list
are removed from the cache on a successful refresh. The script replaces the JSON
atomically, preserving the previous file if fetching or writing fails.

The current 16 tracked papers fit in one response. Revisit pagination if the Scholar
profile exceeds 100 entries; do not introduce per-paper requests.

## Automation

Scheduled and manual deployments refresh the cache before building, using the
committed cache if the refresh fails. The monthly `scholar-stats-pr.yml`
workflow opens a PR updating `data/scholar.json` to keep that fallback current.
Push deployments and page views do not call SerpAPI.
