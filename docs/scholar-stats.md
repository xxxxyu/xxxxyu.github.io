# Scholar citation workflow

Each paper in `data/papers.toml` uses the complete Google Scholar `citation_id`
as its unique reference to citation data, for example
`IjoWeIMAAAAJ:u-x6o8ySG0sC` for FlexNN. Copy the value of `citation_for_view` from
that paper's link on the author's Scholar profile.

## Fetching and display

Each refresh checks the free SerpApi Account API, then issues one
`google_scholar_author` search with `num=100`. `data/scholar.json` stores every
returned article under its `citation_id` in the `articles` object. Each entry
contains the available title, authors, publication, year, links and citation
count. The snapshot also stores the author information and `cited_by`
statistics. The author's `total` comes from Scholar's citation table.

Publication cards use their configured `citation_id` to look up the article and
read `cited_by.value`. Cards show citation badges at 10 or more citations. Counts
below 10 are also refreshed. An absent article or a paper awaiting its Scholar ID
renders its authored publication information. Add the ID after Scholar includes
the paper in the author listing. Update the configured ID if its record is
replaced in Scholar.

The site's titles, authors, links, order and `featured` selection are authored in
`data/papers.toml`. The research map uses readable `work.id` values for its nodes,
edges and deep links, and associates papers through `paper_url`.

## Running a refresh

GitHub Actions reads the repository Actions secret named `SERPAPI_KEY`.
For a local refresh, set that environment variable and run
`uv run --locked python scripts/update_scholar_stats.py`.
Each run makes at most one Account API request and one author-list search. The
script checks the remaining quota before searching and uses SerpApi's default
one-hour cache policy.

A failed request ends that refresh. Missing credentials, unavailable account
data, exhausted quota, invalid results and file-write failures preserve the
existing cache. The maximum author-list page size is 100. A response with a next
page is rejected as incomplete, preserving the one-search-per-refresh budget.
Revisit that budget if the Scholar profile grows beyond 100 articles.

The Account API response contains credentials and account identity. Logs and
Actions Summary receive only the plan name, allowance, used searches, remaining
searches and renewal date. The remaining count is observed before the search.
The summary also records the fetch time and article count.

The updater is one standard-library Python module. `request_serpapi` handles
HTTP responses, `parse_snapshot` validates and indexes the listing, and
`write_stats` handles atomic cache replacement. `main` runs the balance check,
fetch and write in order. Tests can replay a captured public author-list
response through `parse_snapshot` independently of network access.

## Automation and Git history

`.github/workflows/deploy.yml` is the single automation entry point. Its daily
schedule is 08:17 Asia/Shanghai, expressed as `17 0 * * *` in UTC. GitHub may delay
scheduled runs. Manual workflow runs use the same refresh process. Push-triggered
deployments build from the committed cache.

A successful refresh writes the snapshot atomically. JSON keys have stable
ordering. Fetch timestamps, account usage and request metadata are kept in
Actions Summary. Content changes produce one ordinary
`chore: refresh Google Scholar data` commit on `main`. Identical snapshots leave
the cache file untouched and produce an empty Git diff.

Manual runs on a feature branch validate that branch's tests, build and Scholar
refresh. Cache commits and Pages deployment run on `main`.

The workflow checks out the latest requested branch, serializes runs per branch and rebases its
data commit onto concurrent changes before a normal push. A merge conflict or
rejected push fails the job. The same workflow then builds and deploys the
committed snapshot. The `GITHUB_TOKEN` data push is followed by the remaining
steps of that run, so deployment continues within the current workflow.

The initial ID migration and cache snapshot were verified against the
[public Scholar profile](https://scholar.google.com/citations?user=IjoWeIMAAAAJ)
on 2026-11-16. Daily runs normally consume 28–31 searches per month. Official
references cover the [author-list API](https://serpapi.com/google-scholar-author-api),
[Account API](https://serpapi.com/account-api),
[search accounting](https://serpapi.com/pricing) and
[GitHub token behavior](https://docs.github.com/en/actions/concepts/security/github_token).
