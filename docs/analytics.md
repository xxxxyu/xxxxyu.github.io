# Website analytics

The dashboard lives in `/analytics/` and `/zh/analytics/`. Its entry point is
`static/js/analytics.js`; localized labels are in `data/analytics.toml` and
styles in `sass/_analytics.scss`. The separate `analytics-track.js` is loaded
on production pages only when `extra.analytics_endpoint` is configured.
The collector and its SQLite Durable Object live in `analytics-worker/`.

## Local development

Run `./serve.sh` or `.\serve.ps1`. Loopback hosts use
`static/js/analytics-demo.mjs` when no endpoint is configured. The 90-day sample
includes the last 30 days and adds different geographic readership. Local
page views and dashboard visits are excluded from collection.

To exercise the collector, use Node.js 22.13+ and run in `analytics-worker/`:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:1111/analytics/?source=local`. Wrangler stores this local
database in the ignored `.wrangler/` directory and needs no Cloudflare login.

## Deploy the collector

When ready to connect real statistics, run in `analytics-worker/`:

```sh
npx wrangler login
npm run deploy
```

Put the printed workers.dev origin in `config.toml` under `[extra]`, then build
and deploy the Zola site normally:

```toml
analytics_endpoint = "https://xiangyu-site-analytics.YOUR-SUBDOMAIN.workers.dev"
```

Set `ALLOWED_ORIGINS` in `analytics-worker/wrangler.toml` to the site's origin.
The dev command adds localhost access only for local testing. Aggregated
statistics are public. Keep the Worker name, Durable Object binding, class,
migration and `site-v1` object name stable after deployment to preserve data.

Workers Free supports SQLite Durable Objects within the account's shared
request, SQL and storage quotas. Each view writes several rows and indexes;
quota exhaustion causes operations to fail on Free. Check the account's
Workers subscription before deployment. A free workers.dev hostname is
included. See [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/)
and [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/).

## Behavior and storage

- Events are deduplicated by event UUID. Visitors are deduplicated across the
  selected range by the SHA-256 hash of a random, first-party browser ID.
  That ID expires after 365 days without renewal. Clearing browser storage or
  changing browsers creates a new visitor.
- The tracker respects DNT, GPC and previously saved exclusion preferences.
  The collector also rejects common bots, invalid events and foreign origins.
- Stored data includes pathname, referring hostname, country, city and rounded
  1° map coordinates. Query strings, fragments, raw IPs and raw visitor IDs
  are excluded from this database. Cloudflare supplies IP-based geography;
  unknown locations stay unknown. Infrastructure logging follows the
  Cloudflare account settings.
- Days use UTC. Detailed events last 400 days; visitor hashes expire after
  366 inactive days. Lifetime counters survive cleanup. Referring domains
  carry across internal navigation in the same tab until 30 minutes idle.
- sessionStorage caches the selected range, number widths, map and aggregate
  responses. Data older than 60 seconds refreshes in the background. Preview
  and collector caches are separate. Storage failures fall back to loading.
- Refreshes replay counts and bars from zero. Range changes preserve and
  animate existing SVG elements. Number widths stay fixed. Failed requests
  preserve displayed data and offer retry. Effects respect reduced motion.

## Verification

CI runs the Python suite, frontend checks and collector tests. Collector tests
use Node's built-in SQLite and require no npm dependencies or Cloudflare login.
Run `npm --prefix analytics-worker test` from the root. After collector changes,
run `npm --prefix analytics-worker run check` for a Wrangler build without
publishing. Site checks and production builds are documented in the README.
