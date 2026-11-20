// Copyright (c) Xiangyu Li. MIT licensed; see ../../licenses/MIT.txt.
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

export async function checkSummary(response, days, { origin, revision } = {}) {
  assert.equal(response.status, 200, 'Summary request must succeed');
  assert.match(response.headers.get('content-type') || '', /^application\/json\b/);
  if (origin) assert.equal(response.headers.get('access-control-allow-origin'), origin, 'Website CORS origin');
  if (revision) assert.equal(response.headers.get('x-analytics-revision'), revision, 'Deployed revision');
  const data = await response.json();
  const isCount = value => Number.isSafeInteger(value) && value >= 0;
  assert.ok(Number.isFinite(Date.parse(data.generatedAt)), 'Summary timestamp');
  assert.equal(data.period?.days, days, 'Requested range');
  assert.ok([
    data.period.views, data.period.visitors, data.period.countries,
    data.today?.views, data.today?.visitors, data.lifetime?.views, data.lifetime?.visitors,
  ].every(isCount), 'Summary counts');
  assert.ok(Array.isArray(data.days) && data.days.length === days, 'Daily series length');
  for (const key of ['countries', 'cities', 'pages', 'referrers', 'points']) {
    assert.ok(Array.isArray(data[key]), `Missing ${key} breakdown`);
  }
  return data;
}

async function main() {
  const [endpoint, revision] = process.argv.slice(2);
  assert.ok(endpoint, 'Usage: npm run smoke -- <collector-origin> [expected-commit]');
  const origin = process.env.ANALYTICS_SITE_ORIGIN || 'https://xxxxyu.github.io';
  const sites = ['', 'oxygen', 'steerdistill'];
  // Allow the previous deployment's 60-second responses to expire at the edge.
  const attempts = 15;
  for (const days of [30, 90]) {
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        await Promise.all(sites.map(async site => {
          const url = new URL('/api/summary', endpoint);
          url.searchParams.set('days', days);
          if (site) url.searchParams.set('site', site);
          const response = await fetch(url, { headers: { Origin: origin }, signal: AbortSignal.timeout(10000) });
          await checkSummary(response, days, { origin, revision });
        }));
        console.log(`${days}-day summary: OK${revision ? ` (${revision})` : ''}`);
        break;
      } catch (error) {
        if (attempt === attempts) throw error;
        console.warn(`${days}-day check failed; retrying (${attempt}/${attempts}).`);
        await delay(5000);
      }
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => {
    console.error(`Analytics verification failed: ${error.message}`);
    process.exitCode = 1;
  });
}
