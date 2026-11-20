import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import test from 'node:test';
import { createTestHarness } from 'wrangler';
import { checkSummary } from '../../scripts/smoke.js';

// Runtime tests use local geography placeholders, without fetching IP metadata.
process.env.CLOUDFLARE_CF_FETCH_ENABLED = 'false';

test('the configured Workers runtime stores events, deduplicates delivery, and preserves data after eviction', { timeout: 60000 }, async t => {
  const server = createTestHarness({
    workers: [{ configPath: fileURLToPath(new URL('../../wrangler.toml', import.meta.url)) }],
  });
  t.after(() => server.close());
  const { url } = await server.listen();
  const worker = server.getWorker();
  const env = await worker.getEnv();
  const origin = env.ALLOWED_ORIGINS.split(',')[0];
  const redirect = await server.fetch('/', { redirect: 'manual' });
  assert.equal(redirect.status, 302);
  assert.equal(redirect.headers.get('location'), `${origin}/analytics/`);
  for (const [path, method] of [['/api/track', 'POST'], ['/api/analytics', 'GET']]) {
    assert.equal((await server.fetch(path, { method })).status, 404);
  }
  const headers = { Origin: origin, 'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json' };
  const event = { eventId: randomUUID(), visitorId: randomUUID(), path: '/integration/', referrer: '' };
  const responses = await Promise.all([event, event, { ...event, eventId: randomUUID() }].map(payload =>
    server.fetch('/api/events', { method: 'POST', headers, body: JSON.stringify(payload) })));
  for (const response of responses) assert.equal(response.status, 204);

  await worker.evictDurableObject('ANALYTICS', { name: 'site-v1' });
  const response = await server.fetch('/api/summary?days=30', { headers: { Origin: origin } });
  await assert.rejects(checkSummary(response.clone(), 30, { revision: 'wrong-commit' }), /Deployed revision/);
  const data = await checkSummary(response, 30, { origin });
  assert.deepEqual(data.lifetime, { views: 2, visitors: 1 });
  assert.equal(data.period.visitors, 1);
  assert.deepEqual(data.pages, [{ path: '/integration/', views: 2 }]);

  const projectOrigin = 'https://air-embodied-brain.github.io';
  const trackProject = path => server.fetch('/api/events', {
    method: 'POST',
    headers: { ...headers, Origin: projectOrigin },
    body: JSON.stringify({ ...event, path }),
  });
  const projectSummaries = async site => checkSummary(
    await server.fetch(`/api/summary?days=30&site=${site}`, { headers: { Origin: origin } }),
    30, { origin },
  );
  assert.equal((await trackProject('/OxyGen/')).status, 204);
  assert.equal((await trackProject('/SteerDistill/article/?lang=zh')).status, 204);
  assert.equal((await trackProject('/OxyGen-unrelated/')).status, 400);
  assert.equal((await trackProject('/unrelated/')).status, 400);
  assert.deepEqual((await projectSummaries('oxygen')).pages, [{ path: '/OxyGen/', views: 1 }]);
  assert.deepEqual((await projectSummaries('steerdistill')).pages, [{ path: '/SteerDistill/article/', views: 1 }]);
  assert.deepEqual((await projectSummaries('personal')).pages, data.pages);
  assert.equal((await server.fetch('/api/summary?site=unknown')).status, 400);
  const preflight = await server.fetch('/api/events', { method: 'OPTIONS', headers: { Origin: projectOrigin } });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('access-control-allow-origin'), projectOrigin);

  const refreshed = await server.fetch('/api/summary?days=30&refresh=1', { headers: { Origin: origin } });
  assert.equal(refreshed.headers.get('Cache-Control'), 'no-store');
  assert.deepEqual((await checkSummary(refreshed, 30, { origin })).lifetime, data.lifetime);

  // Exercise the exact read-only command used after deployment over local HTTP.
  const { stdout } = await promisify(execFile)(process.execPath, [
    fileURLToPath(new URL('../../scripts/smoke.js', import.meta.url)), String(url), 'local',
  ], { env: { ...process.env, ANALYTICS_SITE_ORIGIN: origin }, timeout: 15000 });
  assert.match(stdout, /30-day summary: OK/);
  assert.match(stdout, /90-day summary: OK/);
});
