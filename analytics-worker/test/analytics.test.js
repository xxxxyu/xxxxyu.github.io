import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';
import worker, { AnalyticsStore } from '../src/index.js';

const ORIGIN = 'https://xxxxyu.github.io';
const NOW = Date.parse('2026-11-16T08:00:00Z');

function database() {
  const db = new DatabaseSync(':memory:');
  const storage = {
    sql: {
      exec(query, ...bindings) {
        if (query.trim().split(';').filter(part => part.trim()).length > 1) {
          db.exec(query);
          return { toArray: () => [] };
        }
        const rows = db.prepare(query).all(...bindings).map(row => ({ ...row }));
        return { toArray: () => rows };
      },
    },
    transactionSync(callback) {
      db.exec('BEGIN');
      try {
        const result = callback();
        db.exec('COMMIT');
        return result;
      } catch (error) {
        db.exec('ROLLBACK');
        throw error;
      }
    },
  };
  const store = new AnalyticsStore({ storage });
  const env = {
    ALLOWED_ORIGINS: ORIGIN,
    ANALYTICS: { getByName: () => ({ fetch: (url, init) => store.fetch(new Request(url, init)) }) },
  };
  return { db, storage, store, env };
}

function event(overrides = {}) {
  return { eventId: randomUUID(), visitor: 'visitor-hash-one', path: '/', referrer: 'google.com', country: 'CN', city: 'Beijing', lat: 40, lng: 116, ...overrides };
}

function request(payload = {}, headers = {}) {
  return new Request('https://analytics.example/api/track', {
    method: 'POST',
    headers: { Origin: ORIGIN, 'User-Agent': 'Mozilla/5.0 Chrome/140', 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify({ eventId: randomUUID(), visitorId: randomUUID(), path: '/', referrer: 'https://google.com/search?q=private', ...payload }),
  });
}

test('duplicate delivery counts once, and a returning visitor is unique across UTC days', () => {
  const { store } = database();
  const first = event();
  store.record(first, NOW - 86400000);
  store.record(first, NOW - 86400000);
  store.record(event(), NOW);
  store.record(event({ visitor: 'visitor-hash-two', path: '/research/' }), NOW);
  const data = store.summary(7, NOW);
  assert.deepEqual(data.lifetime, { views: 3, visitors: 2 });
  assert.equal(data.period.visitors, 2);
  assert.deepEqual(data.today, { views: 2, visitors: 2 });
  assert.deepEqual(data.days.slice(-2).map(day => day.visitors), [1, 2]);
  assert.equal(data.countries[0].views, 3);
  assert.equal(JSON.stringify(data).includes('visitor-hash'), false);
});

test('retention preserves lifetime counters and restarting the object preserves data', () => {
  const { storage, store } = database();
  store.record(event({ visitor: 'old' }), NOW - 450 * 86400000);
  store.record(event(), NOW);
  const reopened = new AnalyticsStore({ storage });
  const data = reopened.summary(90, NOW);
  assert.equal(data.period.views, 1);
  assert.deepEqual(data.lifetime, { views: 2, visitors: 2 });
  assert.equal(storage.sql.exec('SELECT COUNT(*) AS count FROM events').toArray()[0].count, 1);
});

test('a failed write rolls back the event and counters together', () => {
  const { db, store } = database();
  db.exec("CREATE TRIGGER fail_counter BEFORE INSERT ON meta BEGIN SELECT RAISE(ABORT, 'Simulated storage failure'); END");
  assert.throws(() => store.record(event(), NOW), /Simulated storage failure/);
  const data = store.summary(7, NOW);
  assert.deepEqual(data.lifetime, { views: 0, visitors: 0 });
  assert.equal(data.period.views, 0);
});

test('the collector rejects invalid events and respects browser exclusions', async () => {
  const { env, store } = database();
  const malformed = new Request('https://analytics.example/api/track', {
    method: 'POST', headers: { Origin: ORIGIN, 'User-Agent': 'Browser', 'Content-Type': 'application/json' }, body: '{',
  });
  const rejected = [
    [request({}, { Origin: 'https://other.example' }), 403],
    [request({}, { DNT: '1' }), 204],
    [request({}, { 'Sec-GPC': '1' }), 204],
    [request({}, { 'User-Agent': 'Googlebot' }), 204],
    [request({ visitorId: 'invalid' }), 400],
    [request({ path: '/analytics/' }), 400],
    [request({ path: '/zh/analytics/' }), 400],
    [request({ path: '//external.example/' }), 400],
    [request({ path: '/' + 'a'.repeat(2200) }), 400],
    [malformed, 400],
  ];
  for (const [input, status] of rejected) assert.equal((await worker.fetch(input, env)).status, status);
  const preflight = await worker.fetch(new Request('https://analytics.example/api/track', { method: 'OPTIONS', headers: { Origin: ORIGIN } }), env);
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('Access-Control-Allow-Origin'), ORIGIN);
  assert.equal(store.summary(7).lifetime.views, 0);
});

test('concurrent page views count one visitor and expose only aggregate data', async () => {
  const { env, store, storage } = database();
  const visitorId = randomUUID();
  const inputs = Array.from({ length: 25 }, () => request({ visitorId, path: '/blog/post/?token=private#section' }));
  for (const input of inputs) {
    Object.defineProperty(input, 'cf', { value: { country: 'CN', city: 'Beijing', latitude: '39.9', longitude: '116.4' } });
  }
  const responses = await Promise.all(inputs.map(input => worker.fetch(input, env)));
  assert.equal(responses.every(response => response.status === 204), true);
  assert.deepEqual(store.summary(30).lifetime, { views: 25, visitors: 1 });
  const { visitor, ...saved } = storage.sql.exec('SELECT visitor, path, referrer, country, city, lat, lng FROM events LIMIT 1').toArray()[0];
  assert.equal(visitor.length, 64);
  assert.notEqual(visitor, visitorId);
  assert.deepEqual(saved, { path: '/blog/post/', referrer: 'google.com', country: 'CN', city: 'Beijing', lat: 40, lng: 116 });
  const response = await worker.fetch(new Request('https://analytics.example/api/analytics?days=7', { headers: { Origin: ORIGIN } }), env);
  const data = await response.json();
  assert.equal(data.period.visitors, 1);
  assert.equal(data.days.length, 7);
  assert.equal(JSON.stringify(data).includes(visitorId), false);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), ORIGIN);
});

test('unknown and invalid coordinates remain absent in the public summary', async () => {
  const { env, store } = database();
  const invalid = request();
  Object.defineProperty(invalid, 'cf', { value: { country: 'XX', latitude: '', longitude: '999' } });
  for (const input of [request(), invalid]) assert.equal((await worker.fetch(input, env)).status, 204);
  const data = store.summary(30);
  assert.equal(data.lifetime.views, 2);
  assert.deepEqual(data.points, []);
  assert.deepEqual(data.countries, []);
});
