const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.resolve(__dirname, '../../static/js/analytics-track.js'), 'utf8');

function browser(options = {}) {
  const values = new Map(Object.entries(options.storage || {}));
  const requests = [];
  const context = {
    window: {},
    location: { hostname: options.hostname || 'xxxxyu.github.io', origin: 'https://xxxxyu.github.io', pathname: options.pathname || '/' },
    document: { currentScript: { dataset: { endpoint: 'https://stats.example', siteOrigin: 'https://xxxxyu.github.io' } }, referrer: 'https://google.com/search?q=private' },
    navigator: options.navigator || {},
    localStorage: { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) },
    sessionStorage: { getItem: () => null, setItem() {} },
    crypto: { randomUUID }, URL, Date, JSON,
    fetch: (url, init) => { requests.push({ url: String(url), ...init }); return Promise.resolve({ ok: true }); },
  };
  vm.createContext(context);
  return { values, requests, run: () => vm.runInContext(source, context) };
}

test('tracking sends one event without cookies or query strings', () => {
  const app = browser();
  app.run();
  app.run();
  assert.equal(app.requests.length, 1);
  assert.equal(app.requests[0].url, 'https://stats.example/api/events');
  const payload = JSON.parse(app.requests[0].body);
  assert.equal(payload.referrer, 'https://google.com');
  assert.equal(payload.path, '/');
  assert.equal(app.requests[0].credentials, 'omit');
});

test('local previews, dashboard views, opt-out, DNT, and GPC never create an identity', () => {
  for (const options of [
    { hostname: '127.0.0.1' }, { pathname: '/analytics/' }, { pathname: '/zh/analytics/' },
    { storage: { 'site-analytics-optout': '1' } },
    { navigator: { doNotTrack: '1' } }, { navigator: { globalPrivacyControl: true } },
  ]) {
    const app = browser(options);
    app.run();
    assert.equal(app.requests.length, 0);
    assert.equal(app.values.has('site-analytics-visitor'), false);
  }
});

test('visitor identity is reused while valid and regenerated when missing, expired or malformed', () => {
  const id = randomUUID();
  const expires = Date.now() + 100000;
  for (const [stored, reused] of [
    [null, false],
    [{ id, expires }, true],
    [{ id, expires: 1 }, false],
    [{ id: 'invalid', expires }, false],
    [{ id: [id], expires }, false],
  ]) {
    const app = browser({ storage: { 'site-analytics-visitor': JSON.stringify(stored) } });
    app.run();
    const saved = JSON.parse(app.values.get('site-analytics-visitor'));
    assert.equal(JSON.parse(app.requests[0].body).visitorId, saved.id);
    assert.match(saved.id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    assert.equal(saved.id === id, reused);
    assert.equal(saved.expires === expires, reused);
  }
});

test('preview ranges preserve daily data and aggregate visit totals', async () => {
  const { sampleAnalytics } = await import('../../static/js/analytics-demo.mjs');
  const now = Date.parse('2026-11-16T08:00:00Z');
  const recent = sampleAnalytics(30, now);
  const longer = sampleAnalytics(90, now);
  assert.equal(recent.days.length, 30);
  assert.equal(longer.days.length, 90);
  assert.deepEqual(longer.days.slice(-30), recent.days);
  assert.deepEqual(longer.today, recent.today);
  assert.ok(longer.points.length > recent.points.length);
  assert.ok(longer.period.countries > recent.period.countries);
  for (const data of [recent, longer]) {
    for (const rows of [data.days, data.points, data.countries, data.cities, data.pages, data.referrers]) {
      assert.equal(rows.reduce((total, row) => total + row.views, 0), data.period.views);
    }
  }
});
