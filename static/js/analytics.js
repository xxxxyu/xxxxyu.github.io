import { fetchJSON } from './analytics-request.mjs';

const root = document.getElementById('analytics');
const copy = JSON.parse(document.getElementById('analytics-copy').textContent);
const locale = document.documentElement.lang === 'zh' ? 'zh-CN' : 'en-GB';
const numbers = new Intl.NumberFormat(locale);
const regions = new Intl.DisplayNames([locale], { type: 'region' });
const dates = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', timeZone: 'UTC' });
const search = new URLSearchParams(location.search);
const siteLinks = Array.from(root.querySelectorAll('[data-analytics-site]'));
const defaultSite = siteLinks[0];
const selectedSite = siteLinks.find(link => link.dataset.analyticsSite === search.get('site')) || defaultSite;
const siteOrigin = selectedSite.dataset.origin || location.origin;
selectedSite.setAttribute('aria-current', 'page');
const isLocal = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
const localCollector = isLocal && search.get('source') === 'local';
const endpoint = localCollector ? 'http://127.0.0.1:8787' : root.dataset.endpoint;
const isDemo = isLocal && !localCollector;
const status = document.getElementById('analytics-status');
const results = document.getElementById('analytics-results');
const refresh = document.getElementById('analytics-refresh');
const chart = document.getElementById('analytics-chart');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const cacheKey = `site-analytics-v4:${isDemo ? 'demo' : endpoint}:${selectedSite.dataset.analyticsSite}`;
const landCacheKey = `site-analytics-land:${root.dataset.map}`;
const dataCache = new Map();
const SUMMARY_MAX_AGE = 60000;
const metricDigits = { views: 6, visitors: 6, countries: 3, today: 4 };
const axisLayer = svgElement('g');
const barsLayer = svgElement('g');
const ticksLayer = svgElement('g');
chart.append(axisLayer, barsLayer, ticksLayer);
let selectedDays = search.get('days') === '90' ? 90 : 30;
let locationTab = 'countries';
let summary = null;
let controller;
let requestVersion = 0;
let lastAttemptAt = 0;
let landPromise;
let statusTimer;
let countFrame = 0;
let chartWidth = 0;
let chartVisible = false;
let chartNeedsReveal = false;

function validSummary(data) {
  return data && [30, 90].includes(data.period?.days) && Array.isArray(data.days) && data.days.length === data.period.days &&
    [data.period.views, data.period.visitors, data.period.countries, data.today?.views, data.lifetime?.views, data.lifetime?.visitors, Date.parse(data.generatedAt)].every(Number.isFinite) &&
    ['countries', 'cities', 'pages', 'referrers', 'points'].every(key => Array.isArray(data[key]));
}

function isFreshSummary(entry, now = Date.now()) {
  return Boolean(entry &&
    now - Math.min(entry.receivedAt, Date.parse(entry.data.generatedAt)) < SUMMARY_MAX_AGE &&
    entry.data.period.to === new Date(now).toISOString().slice(0, 10));
}

function readSession(key) {
  try {
    return JSON.parse(sessionStorage.getItem(key));
  } catch {
    return null;
  }
}

function writeSession(key, value) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Caching is optional when storage is unavailable or full.
  }
}

function restoreSnapshot() {
  const cached = readSession(cacheKey);
  if (!cached) return;
  for (const entry of Array.isArray(cached.entries) ? cached.entries : []) {
    if (validSummary(entry?.data) && Number.isFinite(entry.receivedAt) && Date.now() - entry.receivedAt < 86400000) {
      dataCache.set(entry.data.period.days, entry);
    }
  }
  for (const key of Object.keys(metricDigits)) {
    const digits = cached.digits?.[key];
    if (Number.isInteger(digits) && digits >= metricDigits[key] && digits <= 20) metricDigits[key] = digits;
  }
}

function updateRanges() {
  const url = new URL(location.href);
  if (selectedDays === 90) url.searchParams.set('days', '90');
  else url.searchParams.delete('days');
  if (url.href !== location.href) history.replaceState(history.state, '', url);

  for (const link of siteLinks) {
    const target = new URL(url);
    if (link === defaultSite) target.searchParams.delete('site');
    else target.searchParams.set('site', link.dataset.analyticsSite);
    link.href = target.href;
  }
  const language = document.getElementById('language-switch');
  if (language) language.search = url.search;

  root.querySelectorAll('[data-days]').forEach(button => {
    button.setAttribute('aria-pressed', String(Number(button.dataset.days) === selectedDays));
  });
}

function renderMetrics(initial) {
  cancelAnimationFrame(countFrame);
  const values = { ...summary.period, today: summary.today.views };
  const counters = [];
  root.querySelectorAll('[data-metric]').forEach(node => {
    const key = node.dataset.metric;
    const target = values[key];
    const from = Number(node.dataset.value || 0);
    const text = numbers.format(target);
    metricDigits[key] = Math.max(metricDigits[key], text.length);
    node.style.setProperty('--metric-digits', metricDigits[key]);
    node.setAttribute('aria-label', text);
    if (reducedMotion.matches || from === target) {
      node.dataset.value = String(target);
      node.textContent = text;
    } else {
      counters.push({ node, target, from });
    }
  });
  if (!counters.length) return;
  const start = performance.now();
  const duration = initial ? 900 : 420;
  const tick = now => {
    const progress = reducedMotion.matches ? 1 : Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    for (const { node, target, from } of counters) {
      const value = Math.round(from + (target - from) * eased);
      node.dataset.value = String(value);
      node.textContent = numbers.format(value);
    }
    countFrame = progress < 1 ? requestAnimationFrame(tick) : 0;
  };
  tick(start);
}

function revealChart() {
  if (!chartNeedsReveal || (!chartVisible && !reducedMotion.matches)) return;
  chartNeedsReveal = false;
  chart.classList.remove('analytics__chart--pending');
  if (reducedMotion.matches) return;
  const groups = Array.from(barsLayer.children);
  for (const [index, group] of groups.entries()) {
    for (const bar of group.querySelectorAll('rect')) {
      bar.animate([{ transform: 'scaleY(0)' }, { transform: 'scaleY(1)' }], {
        duration: 650,
        delay: index / Math.max(1, groups.length - 1) * 180,
        easing: 'cubic-bezier(0.22, 0.68, 0.25, 1)',
        fill: 'backwards',
      });
    }
  }
}

function element(tag, attributes = {}, text = '') {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  node.textContent = text;
  return node;
}

function svgElement(tag, attributes = {}, text = '') {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  node.textContent = text;
  return node;
}

function flag(code) {
  if (!/^[A-Z]{2}$/.test(code)) return '';
  return String.fromCodePoint(...Array.from(code, character => 127397 + character.charCodeAt(0)));
}

function country(code) {
  return /^[A-Z]{2}$/.test(code) ? regions.of(code) : copy.unknown;
}

function renderRanking(id, rows, type) {
  const list = document.getElementById(id);
  const items = [];
  for (const row of rows.slice(0, 8)) {
    const item = element('li');
    const label = element('span', { class: 'analytics__ranking-label' });
    let text;
    if (type === 'countries' || type === 'cities') {
      const code = row.code || row.country;
      label.append(element('span', { class: 'analytics__flag', 'aria-hidden': 'true' }, flag(code)));
      text = type === 'countries' ? country(code) : `${row.city}, ${country(code)}`;
    } else {
      text = type === 'pages' ? row.path : row.host || copy.direct;
    }
    const pageUrl = type === 'pages' && /^\/(?!\/)/.test(text) ? new URL(text, siteOrigin) : null;
    const canLink = pageUrl?.origin === siteOrigin;
    label.append(element(canLink ? 'a' : 'span', canLink ? { href: pageUrl.href, title: text } : { title: text }, text));
    item.append(label, element('span', { class: 'analytics__ranking-count' }, numbers.format(row.views)));
    items.push(item);
  }
  if (!items.length) items.push(element('li', { class: 'analytics__ranking-empty' }, copy.empty));
  list.replaceChildren(...items);
}

function renderLocations() {
  if (summary) renderRanking('analytics-locations', summary[locationTab], locationTab);
}

function loadLand() {
  if (!landPromise) {
    const cached = readSession(landCacheKey);
    const valid = Array.isArray(cached?.paths) && cached.paths.length && cached.paths.every(path => typeof path === 'string');
    const data = valid ? Promise.resolve(cached) : fetchJSON(root.dataset.map);
    landPromise = data.then(land => {
      document.getElementById('analytics-land').replaceChildren(...land.paths.map(path => svgElement('path', { d: path })));
      if (!valid) writeSession(landCacheKey, land);
    }).catch(() => {
      // Keep statistics usable without the base map; retry on the next load.
      landPromise = null;
    });
  }
  return landPromise;
}

function retireGroup(group, animate) {
  group.dataset.active = 'false';
  if (!animate || reducedMotion.matches) {
    group.remove();
    return;
  }
  const opacity = getComputedStyle(group).opacity;
  group.getAnimations().forEach(animation => animation.cancel());
  group.animate([{ opacity }, { opacity: 0 }], { duration: 220, fill: 'forwards' }).finished.then(() => {
    if (group.dataset.active === 'false') group.remove();
  }, () => {});
}

function renderMap() {
  const points = document.getElementById('analytics-points');
  const existing = new Map(Array.from(points.children, group => [group.dataset.key, group]));
  const maximum = Math.max(1, ...summary.points.map(point => point.views));
  for (const point of summary.points) {
    const key = `${point.lat},${point.lng}`;
    let group = existing.get(key);
    const entering = !group;
    if (!group) {
      group = svgElement('g', { 'data-key': key });
      group.append(svgElement('circle', { fill: 'url(#analytics-glow)' }), svgElement('circle'));
      group.lastElementChild.append(svgElement('title'));
      points.append(group);
    }
    existing.delete(key);
    group.getAnimations().forEach(animation => animation.cancel());
    group.dataset.active = 'true';
    const x = (point.lng + 180) / 360 * 1000;
    const y = (85 - point.lat) / 145 * 450;
    const strength = Math.sqrt(point.views / maximum);
    for (const circle of group.children) {
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
    }
    group.firstElementChild.setAttribute('r', 10 + strength * 25);
    group.lastElementChild.setAttribute('r', 2.5 + strength * 3.5);
    group.lastElementChild.setAttribute('opacity', 0.6 + strength * 0.4);
    group.querySelector('title').textContent = `${numbers.format(point.views)} ${copy.views} · ${point.lat}°, ${point.lng}°`;
    if (entering && !reducedMotion.matches) group.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 320 });
  }
  for (const group of existing.values()) retireGroup(group, true);
  document.getElementById('analytics-map-empty').hidden = summary.points.length > 0;
}

function barPositions() {
  const inverse = chart.getScreenCTM().inverse();
  return new Map(Array.from(barsLayer.querySelectorAll('rect'), bar => {
    const bounds = bar.getBoundingClientRect();
    const from = new DOMPoint(bounds.left, bounds.top).matrixTransform(inverse);
    const to = new DOMPoint(bounds.right, bounds.bottom).matrixTransform(inverse);
    return [bar, { x: from.x, y: from.y, width: to.x - from.x, height: to.y - from.y }];
  }));
}

function geometryFrame(geometry) {
  return Object.fromEntries(Object.entries(geometry).map(([key, value]) => [key, `${value}px`]));
}

function renderChart({ animate = false, initial = false } = {}) {
  if (!summary) return;
  const width = Math.max(300, Math.round(chart.parentElement.clientWidth));
  const morph = animate && !initial && chartVisible && !reducedMotion.matches;
  const previous = morph ? barPositions() : new Map();
  chart.getAnimations({ subtree: true }).forEach(animation => animation.cancel());
  chartWidth = width;
  const height = 200;
  const left = 38;
  const plotWidth = width - left - 4;
  const plotHeight = 150;
  const baseline = 168;
  const maximum = Math.max(4, ...summary.days.map(day => day.views));
  const ceiling = Math.ceil(maximum / 4) * 4;
  const slot = plotWidth / summary.days.length;
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  chartNeedsReveal = animate && initial && !reducedMotion.matches;
  chart.classList.toggle('analytics__chart--pending', chartNeedsReveal);
  axisLayer.replaceChildren();
  for (let index = 0; index <= 4; index++) {
    const y = baseline - plotHeight * index / 4;
    axisLayer.append(svgElement('line', { x1: left, x2: width, y1: y, y2: y, class: 'analytics__chart-grid' }));
    axisLayer.append(svgElement('text', { x: left - 9, y: y + 4, 'text-anchor': 'end' }, numbers.format(ceiling * index / 4)));
  }
  const existing = new Map(Array.from(barsLayer.children, group => [group.dataset.date, group]));
  for (const [index, day] of summary.days.entries()) {
    let group = existing.get(day.date);
    if (!group) {
      group = svgElement('g', { class: 'analytics__chart-day', 'data-date': day.date });
      group.append(svgElement('title'), svgElement('rect', { class: 'analytics__bar-views', rx: 1 }), svgElement('rect', { class: 'analytics__bar-visitors', rx: 1 }));
    }
    existing.delete(day.date);
    group.getAnimations().forEach(animation => animation.cancel());
    group.dataset.active = 'true';
    group.querySelector('title').textContent = `${day.date} · ${numbers.format(day.views)} ${copy.views} · ${numbers.format(day.visitors)} ${copy.visitors}`;
    for (const [field, fraction] of [['views', 0.72], ['visitors', 0.3]]) {
      const bar = group.querySelector(`.analytics__bar-${field}`);
      const barHeight = day[field] / ceiling * plotHeight;
      const geometry = { x: left + index * slot + slot * (1 - fraction) / 2, y: baseline - barHeight, width: Math.max(1, slot * fraction), height: barHeight };
      for (const [key, value] of Object.entries(geometry)) bar.setAttribute(key, value);
      if (morph) {
        const from = previous.get(bar) || { ...geometry, y: baseline, height: 0 };
        bar.animate([geometryFrame(from), geometryFrame(geometry)], { duration: 420, easing: 'cubic-bezier(0.22, 0.68, 0.25, 1)', fill: 'backwards' });
      }
    }
    barsLayer.append(group);
  }
  for (const group of existing.values()) retireGroup(group, morph);
  ticksLayer.replaceChildren();
  const ticks = [0, Math.floor((summary.days.length - 1) / 2), summary.days.length - 1];
  for (const [index, position] of ticks.entries()) {
    ticksLayer.append(svgElement('text', { x: left + (position + 0.5) * slot, y: height - 8, 'text-anchor': ['start', 'middle', 'end'][index] }, dates.format(new Date(`${summary.days[position].date}T00:00:00Z`))));
  }
  document.getElementById('analytics-chart-empty').hidden = summary.period.views > 0;
  revealChart();
}

function contentOf(data) {
  const { generatedAt, ...content } = data;
  return JSON.stringify(content);
}

function render(data) {
  const initial = !summary;
  const changed = initial || contentOf(summary) !== contentOf(data);
  summary = data;
  selectedDays = data.period.days;
  updateRanges();
  results.hidden = false;
  root.dataset.state = 'ready';
  updateStatus(data);
  if (!changed) return;
  renderMetrics(initial);
  renderMap();
  renderLocations();
  renderChart({ animate: true, initial });
  renderRanking('analytics-pages', summary.pages, 'pages');
  renderRanking('analytics-referrers', summary.referrers, 'referrers');
  document.getElementById('analytics-dates').textContent = `${summary.period.from} — ${summary.period.to}`;
  document.getElementById('analytics-lifetime').textContent = `${copy.lifetime} · ${numbers.format(summary.lifetime.views)} ${copy.views} · ${numbers.format(summary.lifetime.visitors)} ${copy.visitors}`;
}

function updateStatus(data) {
  const updated = new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(data.generatedAt));
  status.textContent = `${copy.updated} ${updated}`;
}

async function readSummary(range, { force, signal }) {
  const cached = dataCache.get(range);
  if (!force && isFreshSummary(cached)) return cached;
  lastAttemptAt = Date.now();
  let data;
  if (isDemo) {
    const { sampleAnalytics } = await import('./analytics-demo.mjs');
    data = sampleAnalytics(range, Date.now(), selectedSite.dataset.analyticsSite);
  } else {
    const url = new URL('/api/summary', endpoint);
    url.searchParams.set('days', range);
    if (selectedSite !== defaultSite) url.searchParams.set('site', selectedSite.dataset.analyticsSite);
    if (force) url.searchParams.set('refresh', '1');
    data = await fetchJSON(url, { signal });
  }
  if (!validSummary(data) || data.period.days !== range) throw new Error('Invalid statistics');
  return { data, receivedAt: Date.now() };
}

async function load({ force = false } = {}) {
  const version = ++requestVersion;
  const range = selectedDays;
  controller?.abort();
  const requestController = new AbortController();
  controller = requestController;
  clearTimeout(statusTimer);
  results.setAttribute('aria-busy', 'true');
  refresh.disabled = true;
  refresh.title = copy.refresh;
  refresh.setAttribute('aria-label', copy.refresh);
  delete refresh.dataset.range;
  if (!endpoint && !isDemo) {
    status.textContent = copy.unconfigured;
    root.dataset.state = 'unconfigured';
    results.hidden = true;
    results.setAttribute('aria-busy', 'false');
    return;
  }
  refresh.dataset.loading = 'true';
  statusTimer = setTimeout(() => { if (!summary) status.textContent = copy.loading; }, 250);
  loadLand();
  try {
    const cached = dataCache.get(range);
    if (cached) render(cached.data);
    const entry = await readSummary(range, { force, signal: requestController.signal });
    if (version !== requestVersion) return;
    dataCache.set(range, entry);
    render(entry.data);
    writeSession(cacheKey, { entries: Array.from(dataCache.values()), digits: metricDigits });
  } catch {
    if (version !== requestVersion) return;
    requestController.abort();
    status.textContent = summary ? `${copy.unavailable} ${copy.cached}` : copy.unavailable;
    refresh.title = copy.retry;
    refresh.setAttribute('aria-label', copy.retry);
    refresh.dataset.range = range;
    if (summary) {
      selectedDays = summary.period.days;
      updateRanges();
    } else {
      root.dataset.state = 'error';
      results.hidden = true;
    }
  } finally {
    if (version === requestVersion) {
      clearTimeout(statusTimer);
      results.setAttribute('aria-busy', 'false');
      refresh.disabled = false;
      delete refresh.dataset.loading;
    }
  }
}

root.querySelectorAll('[data-days]').forEach(button => {
  button.addEventListener('click', () => {
    const days = Number(button.dataset.days);
    if (days === selectedDays) return;
    selectedDays = days;
    updateRanges();
    load();
  });
});
root.querySelectorAll('[data-location]').forEach(button => {
  button.addEventListener('click', () => {
    locationTab = button.dataset.location;
    root.querySelectorAll('[data-location]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    renderLocations();
  });
});
refresh.addEventListener('click', () => {
  selectedDays = Number(refresh.dataset.range) || selectedDays;
  updateRanges();
  load({ force: true });
});

function refreshIfStale() {
  if (isDemo || document.hidden || refresh.disabled || Date.now() - lastAttemptAt < SUMMARY_MAX_AGE) return;
  if (!isFreshSummary(dataCache.get(selectedDays))) load({ force: true });
}

window.addEventListener('focus', refreshIfStale);
window.addEventListener('pageshow', refreshIfStale);
document.addEventListener('visibilitychange', refreshIfStale);
new ResizeObserver(() => {
  if (Math.max(300, Math.round(chart.parentElement.clientWidth)) !== chartWidth) renderChart();
}).observe(chart.parentElement);
new IntersectionObserver(entries => {
  chartVisible = entries.some(entry => entry.isIntersecting);
  revealChart();
}, { threshold: 0.15 }).observe(chart);
reducedMotion.addEventListener('change', () => {
  if (reducedMotion.matches) {
    root.getAnimations({ subtree: true }).filter(animation => animation.effect.getTiming().iterations !== Infinity)
      .forEach(animation => animation.finish());
    revealChart();
  }
});
restoreSnapshot();
updateRanges();
load();
