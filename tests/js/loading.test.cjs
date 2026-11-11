const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const flush = () => new Promise(setImmediate);
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}
function events(properties = {}) {
  const listeners = new Map();
  return Object.assign(properties, {
    addEventListener(name, listener) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(listener);
    },
    emit(name) { (listeners.get(name) || []).forEach(listener => listener()); },
  });
}
function run(name, context) {
  vm.runInNewContext(readFileSync(path.join(__dirname, '../../static/js', name), 'utf8'), context);
}

function videoPage() {
  const fonts = deferred();
  const pauses = [];
  const video = events({
    paused: true,
    play() { this.paused = false; this.emit('play'); return Promise.resolve(); },
    pause() {
      if (this.paused) return;
      this.paused = true;
      pauses.push(() => this.emit('pause'));
    },
  });
  const motion = events({ matches: false });
  const connection = events({ saveData: false });
  const document = events({
    readyState: 'loading', hidden: false, fonts: { ready: fonts.promise },
    querySelectorAll: () => [video],
  });
  let intersect;
  class IntersectionObserver {
    constructor(callback) { intersect = callback; }
    observe() {}
  }
  const window = events({ IntersectionObserver, matchMedia: () => motion });
  run('blog-video.js', { document, window, navigator: { connection }, IntersectionObserver });
  return {
    video, document, motion, connection,
    show(ratio) { intersect([{ target: video, isIntersecting: ratio > 0, intersectionRatio: ratio }]); },
    deliverPauses() { pauses.splice(0).forEach(deliver => deliver()); },
    async ready() { window.emit('load'); fonts.resolve(); await flush(); },
  };
}

test('video waits for assets and sufficient visibility', async () => {
  const page = videoPage();
  page.show(1);
  assert(page.video.paused);
  await page.ready();
  assert(!page.video.paused);
  page.show(0.05);
  assert(page.video.paused);
});

test('queued automatic pauses do not become reader pauses', async () => {
  const page = videoPage();
  await page.ready();
  page.show(1);
  page.show(0);
  page.show(1);
  page.deliverPauses();
  page.document.hidden = true;
  page.document.emit('visibilitychange');
  page.deliverPauses();
  page.document.hidden = false;
  page.document.emit('visibilitychange');
  assert(!page.video.paused);
  page.video.pause(); // An explicit reader pause survives subsequent visibility changes.
  page.deliverPauses();
  page.show(0);
  page.show(1);
  assert(page.video.paused);
});

test('motion and data-saving preferences react to changes', async () => {
  const page = videoPage();
  page.motion.matches = true;
  await page.ready();
  page.show(1);
  assert(page.video.paused);
  page.motion.matches = false;
  page.motion.emit('change');
  assert(!page.video.paused);
  page.connection.saveData = true;
  page.connection.emit('change');
  page.deliverPauses();
  assert(page.video.paused);
  page.connection.saveData = false;
  page.connection.emit('change');
  assert(!page.video.paused);
});

function searchPage(blockStorage = false) {
  function element() {
    return events({
      children: [], dataset: {}, classList: { add() {}, remove() {} },
      appendChild(child) { this.children.push(child); },
      setAttribute() {}, removeAttribute() {}, scrollIntoView() {}, focus() {},
    });
  }
  const ids = Object.fromEntries(['search-button', 'search-modal', 'search-input', 'search-close', 'search-results'].map(id => [id, element()]));
  ids['search-input'].value = 'needle';
  ids['search-modal'].dataset.emptyLabel = 'Empty';
  const requests = [];
  let language = 'en';
  const document = events({
    documentElement: { lang: 'en' }, body: element(), activeElement: element(),
    getElementById: id => ids[id], createElement: element, createDocumentFragment: element,
  });
  run('search.js', {
    document, window: { location: {} }, navigator: { platform: 'Linux' },
    localStorage: { getItem() { if (blockStorage) throw new Error('Blocked'); return language; } },
    fetch(url) { const request = deferred(); requests.push({ url, ...request }); return request.promise; },
    requestAnimationFrame: callback => callback(), setTimeout, clearTimeout,
    console: { error() {} },
  });
  return {
    requests,
    open(lang = 'en') { language = lang; ids['search-button'].emit('click'); },
    result() { return ids['search-results'].children.at(-1)?.children[0]; },
    async reply(index, payload, ok = true) {
      requests[index].resolve({ ok, status: ok ? 200 : 503, json: async () => payload });
      await flush();
    },
  };
}

test('search retries temporary errors when local storage is blocked', async () => {
  const page = searchPage(true);
  page.open();
  await page.reply(0, null, false);
  page.open();
  await page.reply(1, [{ url: '/paper/', title: 'Needle paper' }]);
  assert.equal(page.requests.length, 2);
  assert.equal(page.result().href, '/paper/');
});

test('older same-language requests cannot replace or invalidate current results', async () => {
  const page = searchPage();
  page.open('en');
  page.open('zh');
  page.open('en');
  await page.reply(2, [{ url: '/fresh/', title: 'Needle fresh' }]);
  await page.reply(0, null, false);
  await page.reply(1, { documents: [{ id: '/zh/old/', title: 'Needle old' }] });
  page.open('en');
  await flush();
  assert.equal(page.requests.length, 3);
  assert.equal(page.result().href, '/fresh/');
});
