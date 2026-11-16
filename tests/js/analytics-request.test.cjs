const assert = require('node:assert/strict');
const test = require('node:test');

function pending(signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  });
}

for (const phase of ['connection', 'response body']) {
  test(`statistics requests time out during a stalled ${phase}`, async t => {
    const { fetchJSON } = await import('../../static/js/analytics-request.mjs');
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let requestSignal;
    t.mock.method(globalThis, 'fetch', async (url, { signal }) => {
      requestSignal = signal;
      return phase === 'connection' ? pending(signal) : { ok: true, json: () => pending(signal) };
    });
    const request = fetchJSON('https://stats.example/api/summary');
    await Promise.resolve();
    t.mock.timers.tick(9999);
    assert.equal(requestSignal.aborted, false);
    t.mock.timers.tick(1);
    await assert.rejects(request, { name: 'TimeoutError' });
    assert.equal(requestSignal.aborted, true);
  });
}

test('range changes cancel requests, including an already cancelled request', async t => {
  const { fetchJSON } = await import('../../static/js/analytics-request.mjs');
  t.mock.method(globalThis, 'fetch', (url, { signal }) => pending(signal));
  for (const alreadyAborted of [false, true]) {
    const controller = new AbortController();
    if (alreadyAborted) controller.abort();
    const request = fetchJSON('https://stats.example/api/summary', { signal: controller.signal });
    controller.abort();
    await assert.rejects(request, { name: 'AbortError' });
  }
});

test('completed requests release their deadline and cancellation listener', async t => {
  const { fetchJSON } = await import('../../static/js/analytics-request.mjs');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const controller = new AbortController();
  let requestSignal;
  t.mock.method(globalThis, 'fetch', async (url, { signal, credentials }) => {
    requestSignal = signal;
    assert.equal(credentials, 'omit');
    return { ok: true, json: async () => ({ views: 3 }) };
  });
  assert.deepEqual(await fetchJSON('https://stats.example', { signal: controller.signal }), { views: 3 });
  controller.abort();
  t.mock.timers.tick(10000);
  assert.equal(requestSignal.aborted, false);
});
