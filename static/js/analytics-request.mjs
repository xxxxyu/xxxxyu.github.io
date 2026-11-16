// Bound both the connection and response body; a stalled collector must not
// leave the dashboard loading indefinitely.
export async function fetchJSON(url, { signal } = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort(signal.reason);
  if (signal?.aborted) cancel();
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => {
    controller.abort(new DOMException('Request timed out', 'TimeoutError'));
  }, 10000);
  try {
    const response = await fetch(url, { signal: controller.signal, credentials: 'omit' });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
