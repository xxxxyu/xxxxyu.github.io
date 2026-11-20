(function() {
  var script = document.currentScript;
  if (!script || window.__siteAnalyticsSent) return;
  if (['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) return;
  if (navigator.doNotTrack === '1' || navigator.globalPrivacyControl === true) return;

  try {
    var endpoint = new URL(script.dataset.endpoint);
    var site = new URL(script.dataset.siteUrl || script.dataset.siteOrigin);
    if (endpoint.protocol !== 'https:' || site.origin !== location.origin) return;
    var entry = window.performance?.getEntriesByType('navigation')[0];
    var path = entry ? new URL(entry.name).pathname : location.pathname;
    if (/^\/(?:zh\/)?analytics(?:\/|$)/.test(path)) return;
    var basePath = site.pathname.endsWith('/') ? site.pathname : site.pathname + '/';
    if (path !== basePath.slice(0, -1) && !path.startsWith(basePath)) return;
    if (localStorage.getItem('site-analytics-optout') === '1') return;

    var scope = basePath === '/' ? '' : ':' + basePath;
    var visitorKey = 'site-analytics-visitor' + scope;
    var sourceKey = 'site-analytics-source' + scope;
    var now = Date.now();
    var visitor;
    try {
      visitor = JSON.parse(localStorage.getItem(visitorKey));
    } catch (error) {
      visitor = null;
    }
    if (!visitor || typeof visitor.id !== 'string' ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(visitor.id) ||
        !(visitor.expires > now)) {
      visitor = { id: crypto.randomUUID(), expires: now + 365 * 86400000 };
      localStorage.setItem(visitorKey, JSON.stringify(visitor));
    }

    var referrer = '';
    try {
      var source = new URL(document.referrer);
      if (source.origin !== location.origin) referrer = source.origin;
    } catch (error) {
      // A direct visit has no referring domain.
    }
    try {
      var session = JSON.parse(sessionStorage.getItem(sourceKey));
      if (!referrer && session && now - session.lastSeen < 30 * 60000) referrer = session.referrer;
      sessionStorage.setItem(sourceKey, JSON.stringify({ referrer: referrer, lastSeen: now }));
    } catch (error) {
      // The current page's referrer is enough when session storage is blocked.
    }

    window.__siteAnalyticsSent = true;
    fetch(new URL('/api/events', endpoint), {
      method: 'POST',
      mode: 'cors',
      credentials: 'omit',
      keepalive: true,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        eventId: crypto.randomUUID(),
        visitorId: visitor.id,
        path: path,
        referrer: referrer
      })
    }).catch(function() {
      // Measurement is optional and must not interrupt navigation.
    });
  } catch (error) {
    // Skip measurement when browser storage or configuration is unavailable.
  }
})();
