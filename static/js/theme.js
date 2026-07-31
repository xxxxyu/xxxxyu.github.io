(function() {
  var btn = document.getElementById('theme-toggle');
  var favicon = document.getElementById('favicon');
  var systemDark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var cycle = ['light', 'dark', 'system'];
  if (!btn) return;

  function current() {
    try {
      return localStorage.getItem('theme') || 'system';
    } catch (error) {
      return 'system';
    }
  }

  function store(mode) {
    try {
      if (mode === 'system') {
        localStorage.removeItem('theme');
      } else {
        localStorage.setItem('theme', mode);
      }
    } catch (error) {
      // The current interaction still applies even if storage is unavailable.
    }
  }

  function resolvedTheme(mode) {
    if (mode === 'system') {
      return systemDark && systemDark.matches ? 'dark' : 'light';
    }
    return mode;
  }

  function updateFavicon(mode) {
    if (!favicon) return;
    var theme = resolvedTheme(mode);
    var href = favicon.getAttribute(theme === 'dark' ? 'data-dark' : 'data-light');
    if (href) {
      favicon.setAttribute('href', href);
    }
  }

  function apply(mode) {
    if (mode === 'system') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', mode);
    }
    store(mode);
    document.documentElement.setAttribute('data-theme-mode', mode);
    btn.setAttribute('aria-label', 'Theme: ' + mode + ' (click to change)');
    updateFavicon(mode);
  }

  btn.addEventListener('click', function() {
    var idx = cycle.indexOf(current());
    apply(cycle[(idx + 1) % cycle.length]);
  });

  if (systemDark) {
    var handleSystemThemeChange = function() {
      if (current() === 'system') {
        updateFavicon('system');
      }
    };
    if (systemDark.addEventListener) {
      systemDark.addEventListener('change', handleSystemThemeChange);
    } else if (systemDark.addListener) {
      systemDark.addListener(handleSystemThemeChange);
    }
  }

  apply(current());
})();
