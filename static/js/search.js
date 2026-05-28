(function () {
  const button = document.getElementById('search-button');
  const modal = document.getElementById('search-modal');
  if (!button || !modal) return;

  const input = document.getElementById('search-input');
  const closeBtn = document.getElementById('search-close');
  const resultsEl = document.getElementById('search-results');

  const MAX_RESULTS = 8;
  const SNIPPET_LEN = 140;

  let docs = null;
  let docsPromise = null;
  let lastFocused = null;
  let activeIndex = -1;
  let currentItems = [];

  function loadDocs() {
    if (docsPromise) return docsPromise;
    const lang = document.documentElement.lang || 'en';
    docsPromise = fetch('/search_index.' + lang + '.json')
      .then((r) => r.json())
      .then((idx) => {
        docs = Object.values(idx.documentStore.docs).map((d) => ({
          id: d.id,
          title: d.title || '',
          description: d.description || '',
          body: d.body || '',
          titleLower: (d.title || '').toLowerCase(),
          descLower: (d.description || '').toLowerCase(),
          bodyLower: (d.body || '').toLowerCase(),
        }));
        return docs;
      })
      .catch((err) => {
        console.error('Failed to load search index', err);
        docs = [];
        return docs;
      });
    return docsPromise;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function highlight(text, terms) {
    if (!terms.length) return escapeHtml(text);
    const pattern = terms
      .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('|');
    return escapeHtml(text).replace(
      new RegExp('(' + pattern + ')', 'gi'),
      '<mark>$1</mark>'
    );
  }

  function makeSnippet(body, terms) {
    if (!body) return '';
    if (!terms.length) {
      return body.length > SNIPPET_LEN
        ? body.slice(0, SNIPPET_LEN) + '…'
        : body;
    }
    const lower = body.toLowerCase();
    let pos = -1;
    for (const t of terms) {
      const i = lower.indexOf(t);
      if (i >= 0 && (pos === -1 || i < pos)) pos = i;
    }
    if (pos === -1) {
      return body.length > SNIPPET_LEN
        ? body.slice(0, SNIPPET_LEN) + '…'
        : body;
    }
    const start = Math.max(0, pos - 40);
    const end = Math.min(body.length, start + SNIPPET_LEN);
    let snippet = body.slice(start, end);
    if (start > 0) snippet = '…' + snippet;
    if (end < body.length) snippet = snippet + '…';
    return snippet;
  }

  function score(doc, terms) {
    let s = 0;
    for (const t of terms) {
      if (!t) continue;
      if (doc.titleLower.includes(t)) s += 10;
      if (doc.descLower.includes(t)) s += 4;
      const bodyMatches = doc.bodyLower.split(t).length - 1;
      s += bodyMatches;
    }
    return s;
  }

  function render(query) {
    const q = query.trim().toLowerCase();
    resultsEl.innerHTML = '';
    activeIndex = -1;
    currentItems = [];

    if (!q || !docs) return;

    const terms = q.split(/\s+/).filter(Boolean);
    const matches = docs
      .map((d) => ({ doc: d, score: score(d, terms) }))
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_RESULTS);

    if (!matches.length) {
      resultsEl.innerHTML =
        '<div class="search-modal__empty">No results found.</div>';
      return;
    }

    const frag = document.createDocumentFragment();
    matches.forEach((m, i) => {
      const a = document.createElement('a');
      a.className = 'search-modal__item';
      a.href = m.doc.id;
      a.setAttribute('role', 'option');
      a.id = 'search-result-' + i;
      a.innerHTML =
        '<span class="search-modal__title">' +
        highlight(m.doc.title, terms) +
        '</span>' +
        '<span class="search-modal__snippet">' +
        highlight(makeSnippet(m.doc.body, terms), terms) +
        '</span>';
      frag.appendChild(a);
      currentItems.push(a);
    });
    resultsEl.appendChild(frag);
    setActive(0);
  }

  function setActive(i) {
    if (!currentItems.length) return;
    if (activeIndex >= 0 && currentItems[activeIndex]) {
      currentItems[activeIndex].removeAttribute('aria-selected');
    }
    activeIndex = ((i % currentItems.length) + currentItems.length) %
      currentItems.length;
    const el = currentItems[activeIndex];
    el.setAttribute('aria-selected', 'true');
    input.setAttribute('aria-activedescendant', el.id);
    el.scrollIntoView({ block: 'nearest' });
  }

  function open() {
    lastFocused = document.activeElement;
    modal.hidden = false;
    document.body.classList.add('search-open');
    loadDocs().then(() => {
      if (input.value) render(input.value);
    });
    requestAnimationFrame(() => input.focus());
  }

  function close() {
    modal.hidden = true;
    document.body.classList.remove('search-open');
    input.value = '';
    resultsEl.innerHTML = '';
    activeIndex = -1;
    currentItems = [];
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  let debounceTimer;
  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => render(input.value), 80);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive(activeIndex + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive(activeIndex - 1);
    } else if (e.key === 'Enter') {
      if (activeIndex >= 0 && currentItems[activeIndex]) {
        e.preventDefault();
        window.location.href = currentItems[activeIndex].href;
      }
    }
  });

  button.addEventListener('click', open);
  button.addEventListener('mouseenter', loadDocs, { once: true });
  closeBtn.addEventListener('click', close);

  modal.addEventListener('click', (e) => {
    if (e.target === modal) close();
  });

  document.addEventListener('keydown', (e) => {
    const isMac = /mac/i.test(navigator.platform);
    const meta = isMac ? e.metaKey : e.ctrlKey;
    if (meta && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      modal.hidden ? open() : close();
    } else if (e.key === 'Escape' && !modal.hidden) {
      close();
    } else if (e.key === '/' && modal.hidden) {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
        open();
      }
    }
  });
})();
