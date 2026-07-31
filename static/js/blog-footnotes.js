(function() {
  var FOOTNOTE_GAP = 12;
  var SIDE_NOTES_QUERY = '(min-width: 1024px)';
  var content = document.querySelector('.blog-post__content');
  var panel = document.getElementById('footnotes-panel');
  if (!content || !panel) return;

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function relocateFootnotes() {
    var defs = content.querySelectorAll('.footnote-definition');
    if (defs.length === 0) return null;

    var ol = document.createElement('ol');
    ol.className = 'footnotes-list';
    defs.forEach(function(def) {
      var li = document.createElement('li');
      li.id = def.id;
      Array.from(def.childNodes).forEach(function(node) {
        if (node.nodeName === 'SUP' && node.classList && node.classList.contains('footnote-definition-label')) return;
        li.appendChild(node.cloneNode(true));
      });
      ol.appendChild(li);
      def.remove();
    });
    panel.appendChild(ol);
    return ol;
  }

  function collectFootnoteReferences() {
    var refs = {};
    content.querySelectorAll('.footnote-reference a[href^="#"]').forEach(function(ref) {
      refs[decodeURIComponent(ref.hash.slice(1))] = ref;
    });
    return refs;
  }

  function getAnchorOffset() {
    var styles = window.getComputedStyle(document.documentElement);
    var offset = parseFloat(styles.scrollPaddingTop);
    if (!Number.isNaN(offset)) return offset;

    var nav = document.querySelector('.nav');
    return nav ? nav.getBoundingClientRect().height : 0;
  }

  function alignFootnotes() {
    var isSideNotes = window.matchMedia(SIDE_NOTES_QUERY).matches;
    var panelBox = panel.getBoundingClientRect();
    if (!isSideNotes || panelBox.width === 0) {
      panel.classList.remove('blog-post-layout__footnotes--aligned');
      list.style.height = '';
      notes.forEach(function(note) {
        note.style.top = '';
      });
      return;
    }

    panel.classList.add('blog-post-layout__footnotes--aligned');
    var panelTop = panelBox.top + window.scrollY;
    var previousBottom = -FOOTNOTE_GAP;

    notes.forEach(function(note) {
      note.style.top = '';
    });

    notes.forEach(function(note) {
      var ref = refsById[note.id];
      var refTop = ref ? ref.getBoundingClientRect().top + window.scrollY : panelTop + previousBottom + FOOTNOTE_GAP;
      var targetTop = Math.max(Math.round(refTop - panelTop), previousBottom + FOOTNOTE_GAP);

      note.style.top = targetTop + 'px';
      previousBottom = targetTop + note.offsetHeight;
    });

    list.style.height = Math.ceil(previousBottom) + 'px';
  }

  function scrollToFootnote(id, updateHash) {
    alignFootnotes();
    var note = document.getElementById(id);
    if (!note || !panel.contains(note)) return false;

    var top = note.getBoundingClientRect().top + window.scrollY - getAnchorOffset();
    if (updateHash && window.history && window.history.pushState) {
      window.history.pushState(null, '', '#' + id);
    }
    window.scrollTo({
      top: Math.max(0, Math.round(top)),
      behavior: prefersReducedMotion() ? 'auto' : 'smooth'
    });
    return true;
  }

  function restoreInitialFootnoteHash() {
    var id = decodeURIComponent(window.location.hash.slice(1));
    if (!id) return;

    window.requestAnimationFrame(function() {
      scrollToFootnote(id, false);
    });
  }

  var list = relocateFootnotes();
  if (!list) return;

  var notes = Array.from(list.querySelectorAll('li'));
  var refsById = collectFootnoteReferences();

  var rafId = null;
  function scheduleAlign() {
    if (rafId !== null) return;
    rafId = window.requestAnimationFrame(function() {
      rafId = null;
      alignFootnotes();
    });
  }

  scheduleAlign();
  restoreInitialFootnoteHash();
  content.addEventListener('click', function(event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var link = event.target.closest && event.target.closest('.footnote-reference a[href^="#"]');
    if (!link || !content.contains(link)) return;

    var id = decodeURIComponent(link.hash.slice(1));
    if (!id) return;

    if (scrollToFootnote(id, true)) {
      event.preventDefault();
    }
  });
  window.addEventListener('resize', scheduleAlign, { passive: true });
  window.addEventListener('load', scheduleAlign, { once: true });
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(scheduleAlign);
  }
  content.querySelectorAll('img, video').forEach(function(media) {
    media.addEventListener('load', scheduleAlign, { passive: true });
    media.addEventListener('loadedmetadata', scheduleAlign, { passive: true });
  });
})();
