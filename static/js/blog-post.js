(function() {
  var WIDE_TOC_QUERY = '(min-width: 1440px)';
  var content = document.querySelector('.blog-post__content');
  if (!content) return;

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function initMotionPreferences() {
    if (!prefersReducedMotion()) return;

    content.querySelectorAll('video[autoplay]').forEach(function(video) {
      video.removeAttribute('autoplay');
      video.pause();
    });
  }

  function getAnchorOffset() {
    var styles = window.getComputedStyle(document.documentElement);
    var offset = parseFloat(styles.scrollPaddingTop);
    if (!Number.isNaN(offset)) return offset;

    var nav = document.querySelector('.nav');
    return nav ? nav.getBoundingClientRect().height : 0;
  }

  function initSectionAnchors() {
    content.querySelectorAll('h2[id], h3[id], h4[id]').forEach(function(heading) {
      if (heading.querySelector('.heading-anchor')) return;

      var anchor = document.createElement('a');
      anchor.className = 'heading-anchor';
      anchor.href = '#' + encodeURIComponent(heading.id);
      anchor.setAttribute('aria-label', 'Link to "' + heading.textContent.trim() + '"');
      anchor.textContent = '#';
      heading.insertBefore(anchor, heading.firstChild);
    });
  }

  function initActiveToc() {
    var toc = document.querySelector('.blog-post-layout__toc');
    if (!toc) return;

    var items = Array.from(toc.querySelectorAll('.toc__list a[href^="#"]')).map(function(link) {
      var id = decodeURIComponent(link.hash.slice(1));
      return {
        link: link,
        heading: id ? document.getElementById(id) : null
      };
    }).filter(function(item) {
      return item.heading;
    });
    if (items.length === 0) return;

    var activeLink = null;
    var rafId = null;
    var wideToc = window.matchMedia(WIDE_TOC_QUERY);

    function setActive(link) {
      if (activeLink === link) return;
      if (activeLink) {
        activeLink.classList.remove('toc__link--active');
        activeLink.removeAttribute('aria-current');
      }
      activeLink = link;
      if (activeLink) {
        activeLink.classList.add('toc__link--active');
        activeLink.setAttribute('aria-current', 'location');
      }
    }

    function updateActiveToc() {
      rafId = null;
      if (!wideToc.matches) {
        setActive(null);
        return;
      }

      var offset = getAnchorOffset() + 8;
      var current = items[0];
      items.forEach(function(item) {
        if (item.heading.getBoundingClientRect().top <= offset) {
          current = item;
        }
      });

      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        current = items[items.length - 1];
      }
      setActive(current.link);
    }

    function scheduleActiveToc() {
      if (rafId !== null) return;
      rafId = window.requestAnimationFrame(updateActiveToc);
    }

    scheduleActiveToc();
    window.addEventListener('scroll', scheduleActiveToc, { passive: true });
    window.addEventListener('resize', scheduleActiveToc, { passive: true });
    window.addEventListener('load', scheduleActiveToc, { once: true });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(scheduleActiveToc);
    }
  }

  initMotionPreferences();
  initSectionAnchors();
  initActiveToc();
})();
