(function() {
  var switcher = document.getElementById('language-switch');
  var messages = window.siteI18n;
  if (!switcher || !messages) return;

  function storedLanguage() {
    try {
      var stored = localStorage.getItem('site-language');
      return stored === 'en' || stored === 'zh' ? stored : null;
    } catch (error) {
      return null;
    }
  }

  function alternateFor(language) {
    return document.querySelector('link[rel="alternate"][hreflang="' + language + '"]');
  }

  function apply(language) {
    var copy = messages[language];
    document.documentElement.setAttribute('data-site-language', language);
    document.querySelectorAll('[data-i18n-text]').forEach(function(element) {
      element.textContent = copy[element.getAttribute('data-i18n-text')];
    });
    document.querySelectorAll('[data-i18n-aria]').forEach(function(element) {
      element.setAttribute('aria-label', copy[element.getAttribute('data-i18n-aria')]);
    });
    document.querySelectorAll('[data-i18n-title]').forEach(function(element) {
      var suffix = element.getAttribute('data-title-suffix') || '';
      element.title = copy[element.getAttribute('data-i18n-title')] + suffix;
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function(element) {
      element.placeholder = copy[element.getAttribute('data-i18n-placeholder')];
    });
    document.querySelectorAll('[data-i18n-empty]').forEach(function(element) {
      element.setAttribute('data-empty-label', copy[element.getAttribute('data-i18n-empty')]);
    });
    document.querySelectorAll('[data-href-en][data-href-zh]').forEach(function(element) {
      element.href = element.getAttribute('data-href-' + language);
      var destinationLanguage = element.getAttribute('data-content-lang-' + language);
      if (destinationLanguage) {
        element.setAttribute('lang', destinationLanguage);
        element.setAttribute('hreflang', destinationLanguage);
      } else {
        element.removeAttribute('lang');
        element.removeAttribute('hreflang');
      }
    });
    document.querySelectorAll('[data-title-en][data-title-zh]').forEach(function(element) {
      element.textContent = element.getAttribute('data-title-' + language);
    });

    var target = language === 'zh' ? 'en' : 'zh';
    var alternate = alternateFor(target);
    var hasExactTranslation = alternate && alternate.href !== window.location.href;
    var label = hasExactTranslation ? copy.language_switch_exact : copy.language_switch;
    switcher.href = hasExactTranslation ? alternate.href : window.location.href;
    switcher.textContent = target === 'zh' ? '中' : 'EN';
    switcher.setAttribute('lang', target);
    switcher.setAttribute('hreflang', target);
    switcher.setAttribute('aria-label', label);
    switcher.title = label;
  }

  switcher.addEventListener('click', function(event) {
    event.preventDefault();
    var current = storedLanguage() || document.documentElement.lang;
    var target = current === 'zh' ? 'en' : 'zh';
    try {
      localStorage.setItem('site-language', target);
    } catch (error) {
      // The current interaction still applies even if storage is unavailable.
    }
    var alternate = alternateFor(target);
    if (alternate && alternate.href !== window.location.href) {
      window.location.assign(alternate.href);
    } else {
      apply(target);
    }
  });

  document.querySelectorAll('[data-set-site-language]').forEach(function(link) {
    link.addEventListener('click', function() {
      var target = link.getAttribute('data-set-site-language');
      if (target !== 'en' && target !== 'zh') return;
      try {
        localStorage.setItem('site-language', target);
      } catch (error) {
        // The destination still opens when storage is unavailable.
      }
    });
  });

  apply(storedLanguage() || document.documentElement.lang);
})();
