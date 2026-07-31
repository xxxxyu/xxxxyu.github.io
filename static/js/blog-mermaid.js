document.addEventListener('DOMContentLoaded', function() {
  var explicitTheme = document.documentElement.getAttribute('data-theme');
  var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  var isDark = explicitTheme === 'dark' || (explicitTheme !== 'light' && systemDark);
  mermaid.initialize({ startOnLoad: true, theme: isDark ? 'dark' : 'default' });
});
