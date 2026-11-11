(function() {
  const videos = Array.from(document.querySelectorAll('video[data-autoplay]'));
  if (!videos.length || !('IntersectionObserver' in window)) return;

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const connection = navigator.connection;
  const visible = new Set();
  const userPaused = new WeakSet();
  const pausedByPage = new WeakSet();
  let assetsReady = false;

  function update(video) {
    if (!assetsReady) return;
    if (!visible.has(video) || document.hidden || motion.matches || (connection && connection.saveData)) {
      if (!video.paused) {
        // pause events are queued: visibility may change again before delivery.
        pausedByPage.add(video);
        video.pause();
      }
    } else if (video.paused && !userPaused.has(video)) {
      video.play().catch(function() { /* Native controls remain available. */ });
    }
  }

  function updateAll() { videos.forEach(update); }

  videos.forEach(function(video) {
    video.addEventListener('pause', function() {
      if (!pausedByPage.delete(video)) userPaused.add(video);
    });
    video.addEventListener('play', function() { userPaused.delete(video); });
  });

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting && entry.intersectionRatio >= 0.1) visible.add(entry.target);
      else visible.delete(entry.target);
      update(entry.target);
    });
  }, { threshold: 0.1 });
  videos.forEach(function(video) { observer.observe(video); });
  document.addEventListener('visibilitychange', updateAll);
  motion.addEventListener('change', updateAll);
  if (connection && connection.addEventListener) connection.addEventListener('change', updateAll);

  // Native controls work immediately; only automatic playback waits for assets.
  new Promise(function(resolve) {
    if (document.readyState === 'complete') resolve();
    else window.addEventListener('load', resolve, { once: true });
  }).then(function() {
    return document.fonts ? document.fonts.ready : undefined;
  }).then(function() {
    assetsReady = true;
    updateAll();
  });
})();
