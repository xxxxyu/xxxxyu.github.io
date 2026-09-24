/* Authored SVG layout with progressive disclosure and direct links. */
(function () {
  'use strict';
  const root = document.querySelector('[data-research]');
  if (!root) return;
  const zh = root.lang === 'zh';
  const cards = Array.from(root.querySelectorAll('[data-work]'));
  const nodes = Array.from(root.querySelectorAll('[data-node]'));
  const byId = new Map(cards.map(card => [card.dataset.work, card]));
  const nodeById = new Map(nodes.map(node => [node.dataset.node, node]));
  const map = root.querySelector('.research-map');
  const list = root.querySelector('.research-list');
  const preview = root.querySelector('.research-preview');
  const edges = root.querySelector('[data-edges]');
  const hullGroup = root.querySelector('[data-hulls]');
  const search = root.querySelector('[data-search]');
  const role = root.querySelector('select[data-role]');
  const kind = root.querySelector('select[data-kind]');
  const mobile = window.matchMedia('(max-width: 800px)');
  let view = 'map';
  let floating = true;
  let topic = 'all';
  let visible = new Set(byId.keys());
  let previewId = null;
  let layout = 'dynamic';
  let simulationFrame = null;
  let closeTimer;
  let openTimer;
  let edgeSignature = '';

  function convexHull(points) {
    const sorted = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    if (sorted.length < 3) return sorted;
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [];
    sorted.forEach(point => {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
      lower.push(point);
    });
    const upper = [];
    sorted.slice().reverse().forEach(point => {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
      upper.push(point);
    });
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function hullPath(points) {
    if (points.length < 3) return '';
    const expanded = convexHull(points.flatMap(point => Array.from({ length: 8 }, (_, i) => ({
      x: point.x + Math.cos(i * Math.PI / 4) * 20,
      y: point.y + Math.sin(i * Math.PI / 4) * 20
    }))));
    if (expanded.length < 3) return '';
    const padded = expanded;
    const start = { x: (padded[0].x + padded[padded.length - 1].x) / 2, y: (padded[0].y + padded[padded.length - 1].y) / 2 };
    let d = `M${start.x},${start.y}`;
    padded.forEach((point, index) => {
      const next = padded[(index + 1) % padded.length];
      d += ` Q${point.x},${point.y} ${(point.x + next.x) / 2},${(point.y + next.y) / 2}`;
    });
    return `${d} Z`;
  }

  function updateHulls() {
    if (!hullGroup) return;
    ['systems', 'embodied', 'agents'].forEach(topicName => {
      const path = hullGroup.querySelector(`[data-hull="${topicName}"]`);
      const points = nodes.filter(node => visible.has(node.dataset.node) && node.dataset.topics?.split(' ').includes(topicName))
        .map(node => ({ x: +node.dataset.x, y: +node.dataset.y }));
      const d = hullPath(points);
      path.toggleAttribute('hidden', !d);
      if (d) path.setAttribute('d', d);
    });
  }

  // The taxonomy has a small amount of hierarchy: the broad inference theme
  // should connect to its concrete inference terms (and to Personal Agents).
  const inferenceTerms = new Set(['inference', 'memory-management', 'kv-cache', 'quantization', 'low-bit-inference', 'kernel-optimization', 'matrix-multiplication', 'scheduling', 'batching', 'adaptive-inference', 'inference-runtimes', 'heterogeneous-computing']);
  function termsOverlap(a, b) {
    const left = a.dataset.terms.split(' ').filter(Boolean);
    const right = b.dataset.terms.split(' ').filter(Boolean);
    return left.filter(term => right.includes(term) || (term === 'inference' && right.some(item => inferenceTerms.has(item))) || (right.includes('inference') && inferenceTerms.has(term)));
  }

  function neighbours(id) {
    const source = byId.get(id);
    const a = nodeById.get(id);
    if (!source || !a) return [];
    return cards.filter(card => card !== source && visible.has(card.dataset.work))
      .map(card => {
        const sharedTerms = termsOverlap(source, card);
        const b = nodeById.get(card.dataset.work);
        return { id: card.dataset.work, shared: sharedTerms.length, distance: Math.hypot(+a.dataset.x - +b.dataset.x, +a.dataset.y - +b.dataset.y) };
      })
      .filter(item => item.shared > 0)
      .sort((a, b) => b.shared - a.shared || a.distance - b.distance)
      .slice(0, 3);
  }

  function updateEdgeGeometry() {
    edges.querySelectorAll('path[data-source]').forEach(path => {
      const a = nodeById.get(path.dataset.source), b = nodeById.get(path.dataset.target);
      if (!a || !b) return;
      const ax = +a.dataset.x, ay = +a.dataset.y, bx = +b.dataset.x, by = +b.dataset.y;
      path.setAttribute('d', `M${ax},${ay} Q${(ax + bx) / 2 + 15},${(ay + by) / 2 - 12} ${bx},${by}`);
    });
  }

  function drawConnections(id) {
    const related = id ? neighbours(id) : defaultConnections();
    const signature = `${id || 'default'}:${related.map(item => `${item.source || id}-${item.id}`).join('|')}`;
    if (signature !== edgeSignature) {
      edges.replaceChildren();
      if (!id) related.forEach(item => addEdge(nodeById.get(item.source), nodeById.get(item.id), false));
      else related.forEach(other => addEdge(nodeById.get(id), nodeById.get(other.id), true));
      edgeSignature = signature;
    }
    nodes.forEach(node => {
      node.classList.toggle('is-previewed', node.dataset.node === previewId);
      node.classList.toggle('is-muted', Boolean(previewId) && node.dataset.node !== id && !related.some(item => item.id === node.dataset.node));
    });
    updateEdgeGeometry();
    updateHulls();
  }

  function defaultConnections() {
    const defaultPairs = [
      ['doctoral-overview', 'flexnn'], ['doctoral-overview', 'vec-lut'],
      ['doctoral-overview', 'dwi'], ['doctoral-overview', 'oxygen'],
      ['oxygen', 'embodied-cpp'], ['oxygen', 'cosmos-lite'],
      ['actprobe', 'zetta'], ['zetta', 'embodiskill']
    ];
    const curated = defaultPairs.map(([source, id]) => ({ source, id, shared: termsOverlap(byId.get(source), byId.get(id)).length }))
      .filter(item => item.shared && visible.has(item.source) && visible.has(item.id));
    if (curated.length) return curated;
    const result = [];
    for (let i = 0; i < cards.length; i += 1) {
      for (let j = i + 1; j < cards.length; j += 1) {
        if (!visible.has(cards[i].dataset.work) || !visible.has(cards[j].dataset.work)) continue;
        const shared = termsOverlap(cards[i], cards[j]).length;
        if (shared) {
          const a = nodeById.get(cards[i].dataset.work), b = nodeById.get(cards[j].dataset.work);
          result.push({ source: cards[i].dataset.work, id: cards[j].dataset.work, shared, distance: Math.hypot(+a.dataset.x - +b.dataset.x, +a.dataset.y - +b.dataset.y) });
        }
      }
    }
    return result.sort((a, b) => b.shared - a.shared || a.distance - b.distance).slice(0, 11);
  }

  function addEdge(a, b, active) {
    if (!a || !b) return;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const ax = +a.dataset.x, ay = +a.dataset.y, bx = +b.dataset.x, by = +b.dataset.y;
    path.setAttribute('d', `M${ax},${ay} Q${(ax + bx) / 2 + 15},${(ay + by) / 2 - 12} ${bx},${by}`);
    path.dataset.source = a.dataset.node; path.dataset.target = b.dataset.node;
    path.setAttribute('class', active ? 'research-edge research-edge--active' : 'research-edge');
    edges.append(path);
  }

  function previewCard(id) {
    const card = byId.get(id);
    const node = nodeById.get(id);
    const box = document.createElement('div');
    box.className = 'research-preview__card';
    const title = document.createElement('strong');
    title.textContent = card.dataset.label;
    const meta = document.createElement('span');
    meta.className = 'research-preview__meta';
    meta.textContent = card.querySelector('.research-work__role')?.textContent.trim() || '';
    const terms = document.createElement('span');
    terms.className = 'research-preview__terms';
    terms.textContent = node.dataset.termLabels || 'Research work';
    const summary = document.createElement('span');
    summary.className = 'research-preview__summary';
    summary.textContent = card.querySelector('.research-work__summary')?.textContent.trim() || '';
    const hint = document.createElement('span');
    hint.className = 'research-preview__hint';
    hint.textContent = zh ? '点击打开内容 ↗' : 'Click to open ↗';
    box.append(title, meta, terms, summary, hint);
    return box;
  }

  function hidePreview() {
    clearTimeout(openTimer);
    clearTimeout(closeTimer);
    preview.hidden = true;
    previewId = null;
    drawConnections(null);
  }

  function showPreview(id, node) {
    clearTimeout(closeTimer);
    previewId = id;
    preview.replaceChildren(previewCard(id));
    preview.hidden = false;
    const box = node.getBoundingClientRect();
    const width = preview.offsetWidth;
    const height = preview.offsetHeight;
    let left = box.right + 10;
    if (left + width > window.innerWidth - 12) left = box.left - width - 10;
    preview.style.left = Math.max(12, Math.min(left, window.innerWidth - width - 12)) + 'px';
    preview.style.top = Math.max(12, Math.min(box.top - 20, window.innerHeight - height - 12)) + 'px';
    drawConnections(id);
  }

  function scheduleClose() {
    clearTimeout(openTimer);
    closeTimer = setTimeout(hidePreview, 220);
  }

  function setView(next) {
    view = next;
    root.dataset.viewMode = next;
    hidePreview();
    map.hidden = view !== 'map';
    list.hidden = true;
  }

  function setLayout(next) {
    layout = next;
    root.dataset.layout = next;
    root.querySelector('.research-map__svg').setAttribute('viewBox', '0 0 820 400');
    if (next === 'dynamic') startSimulation();
    else stopSimulation(true);
  }

  function stopSimulation(reset) {
    if (simulationFrame) cancelAnimationFrame(simulationFrame);
    simulationFrame = null;
    if (!reset) return;
    nodes.forEach(node => {
      node.dataset.x = node.dataset.authoredX;
      node.dataset.y = node.dataset.authoredY;
      node.setAttribute('transform', `translate(${node.dataset.x} ${node.dataset.y})`);
    });
    drawConnections(previewId);
  }

  // Visual physics, not literal gravitational mass. Contribution changes only
  // centrality and collision mobility; taxonomy alone determines link strength.
  const physics = Object.freeze({
    width: 820, height: 400, cx: 410, cy: 180,
    centerX: .05, centerY: .06, leadershipBoost: 4.5,
    repulsion: -135, linkLength: 112, linkStrength: .085,
    collisionRadius: 36, labelGap: 24, rowGap: 88,
    edgeBand: 65, edgeStrength: .12, paddingX: 14, top: 45, bottom: 320,
    cooling: .028, stopAlpha: .018, damping: .4, floatAmplitude: 5
  });

  function startSimulation() {
    stopSimulation(false);
    const active = nodes.filter(n => visible.has(n.dataset.node));
    const links = [];
    for (let i = 0; i < active.length; i += 1) for (let j = i + 1; j < active.length; j += 1) {
      const a = active[i], b = active[j];
      const shared = Math.min(termsOverlap(a, b).length, termsOverlap(b, a).length);
      if (shared) links.push({ source: a.dataset.node, target: b.dataset.node, shared });
    }
    const simNodes = active.map(node => {
      const contribution = Math.max(0, Math.min(1, Number(node.dataset.weight) || .2));
      const lead = node.classList.contains('research-node--author') || node.classList.contains('research-node--section');
      const importance = lead ? Math.max(.85, contribution) : contribution;
      return {
      id: node.dataset.node, node, importance, mobility: 1 / (1 + importance),
      half: Math.max(...[...node.querySelectorAll('text')].map(el => el.getComputedTextLength()), 32) / 2 + 10,
      // Begin as a compact cloud. Leads start slightly nearer its core, while
      // still receiving a random angle and radius before the force pass opens it.
      ...(() => { const angle = Math.random() * Math.PI * 2; const radius = lead ? Math.random() * 12 : 8 + Math.random() * 18; return { x: physics.cx + Math.cos(angle) * radius, y: physics.cy + Math.sin(angle) * radius }; })(),
      vx: (Math.random() - .5) * 2.5, vy: (Math.random() - .5) * 2.5
      };
    });
    const simulation = d3.forceSimulation(simNodes).stop()
      .alpha(1).alphaDecay(physics.cooling).velocityDecay(physics.damping)
      .force('links', d3.forceLink(links).id(n => n.id)
        .distance(l => physics.linkLength / Math.sqrt(l.shared))
        .strength(l => Math.min(.35, physics.linkStrength * l.shared)))
      .force('charge', d3.forceManyBody().strength(physics.repulsion))
      .force('x', d3.forceX(physics.cx).strength(n => physics.centerX * (1 + physics.leadershipBoost * n.importance ** 2)))
      .force('y', d3.forceY(physics.cy).strength(n => physics.centerY * (1 + physics.leadershipBoost * n.importance ** 2)))
      // Translate the cloud gently toward its center, preserving relative geometry.
      .force('center', d3.forceCenter(physics.cx, physics.cy).strength(.025))
      .force('edges', alpha => {
        simNodes.forEach(n => {
          const left = n.half + physics.paddingX, right = physics.width - left;
          const push = distance => Math.max(0, physics.edgeBand - distance) ** 2 / physics.edgeBand;
          n.vx += (push(n.x - left) - push(right - n.x)) * physics.edgeStrength * alpha;
          n.vy += (push(n.y - physics.top) - push(physics.bottom - n.y)) * physics.edgeStrength * alpha;
        });
      })
      .force('collision', d3.forceCollide(physics.collisionRadius).iterations(3));

    function constrain() {
      for (let pass = 0; pass < 24; pass++) {
        for (let i = 0; i < simNodes.length; i++) for (let j = i + 1; j < simNodes.length; j++) {
          const a = simNodes[i], b = simNodes[j], dx = b.x - a.x, dy = b.y - a.y;
          const ox = a.half + b.half + physics.labelGap - Math.abs(dx), oy = physics.rowGap + 18 - Math.abs(dy);
          if (ox > 0 && oy > 0) {
            const share = a.mobility / (a.mobility + b.mobility);
            if (ox < oy) { const push = (dx >= 0 ? 1 : -1) * ox; a.x -= push * share; b.x += push * (1 - share); }
            else { const push = (dy >= 0 ? 1 : -1) * oy; a.y -= push * share; b.y += push * (1 - share); }
          }
        }
      simNodes.forEach(n => {
        n.x = Math.max(n.half + physics.paddingX, Math.min(physics.width - physics.paddingX - n.half, n.x));
        n.y = Math.max(physics.top, Math.min(physics.bottom, n.y));
      });
      }

    }

    const svg = root.querySelector('.research-map__svg');
    svg.setAttribute('viewBox', '0 0 820 400');
    svg.style.minWidth = active.length > 3 ? '660px' : '0';
    let phase = Math.random() * Math.PI * 2;
    let last = 0;
    const frame = time => {
      simulationFrame = null;
      if (layout !== 'dynamic' || view !== 'map') return;
      const elapsed = last ? Math.min(50, time - last) : 0; last = time;
      // Motion is the explicitly selected presentation for this landscape.
      const reduced = false;
      if (!previewId) {
        // A continuously running low-amplitude drift gives the settled graph
        // the breathing quality of the previous landscape version.
        if (floating && !reduced) phase += elapsed / 1000;
        if (simulation.alpha() > physics.stopAlpha && !reduced) { simulation.tick(); constrain(); }
      }
      simNodes.forEach((n, i) => {
        const motion = floating && !reduced ? physics.floatAmplitude : 0;
        const x = n.x + motion * Math.sin(phase * .72 + i * 1.7);
        const y = n.y + motion * .72 * Math.cos(phase * .61 + i * 1.3);
        n.node.dataset.x = x; n.node.dataset.y = y;
        n.node.setAttribute('transform', `translate(${x} ${y})`);
      });
      if (hullGroup) hullGroup.style.opacity = '1';
      drawConnections(previewId);
      if (!reduced && (floating || simulation.alpha() > physics.stopAlpha)) simulationFrame = requestAnimationFrame(frame);
    };
    frame(performance.now());
  }

  function filter() {
    hidePreview();
    const query = search.value.trim().toLocaleLowerCase();
    visible = new Set();
    cards.forEach(card => {
      const matches = (topic === 'all' || card.dataset.topics.split(' ').includes(topic)) &&
        (role.value === 'all' || role.value === card.dataset.role || (role.value === 'author' && card.dataset.role === 'section')) &&
        (kind.value === 'all' || kind.value === card.dataset.kind) &&
        (!query || card.textContent.toLocaleLowerCase().includes(query));
      card.hidden = !matches;
      const node = nodeById.get(card.dataset.work);
      node.style.display = matches ? '' : 'none';
      if (matches) visible.add(card.dataset.work);
    });
    root.querySelector('.research-empty').hidden = visible.size !== 0;
    map.hidden = view !== 'map' || visible.size === 0;
    root.querySelectorAll('[data-topic]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.topic === topic)));
    drawConnections(null);
    if (root.dataset.layout && visible.size) startSimulation();
    else stopSimulation(false);
  }

  function reset() {
    topic = 'all'; search.value = ''; role.value = 'all'; kind.value = 'all'; filter();
  }

  nodes.forEach(node => {
    node.dataset.authoredX = node.dataset.x;
    node.dataset.authoredY = node.dataset.y;
    node.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'mouse' || mobile.matches) return;
      clearTimeout(openTimer); clearTimeout(closeTimer);
      openTimer = setTimeout(() => showPreview(node.dataset.node, node), 140);
    });
    node.addEventListener('pointerleave', scheduleClose);
    node.addEventListener('focus', () => { if (!mobile.matches) showPreview(node.dataset.node, node); });
    node.addEventListener('blur', scheduleClose);
    node.addEventListener('click', () => { hidePreview(); setTimeout(() => node.blur(), 0); });
  });
  preview.addEventListener('pointerenter', () => clearTimeout(closeTimer));
  preview.addEventListener('pointerleave', scheduleClose);
  preview.addEventListener('focusin', () => clearTimeout(closeTimer));
  preview.addEventListener('focusout', event => { if (!preview.contains(event.relatedTarget)) scheduleClose(); });
  root.addEventListener('keydown', event => { if (event.key === 'Escape') hidePreview(); });
  window.addEventListener('scroll', hidePreview, { passive: true });
  window.addEventListener('resize', hidePreview, { passive: true });
  root.querySelector('.research-map__scroll').addEventListener('scroll', hidePreview, { passive: true });
  root.querySelectorAll('[data-topic]').forEach(button => button.addEventListener('click', () => { topic = button.dataset.topic; filter(); }));
  search.addEventListener('input', filter); role.addEventListener('change', filter); kind.addEventListener('change', filter);
  root.querySelector('[data-reset]').addEventListener('click', reset);
  root.addEventListener('click', event => {
    const link = event.target.closest('[data-locate]');
    if (!link) return;
    event.preventDefault();
    const id = link.dataset.locate;
    if (!visible.has(id)) reset();
    setView('map');
    nodeById.get(id).scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' });
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopSimulation(false); else if (layout === 'dynamic' && view === 'map') startSimulation(); });

  root.classList.add('is-enhanced');
  root.querySelector('.research-controls').hidden = false;
  filter();
  setView('map');
  setLayout('dynamic');
  document.fonts.ready.then(() => { if (visible.size) startSimulation(); });
})();
