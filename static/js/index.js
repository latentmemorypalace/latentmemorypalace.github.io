document.addEventListener('DOMContentLoaded', function () {
  initializeBulmaComponents();
  initReasoningCarousels();
  initAdaptiveViz();
  setupLazyLoading();
});

// Adaptive reasoning via compression — combined 3-pane interactive.
//   top-left:    latent trellis (root + 4 layers, each 2 tokens + eos)
//   bottom-left: decaying σ schedule, x-aligned with the trellis layers
//   right:       ring-shaped data covered by 2^(k-1) Gaussians at reasoning length k
// A continuous position p ∈ [1,4] drives all three panes. It autoplays (ping-pong sweep
// with dwells at each integer step); dragging on the σ pane scrubs p directly, and
// autoplay resumes a moment after release.
function initAdaptiveViz() {
  const svg = document.getElementById('adaptive-viz');
  if (!svg) return;
  const NS = 'http://www.w3.org/2000/svg';
  const SERIF = 'Georgia, "Times New Roman", serif';
  const C1 = [106, 79, 180], C2 = [227, 152, 174];                  // purple -> pink, per step
  const col = t => { t = Math.max(0, Math.min(1, t)); return 'rgb(' + C1.map((v, i) => Math.round(v + (C2[i] - v) * t)).join(',') + ')'; };
  const colF = p => col((p - 1) / 3);
  const rgbOf = t => { t = Math.max(0, Math.min(1, t)); return C1.map((v, i) => v + (C2[i] - v) * t); };
  const mix = (a, b, t) => 'rgb(' + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',') + ')';
  const WHITE = [255, 255, 255], EDGE_GRAY = [225, 225, 225];
  const dark = (p, f) => { const t = Math.max(0, Math.min(1, (p - 1) / 3)); return 'rgb(' + C1.map((v, i) => Math.round((v + (C2[i] - v) * t) * f)).join(',') + ')'; };
  const clamp01 = v => Math.max(0, Math.min(1, v));
  const easeIO = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const make = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };

  // --- shared x positions: trellis layers and σ steps are the same columns ---
  const XS = k => 125 + (k - 1) * 85;                               // steps 1..4 (fractional ok); root at 40 = one step before layer 1
  const ROWS = [60, 108, 156];                                      // token, token, eos
  const NODE_R = 11;
  // σ pane
  const SY1 = 252, SY0 = 358;                                       // y at σ=1 / σ=0
  const yS = v => SY0 - v * (SY0 - SY1);
  // true geometric decay σ_k = σ_max·(σ_min/σ_max)^((k-1)/(K-1)) with σ_max=1, σ_min=0.01
  const sigAt = p => Math.pow(0.01, (p - 1) / 3);
  const SIG = [1, 2, 3, 4].map(sigAt);
  // 3D data pane: the ring lies on a tilted ground plane; the mixture density rises along z
  // as a ridgeline surface. Level s has 2^(s-1) Gaussians on nested half-arcs — fewer steps =
  // one broad flat bump, more steps = many sharp tall bumps (higher density).
  const CX = 660, CY = 238, RING = 110, BETA = -Math.PI / 2;
  const SD = [61, 47, 32, 21];                                      // σ per level (data units)
  const HGT = [46, 62, 86, 118];                                    // peak height per level (px)
  const DIST = [0, 70, 99, 107];                                    // component centroid distance
  const LVL = [1, 2, 4, 8].map((m, li) => Array.from({ length: m }, (_, i) => {
    const th = BETA + (i + 0.5) * 2 * Math.PI / m;
    return { x: DIST[li] * Math.cos(th), y: DIST[li] * Math.sin(th), s: SD[li], h: HGT[li] };
  }));
  const EXT = 150, NX = 46, NY = 26;                                // grid extent + mesh resolution
  const RH = 2 * EXT / NY;                                          // row pitch (data units)
  // Classic xyz corner view: the x-y plane is yawed 45° to the screen (x recedes down-right,
  // y up-right), z vertical. C1/C2 fold in cos45, overall scale and elevation.
  const KX = 0.601, KY = 0.330, ZS = 0.8;
  const px3 = (x, y) => CX + (x - y) * KX;
  const py3 = (x, y, z) => CY + (x + y) * KY - z * ZS;

  // --- static scenery ---
  const defs = make('defs', {});
  const grad = make('linearGradient', { id: 'av-grad', gradientUnits: 'userSpaceOnUse', x1: XS(1), y1: 0, x2: XS(4), y2: 0 });
  grad.appendChild(make('stop', { offset: '0%', 'stop-color': col(0) }));
  grad.appendChild(make('stop', { offset: '100%', 'stop-color': col(1) }));
  defs.appendChild(grad); svg.appendChild(defs);

  // pane titles match the animated method figures: 15px serif, gray, title case
  const label = (x, y, s) => make('text', { x: x, y: y, 'font-size': 15, fill: '#9A9A9A', 'font-family': SERIF }, s);
  svg.appendChild(label(24, 26, 'Latent Reasoning Trace'));
  svg.appendChild(label(24, 226, 'Decoder Standard Deviation'));
  svg.appendChild(label(430, 26, 'Action Distribution'));

  // playhead line spanning both left panes (behind everything else)
  const vline = make('line', { y1: 38, y2: SY0, stroke: '#c9c9c9', 'stroke-width': 1.2, 'stroke-dasharray': '4 4' });
  svg.appendChild(vline);

  // trellis edges (behind the nodes): tokens of layer j -> tokens/eos of layer j+1
  const edges = [];   // {el, src, tgt, isEos}
  const addEdge = (x1, y1, x2, y2, src, tgt, isEos) => {
    const l = make('line', { x1: x1, y1: y1, x2: x2, y2: y2, stroke: mix(EDGE_GRAY, EDGE_GRAY, 0), 'stroke-width': 1.3 });
    svg.appendChild(l); edges.push({ el: l, src: src, tgt: tgt, isEos: isEos });
  };
  for (let j = 1; j <= 3; j++) {
    [ROWS[0], ROWS[1]].forEach(ys => {
      [ROWS[0], ROWS[1]].forEach(yt => addEdge(XS(j), ys, XS(j + 1), yt, j, j + 1, false));
      addEdge(XS(j), ys, XS(j + 1), ROWS[2], j, j + 1, true);
    });
  }
  // observation -> layer-1 entry edges (fan out from behind the obs tile; tgt=1 makes them
  // light up with the sweep exactly like the layer-1 nodes)
  [ROWS[0], ROWS[1]].forEach(yt => addEdge(40, 108, XS(1), yt, 0, 1, false));
  addEdge(40, 108, XS(1), ROWS[2], 0, 1, true);   // obs -> immediate eos (zero-step trace)

  // trellis: 4 layers x (2 tokens + eos); nodes are solid-filled so edges pass behind them
  svg.appendChild(make('text', { x: 101, y: ROWS[2] + 4, 'text-anchor': 'end', 'font-size': 13, fill: '#999', 'font-family': SERIF }, 'eos'));
  const nodes = [];   // {el, layer, isEos}
  for (let j = 1; j <= 4; j++) {
    ROWS.forEach((y, ri) => {
      const c = make('circle', { cx: XS(j), cy: y, r: NODE_R, fill: '#fff', stroke: '#c4c4c4', 'stroke-width': 2 });
      svg.appendChild(c);
      nodes.push({ el: c, layer: j, isEos: ri === 2 });
    });
  }

  // observation tile at the trellis root — a real camera thumbnail, matching the method
  // figures' convention (observations are real images, latents are symbols)
  const obsClip = make('clipPath', { id: 'av-obsclip' });
  obsClip.appendChild(make('rect', { x: 18, y: 86, width: 44, height: 44, rx: 8 }));
  defs.appendChild(obsClip);
  svg.appendChild(make('image', {
    href: './static/figanim/obs_full.png', x: 18, y: 86, width: 44, height: 44,
    'clip-path': 'url(#av-obsclip)', preserveAspectRatio: 'xMidYMid slice'
  }));
  svg.appendChild(make('rect', { x: 18, y: 86, width: 44, height: 44, rx: 8, fill: 'none', stroke: '#D6D6D6', 'stroke-width': 1.2 }));

  // σ pane: axes, curve, step markers, playhead
  svg.appendChild(make('line', { x1: 109, y1: SY1 - 8, x2: 109, y2: SY0, stroke: '#cfcfcf', 'stroke-width': 1.4 }));
  svg.appendChild(make('line', { x1: 109, y1: SY0, x2: 392, y2: SY0, stroke: '#cfcfcf', 'stroke-width': 1.4 }));
  for (let k = 1; k <= 4; k++) {
    svg.appendChild(make('text', { x: XS(k), y: SY0 + 18, 'text-anchor': 'middle', 'font-size': 12, fill: '#999', 'font-family': SERIF }, String(k)));
  }
  svg.appendChild(make('text', { x: (XS(1) + XS(4)) / 2, y: SY0 + 36, 'text-anchor': 'middle', 'font-size': 13, fill: '#999', 'font-family': SERIF }, 'latent step k'));
  svg.appendChild(make('text', { x: 91, y: (SY1 + SY0) / 2, 'text-anchor': 'middle', 'font-size': 13, fill: '#999', 'font-family': SERIF, 'font-style': 'italic', transform: 'rotate(-90 91 ' + (SY1 + SY0) / 2 + ')' }, 'σ(k)'));
  let pts = '';
  for (let p = 1; p <= 4.001; p += 0.025) pts += XS(Math.min(p, 4)) + ',' + yS(sigAt(Math.min(p, 4))) + ' ';   // fine sampling: the decay is steep near step 1
  svg.appendChild(make('polyline', { points: pts.trim(), fill: 'none', stroke: 'url(#av-grad)', 'stroke-width': 2.5, 'stroke-linecap': 'round' }));
  const sigMarkers = [];
  for (let k = 1; k <= 4; k++) {
    const c = make('circle', { cx: XS(k), cy: yS(SIG[k - 1]), r: 4.5, fill: col((k - 1) / 3), stroke: '#fff', 'stroke-width': 1.5 });
    svg.appendChild(c); sigMarkers.push(c);
  }
  const playhead = make('circle', { r: 8, fill: '#fff', 'stroke-width': 3.5 });
  svg.appendChild(playhead);

  // Seeded ring points on the ground plane, interleaved row-by-row with the surface so
  // nearer ridges occlude farther points (points under a bump hide behind its front face).
  let seed = 7;
  const rnd = () => { seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const ringPts = [];
  for (let i = 0; i < 150; i++) {
    const a = rnd() * 2 * Math.PI;
    const g = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const r = RING + g * 10;
    ringPts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
  }
  // Ground scatter first — it shows through the translucent blanket above it.
  ringPts.forEach(q => {
    svg.appendChild(make('circle', { cx: px3(q.x, q.y), cy: py3(q.x, q.y, 0), r: 1.8, fill: '#555', opacity: 0.5 }));
  });
  // Transparent blanket: translucent under-fills give the sheet its body (overlaps darken
  // where the surface is tall) + a light two-direction wireframe; the back stays visible.
  const ROWS3 = [];
  for (let j = 0; j < NY; j++) ROWS3.push(-EXT + (j + 0.5) * RH);
  const rowFills = ROWS3.map(() => { const e = make('path', { 'fill-opacity': 0.05, stroke: 'none' }); svg.appendChild(e); return e; });
  const rowLines = ROWS3.map((_, j) => { const e = make('path', { fill: 'none', 'stroke-width': 1.1, 'stroke-opacity': (0.3 + 0.3 * j / (NY - 1)).toFixed(2) }); svg.appendChild(e); return e; });
  const COLS3 = [];
  for (let i = 0; i <= NX; i += 2) COLS3.push(i);
  const colLines = COLS3.map(() => { const e = make('path', { fill: 'none', 'stroke-width': 1.1, 'stroke-opacity': 0.28 }); svg.appendChild(e); return e; });
  // Classic xyz triad from the plane's left corner: x and y run along the ground edges,
  // z rises vertically with an arrow + label.
  const ox = px3(-EXT, EXT), oy = py3(-EXT, EXT, 0), zTop = oy - 132 * ZS;
  const axisLine = (x2, y2) => svg.appendChild(make('line', { x1: ox, y1: oy, x2: x2, y2: y2, stroke: '#b0b0b0', 'stroke-width': 1.2 }));
  axisLine(px3(EXT, EXT), py3(EXT, EXT, 0));      // x edge (down-right)
  axisLine(px3(-EXT, -EXT), py3(-EXT, -EXT, 0));  // y edge (up-right)
  axisLine(ox, zTop);                             // z (vertical)
  svg.appendChild(make('path', { d: 'M' + (ox - 3.5) + ' ' + (zTop + 7) + ' L' + ox + ' ' + zTop + ' L' + (ox + 3.5) + ' ' + (zTop + 7), fill: 'none', stroke: '#b0b0b0', 'stroke-width': 1.2 }));
  svg.appendChild(make('text', { x: ox - 8, y: (oy + zTop) / 2, 'text-anchor': 'middle', 'font-size': 13, fill: '#999', 'font-family': SERIF, transform: 'rotate(-90 ' + (ox - 8) + ' ' + (oy + zTop) / 2 + ')' }, 'density'));
  // "data" label painted ON the ground plane: baseline follows the x edge and glyph
  // verticals stay parallel to the receding y axis, via the plane's exact affine projection
  // (e_x = (KX,KY), glyph-up = e_{-y} = (KX,-KY); local y offset pushes it outside the sheet)
  {
    const ex = px3(EXT, EXT), ey = py3(EXT, EXT, 0);
    const mx = (ox + ex) / 2, my = (oy + ey) / 2;
    const L = Math.hypot(KX, KY), ux = KX / L, uy = KY / L;
    svg.appendChild(make('text', { x: 0, y: 17, 'text-anchor': 'middle', 'font-size': 13, fill: '#999',
      'font-family': SERIF,
      transform: 'matrix(' + [ux, uy, -ux, uy, mx, my].map(v => v.toFixed(4)).join(' ') + ')' }, 'action space'));
  }

  // --- render everything for a continuous position p ∈ [1,4] ---
  let lastP = NaN;
  function render(p) {
    p = Math.max(1, Math.min(4, p));
    if (p === lastP) return;   // the surface is ~10k exp() per paint; skip no-op frames
    lastP = p;
    vline.setAttribute('x1', XS(p)); vline.setAttribute('x2', XS(p));
    nodes.forEach(n => {
      const a = n.isEos ? clamp01(1 - Math.abs(p - n.layer)) : clamp01(p - n.layer);
      n.el.setAttribute('fill', mix(WHITE, rgbOf((n.layer - 1) / 3), a * (n.isEos ? 1 : 0.85)));
      n.el.setAttribute('stroke', a > 0.02 ? dark(n.layer, 0.72) : '#c4c4c4');
      n.el.setAttribute('stroke-opacity', (0.5 + 0.5 * a).toFixed(3));
    });
    edges.forEach(g => {
      const srcA = clamp01(p - g.src);
      const a = g.isEos ? Math.min(srcA, clamp01(1 - Math.abs(p - g.tgt))) : clamp01(p - g.tgt);
      g.el.setAttribute('stroke', mix(EDGE_GRAY, rgbOf((g.tgt - 1) / 3), a * 0.9));
      g.el.setAttribute('stroke-width', (1.3 + 0.7 * a).toFixed(2));
    });
    sigMarkers.forEach((m, i) => m.setAttribute('r', (4.5 + 4 * clamp01(1 - Math.abs(p - (i + 1)))).toFixed(2)));
    playhead.setAttribute('cx', XS(p)); playhead.setAttribute('cy', yS(sigAt(p)));
    playhead.setAttribute('stroke', colF(p));
    // Mixture components at position p: at integer p show level p; between p and p+1 each
    // parent splits into two children (position, σ and height all interpolate).
    const st4 = Math.min(4, Math.floor(p + 1e-9)), f = p - st4;
    let comps;
    if (f < 1e-4 || st4 >= 4) comps = LVL[st4 - 1];
    else {
      const e = easeIO(f);
      comps = LVL[st4].map((c, j) => {
        const par = LVL[st4 - 1][j >> 1];
        // mass conservation: two coincident children must SUM to the parent bump at the
        // moment of splitting, so each starts at half the parent's height
        return {
          x: par.x + (c.x - par.x) * e, y: par.y + (c.y - par.y) * e,
          s: par.s + (c.s - par.s) * e, h: par.h / 2 + (c.h - par.h / 2) * e
        };
      });
    }
    // Evaluate the density grid once (z = Σ h·exp(−d²/2σ²)), then draw the blanket from it.
    const cc = colF(p);
    const grid = [];
    for (let j = 0; j < NY; j++) {
      const row = new Float64Array(NX + 1);
      for (let i = 0; i <= NX; i++) {
        const x = -EXT + (2 * EXT * i) / NX;
        let z = 0;
        for (const c of comps) {
          const dx = x - c.x, dy = ROWS3[j] - c.y;
          z += c.h * Math.exp(-(dx * dx + dy * dy) / (2 * c.s * c.s));
        }
        row[i] = z;
      }
      grid.push(row);
    }
    for (let j = 0; j < NY; j++) {
      const ry = ROWS3[j];
      let d = '';
      for (let i = 0; i <= NX; i++) {
        const x = -EXT + (2 * EXT * i) / NX;
        d += (i ? ' L' : 'M') + px3(x, ry).toFixed(1) + ' ' + py3(x, ry, grid[j][i]).toFixed(1);
      }
      rowLines[j].setAttribute('d', d); rowLines[j].setAttribute('stroke', cc);
      rowFills[j].setAttribute('d', d + ' L' + px3(EXT, ry).toFixed(1) + ' ' + py3(EXT, ry, 0).toFixed(1) +
        ' L' + px3(-EXT, ry).toFixed(1) + ' ' + py3(-EXT, ry, 0).toFixed(1) + ' Z');
      rowFills[j].setAttribute('fill', cc);
    }
    COLS3.forEach((ci, k) => {
      const x = -EXT + (2 * EXT * ci) / NX;
      let d = '';
      for (let j = 0; j < NY; j++) d += (j ? ' L' : 'M') + px3(x, ROWS3[j]).toFixed(1) + ' ' + py3(x, ROWS3[j], grid[j][ci]).toFixed(1);
      colLines[k].setAttribute('d', d); colLines[k].setAttribute('stroke', cc);
    });
  }

  // --- timeline: autoplay ping-pong with dwells; drag scrubs; resume after release ---
  const st = { mode: 'dwell', p: 1, t0: null, from: 1, to: 1, dur: 850, resumeAt: 0 };
  function advance(now) {
    if (st.t0 === null) st.t0 = now;
    if (st.mode === 'dwell') {
      const dwell = (st.p <= 1 || st.p >= 4) ? 1600 : 1100;
      if (now - st.t0 >= dwell) {
        st.from = st.p;
        st.to = st.p >= 4 ? 1 : Math.round(st.p) + 1;   // 1→2→3→4, then wrap back to 1
        st.dur = st.to < st.from ? 1300 : 850;          // the wrap rewinds through all levels
        st.mode = 'move'; st.t0 = now;
      }
    } else if (st.mode === 'move') {
      const f = (now - st.t0) / st.dur;
      if (f >= 1) { st.p = st.to; st.mode = 'dwell'; st.t0 = now; }
      else st.p = st.from + (st.to - st.from) * f;    // linear sweep; the split itself is eased
    } else if (st.mode === 'resume') {
      if (now >= st.resumeAt) {
        st.from = st.p; st.to = Math.round(st.p);
        if (Math.abs(st.to - st.from) < 1e-3) { st.p = st.to; st.mode = 'dwell'; st.t0 = now; }
        else { st.dur = Math.max(250, 850 * Math.abs(st.to - st.from)); st.mode = 'move'; st.t0 = now; }
      }
    }                                                  // 'drag': p is driven by the pointer
    render(st.p);
  }

  // Play by default; the IntersectionObserver only PAUSES the loop while off-screen.
  // (Never gate the start on an observer callback: Chrome batches entries — reading a stale
  // one — and skips nonzero thresholds for elements it measured as zero-area, so a
  // start-only-on-intersect loop can simply never begin. Worst case here is the inverse:
  // a few frames render off-screen before the first observer callback pauses them.)
  let rafId = null, visible = true;
  const loop = ts => { rafId = null; advance(ts); if (visible) rafId = requestAnimationFrame(loop); };
  const start = () => { if (rafId === null) { st.t0 = null; rafId = requestAnimationFrame(loop); } };
  start();
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(es => {
      visible = es[es.length - 1].isIntersecting;   // last entry = most recent state
      if (visible) start();
    }, { threshold: 0 }).observe(svg);
  }

  // drag anywhere on the σ pane
  const overlay = make('rect', { x: 100, y: 234, width: 302, height: 162, fill: '#000', 'fill-opacity': 0, 'pointer-events': 'all' });
  overlay.style.cursor = 'ew-resize';
  svg.appendChild(overlay);
  const xToP = e => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const x = pt.matrixTransform(svg.getScreenCTM().inverse()).x;
    return Math.max(1, Math.min(4, 1 + (x - XS(1)) / (XS(2) - XS(1))));
  };
  const mv = e => { if (st.mode === 'drag') { e.preventDefault(); st.p = xToP(e); render(st.p); } };
  const up = () => {
    if (st.mode !== 'drag') return;
    st.mode = 'resume'; st.resumeAt = performance.now() + 2200;
    window.removeEventListener('pointermove', mv);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', up);
  };
  overlay.addEventListener('pointerdown', e => {
    e.preventDefault();
    st.mode = 'drag'; st.p = xToP(e); render(st.p);
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  });

  render(st.p);
  svg._viz = { setP: v => { st.mode = 'drag'; st.p = v; render(v); }, getP: () => st.p, getMode: () => st.mode, advance: advance, st: st };
}

// Grouped bar chart with cross-linked hover: hovering a method (bar or legend entry)
// highlights its bars in every group and its column in the linked results table, dimming
// the rest. METHODS: [{name, color, col}] where col = 1-based column in the linked table.
// GROUPS: [{name, vals}] with vals aligned to METHODS order.
function groupedBarPlot(rootId, tableSel, METHODS, GROUPS) {
  var root = document.getElementById(rootId);
  if (!root) return;
  var H = 250, hoverables = [], tbl = null;

  var row = document.createElement('div');
  row.style.cssText = 'display:flex; align-items:stretch;';
  var yTitle = document.createElement('div');
  yTitle.style.cssText = 'flex:none; width:22px; position:relative;';
  yTitle.innerHTML = '<span style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%) rotate(-90deg); white-space:nowrap; font-size:14px; color:#777;">Success Rate</span>';
  var areaWrap = document.createElement('div');
  areaWrap.style.cssText = 'flex:1; padding-left:34px;';
  var area = document.createElement('div');
  area.style.cssText = 'position:relative; height:' + H + 'px; border-left:1.4px solid #cfcfcf; border-bottom:1.4px solid #cfcfcf;';
  for (var v = 0; v <= 1.0001; v += 0.2) {
    var y = (v * 100).toFixed(1);
    if (v > 0.001) area.insertAdjacentHTML('beforeend',
      '<div style="position:absolute; left:0; right:0; bottom:' + y + '%; border-top:1px dashed #ececec;"></div>');
    area.insertAdjacentHTML('beforeend',
      '<div style="position:absolute; right:100%; margin-right:9px; bottom:' + y + '%; transform:translateY(50%); font-size:12px; color:#999;">' + v.toFixed(1) + '</div>');
  }
  var groupsRow = document.createElement('div');
  groupsRow.style.cssText = 'position:absolute; inset:0; display:flex; justify-content:space-around; align-items:flex-end; padding:0 14px;';
  var labelsRow = document.createElement('div');
  labelsRow.style.cssText = 'display:flex; justify-content:space-around; padding:6px 14px 0 0; margin-left:34px;';

  GROUPS.forEach(function (grp) {
    var g = document.createElement('div');
    g.style.cssText = 'display:flex; align-items:flex-end; gap:7px;';
    grp.vals.forEach(function (val, m) {
      var cell = document.createElement('div');
      cell.style.cssText = 'display:flex; flex-direction:column; align-items:center; justify-content:flex-end;';
      cell.setAttribute('data-m', m);
      cell.innerHTML =
        '<div style="font-size:13px; color:#555; margin-bottom:3px;">' + val.toFixed(2) + '</div>' +
        '<div class="gbp-bar" style="width:34px; height:' + (val * H) + 'px; background:' + METHODS[m].color + '; border-radius:3px 3px 0 0;"></div>';
      g.appendChild(cell);
      hoverables.push({ el: cell, m: m });
    });
    groupsRow.appendChild(g);
    labelsRow.insertAdjacentHTML('beforeend', '<div style="font-size:15px; color:#555;">' + grp.name + '</div>');
  });
  area.appendChild(groupsRow);
  areaWrap.appendChild(area);
  row.appendChild(yTitle); row.appendChild(areaWrap);
  root.appendChild(row);
  root.appendChild(labelsRow);

  var leg = document.createElement('div');
  leg.style.cssText = 'display:flex; justify-content:center; gap:22px; margin-top:14px; flex-wrap:wrap;';
  METHODS.forEach(function (meth, m) {
    var item = document.createElement('div');
    item.style.cssText = 'display:flex; align-items:center; gap:7px; font-size:13.5px; color:#555; cursor:default;';
    item.setAttribute('data-m', m);
    item.innerHTML = '<span style="width:15px; height:15px; border-radius:3px; background:' + meth.color + '; display:inline-block;"></span>' + meth.name;
    leg.appendChild(item);
    hoverables.push({ el: item, m: m });
  });
  root.appendChild(leg);

  function setHover(m) {
    hoverables.forEach(function (h) {
      var on = (m === null || h.m === m);
      h.el.style.opacity = on ? '1' : '0.25';
      var bar = h.el.querySelector('.gbp-bar');
      if (bar) bar.style.filter = (m !== null && h.m === m) ? 'brightness(1.08) saturate(1.15)' : '';
    });
    if (!tbl) {   // lazy: the table may be parsed after this plot is built
      tbl = document.querySelector(tableSel);
      if (tbl) tbl.querySelectorAll('td,th').forEach(function (c) { c.style.transition = 'background .18s ease, opacity .18s ease'; });
    }
    if (tbl) {
      var hotCol = (m === null) ? -1 : METHODS[m].col;
      tbl.querySelectorAll('tr').forEach(function (row) {
        var cells = row.children;
        for (var i = 1; i < cells.length; i++) {
          cells[i].style.background = (i === hotCol) ? '#f0ecf7' : '';   // same neutral shade for every method
          cells[i].style.opacity = (m !== null && i !== hotCol) ? '0.35' : '1';
        }
      });
    }
  }
  hoverables.forEach(function (h) {
    h.el.style.transition = 'opacity .18s ease';
    h.el.addEventListener('mouseenter', function () { setHover(h.m); });
    h.el.addEventListener('mouseleave', function () { setHover(null); });
  });
}

function showDroid(task, btn) {
  document.querySelectorAll('.droid-eval-container').forEach(function (el) { el.style.display = 'none'; });
  document.getElementById('droid-' + task).style.display = 'block';
  if (btn) {
    const toggle = btn.closest('.seg-toggle');
    const btns = Array.prototype.slice.call(toggle.querySelectorAll('.seg-btn'));
    const idx = btns.indexOf(btn);
    btns.forEach(function (b, i) { b.classList.toggle('is-active', i === idx); });
    const thumb = toggle.querySelector('.seg-thumb');
    if (thumb) thumb.style.transform = 'translateX(' + (idx * 100) + '%)';   // slide pill to the active segment
  }
}

// RoboMimic / LIBERO suite toggle for the tokenizer rollout grids (same pattern as showDroid);
// hidden grids pause their videos so only the visible suite plays.
function showTokSuite(suite, btn) {
  document.querySelectorAll('.tok-suite-grid').forEach(function (el) {
    const on = el.id === 'tok-' + suite;
    el.style.display = on ? '' : 'none';
    el.querySelectorAll('video').forEach(function (v) {
      if (on) { v.play().catch(function () { }); } else { v.pause(); }
    });
  });
  if (btn) {
    const toggle = btn.closest('.seg-toggle');
    const btns = Array.prototype.slice.call(toggle.querySelectorAll('.seg-btn'));
    const idx = btns.indexOf(btn);
    btns.forEach(function (b, i) { b.classList.toggle('is-active', i === idx); });
    const thumb = toggle.querySelector('.seg-thumb');
    if (thumb) thumb.style.transform = 'translateX(' + (idx * 100) + '%)';
  }
}

// Carousel for reasoning-trace videos: a wrap-around carousel that advances one video at a
// time (two visible). The track slides via a CSS transform transition (compositor-driven, no
// per-frame JS), with the clips playing throughout. Cloned head/tail slides make the loop
// seamless; at the seam the real slide's playback time is matched to the clone's so the swap is
// invisible. (A ~1px reflow of the live <video> textures at slide-end is a Chrome-only
// compositor quirk, absent in Safari, that we accept rather than freezing the clips.)
function initReasoningCarousels() {
  const GAP = 16;        // px gap between slides
  const DUR = 400;       // slide duration (ms)
  const TRANS = 'transform ' + (DUR / 1000) + 's cubic-bezier(0.4, 0, 0.2, 1)';
  // One video per row on mobile, two on desktop. matchMedia is the JS mirror of a CSS
  // media query: read .matches at build time, rebuild on its change event (rotation etc.).
  const mqMobile = window.matchMedia('(max-width: 768px)');   // Bulma's mobile breakpoint

  function buildCarousel(car) {
    const VISIBLE = mqMobile.matches ? 1 : 2;   // videos shown at once
    // Build the skeleton so the markup is just <div class="rt-carousel" data-...>.
    car.innerHTML =
      '<div class="rt-row">' +
      '<button class="rt-prev" aria-label="Previous">←</button>' +
      '<div class="rt-viewport"><div class="rt-track"></div></div>' +
      '<button class="rt-next" aria-label="Next">→</button>' +
      '</div><div class="rt-indicator"></div>';

    const base = car.dataset.base;
    const eps = parseInt(car.dataset.eps, 10);
    const task = car.dataset.task;          // one task per carousel; a sibling toggle switches them
    const videos = [];
    for (let i = 0; i < eps; i++) videos.push({ src: base + '/' + task + '/ep' + i + '.mp4' });

    const viewport = car.querySelector('.rt-viewport');
    const track = car.querySelector('.rt-track');
    const indicator = car.querySelector('.rt-indicator');
    const prev = car.querySelector('.rt-prev');
    const next = car.querySelector('.rt-next');

    const N = videos.length;
    const perVisible = Math.min(VISIBLE, N);
    const wrap = N > perVisible;
    const K = perVisible;   // clones on each side

    // Extended slide list: [last K clones] + [N real] + [first K clones].
    const ext = [];
    if (wrap) for (let i = N - K; i < N; i++) ext.push(videos[i]);
    for (let i = 0; i < N; i++) ext.push(videos[i]);
    if (wrap) for (let i = 0; i < K; i++) ext.push(videos[i]);
    const M = ext.length;
    let pos = wrap ? K : 0;          // left-edge slide index (real video 0 starts here)
    let animating = false, settleT = null, activated = false;

    track.style.gap = GAP + 'px';
    track.style.transition = TRANS;
    track.style.willChange = 'transform';

    const slides = ext.map(function (item) {
      const slide = document.createElement('div');
      slide.style.cssText = 'flex:0 0 auto;';   // width set in px by computeMetrics()
      const vid = document.createElement('video');
      vid.muted = true; vid.loop = true; vid.setAttribute('playsinline', '');
      // A fixed aspect-ratio reserves the slot's height so a clip loading mid-slide can't shift layout.
      vid.style.cssText = 'width:100%; border-radius:6px; display:block; background:#f4f4f4; aspect-ratio:1120/480;';
      slide.appendChild(vid);
      track.appendChild(slide);
      return { el: slide, vid: vid, src: item.src, loaded: false };
    });

    // Lock the slots to the clips' true aspect-ratio once the first one reports it.
    let arSet = false;
    function lockAR(v) {
      if (arSet || !v.videoWidth || !v.videoHeight) return;
      arSet = true;
      const ar = v.videoWidth + ' / ' + v.videoHeight;
      slides.forEach(function (s) { s.vid.style.aspectRatio = ar; });
    }

    // Integer slide widths -> integer transform offsets (one slide step = w + gap).
    let step = 0;
    function computeMetrics() {
      const w = Math.max(1, Math.floor((viewport.clientWidth - (perVisible - 1) * GAP) / perVisible));
      step = w + GAP;
      slides.forEach(function (s) { s.el.style.flexBasis = w + 'px'; });
    }
    function place(animate) {
      track.style.transition = animate ? TRANS : 'none';
      track.style.transform = 'translateX(' + (-pos * step) + 'px)';
      if (!animate) { void track.offsetHeight; track.style.transition = TRANS; }   // commit, then re-arm
    }

    function loadNear() {
      for (let i = pos - 1; i <= pos + perVisible; i++) {
        if (i < 0 || i >= M || slides[i].loaded) continue;
        const s = slides[i];
        s.vid.src = s.src; s.vid.load(); s.loaded = true;
        s.vid.addEventListener('loadedmetadata', function () { lockAR(this); }, { once: true });
      }
    }
    function playVisible() {
      slides.forEach(function (s, i) {
        const vis = (i >= pos && i < pos + perVisible);
        if (vis && s.loaded) s.vid.play().catch(function () { });
        else s.vid.pause();
      });
    }
    // Dots (one per episode), matching the teaser / More Analysis carousels.
    for (let i = 0; i < N; i++) {
      const d = document.createElement('div');
      d.style.cssText = 'width:8px; height:8px; border-radius:50%; background:#ccc; cursor:pointer; transition:background 0.2s;';
      d.addEventListener('click', function () { goTo(i); });
      indicator.appendChild(d);
    }
    function setIndicator() {
      const cur = wrap ? (((pos - K) % N) + N) % N : pos;
      Array.from(indicator.children).forEach(function (d, i) {
        d.style.background = i === cur ? '#6a4fb4' : '#ccc';
      });
    }

    function onSettle() {
      if (!animating) return;
      animating = false;
      if (settleT) { clearTimeout(settleT); settleT = null; }
      if (wrap && (pos > K + N - 1 || pos < K)) {       // landed on a clone -> jump to its real twin
        const old = pos;
        pos += (pos < K) ? N : -N;
        for (let i = 0; i < perVisible; i++) {          // keep frames continuous across the seam
          const a = slides[old + i], b = slides[pos + i];
          if (a && b && a.loaded && b.loaded) { try { b.vid.currentTime = a.vid.currentTime; } catch (e) { } }
        }
        place(false);
        loadNear();
      }
      playVisible();
    }

    function go(dir) {
      if (animating || !wrap) return;
      animating = true;
      pos += dir;
      place(true);
      loadNear(); playVisible(); setIndicator();
      settleT = setTimeout(onSettle, DUR + 140);   // fallback if transitionend is missed
    }
    function goTo(i) {                             // dot click: slide straight to episode i
      const target = wrap ? K + i : i;
      if (animating || target === pos) return;
      animating = true;
      pos = target;
      place(true);
      loadNear(); playVisible(); setIndicator();
      settleT = setTimeout(onSettle, DUR + 140);
    }

    track.addEventListener('transitionend', function (e) {
      if (e.target === track && e.propertyName === 'transform') onSettle();
    });
    prev.addEventListener('click', function () { go(-1); });
    next.addEventListener('click', function () { go(1); });
    car._onResize = function () { if (activated) { computeMetrics(); place(false); } };

    function activate() {
      computeMetrics(); place(false); loadNear(); playVisible(); setIndicator();
      activated = true;
    }
    car._activate = activate;   // a sibling toggle calls this when it reveals a hidden task

    if (!wrap) { prev.style.display = 'none'; next.style.display = 'none'; indicator.style.display = 'none'; }
    if (car.offsetParent !== null) activate();   // visible now; hidden tasks wait for their toggle
    // Re-measure once layout/fonts settle, in case the viewport width wasn't final at init.
    requestAnimationFrame(function () { if (activated) { computeMetrics(); place(false); } });
    car._onLoad = function () { if (car.offsetParent !== null) activate(); };
  }

  const cars = Array.prototype.slice.call(document.querySelectorAll('.rt-carousel'));
  cars.forEach(buildCarousel);
  // Single delegated listeners: rebuilds swap the per-carousel handlers, so no stale closures pile up.
  window.addEventListener('resize', function () { cars.forEach(function (c) { if (c._onResize) c._onResize(); }); });
  window.addEventListener('load', function () { cars.forEach(function (c) { if (c._onLoad) c._onLoad(); }); });
  // A breakpoint flip changes the per-row count (and with it clones and widths) -> rebuild.
  mqMobile.addEventListener('change', function () { cars.forEach(buildCarousel); });
}

// Switch the active task in a reasoning-trace group: slide the toggle thumb, reveal that task's carousel.
function rtTask(groupId, idx, btn) {
  const toggle = btn.closest('.seg-toggle');
  const btns = Array.prototype.slice.call(toggle.querySelectorAll('.seg-btn'));
  btns.forEach(function (b, i) { b.classList.toggle('is-active', i === idx); });
  const thumb = toggle.querySelector('.seg-thumb');
  if (thumb) thumb.style.transform = 'translateX(' + (idx * 100) + '%)';
  const cars = document.getElementById(groupId).querySelectorAll('.rt-carousel');
  cars.forEach(function (c, i) {
    c.style.display = (i === idx) ? '' : 'none';
    if (i !== idx) c.querySelectorAll('video').forEach(function (v) { v.pause(); });
  });
  if (cars[idx] && cars[idx]._activate) cars[idx]._activate();
}

function initializeBulmaComponents() {
  const navbarBurgers = document.querySelectorAll('.navbar-burger');
  navbarBurgers.forEach(burger => {
    burger.addEventListener('click', () => {
      const target = document.getElementById(burger.dataset.target);
      burger.classList.toggle('is-active');
      target.classList.toggle('is-active');
    });
  });
}

function isElementInViewport(el) {
  const rect = el.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

function isInVisibleContainer(el) {
  let parent = el.parentElement;
  while (parent && parent !== document.body) {
    const style = window.getComputedStyle(parent);
    if (style.display === 'none') return false;
    parent = parent.parentElement;
  }
  return true;
}

function setupLazyLoading() {
  if ('IntersectionObserver' in window) {
    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const video = entry.target;
        if (entry.isIntersecting && isInVisibleContainer(video)) {
          video.play().catch(() => { });
        } else {
          video.pause();
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('video').forEach(video => {
      videoObserver.observe(video);
    });
  }
}
