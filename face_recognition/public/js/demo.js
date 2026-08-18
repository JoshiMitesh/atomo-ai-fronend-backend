(() => {
  'use strict';

  const state = {
    screen: 'home',
    videos: [],
    fireVideos: [],
    events: [],
    selectedFire: null,
    job: null,
    poll: null,
    previewTimer: null,
    renderToken: 0,
    controllers: new Set(),
  };

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  })[c]);

  async function api(url, opts = {}) {
    const controller = new AbortController();
    state.controllers.add(controller);
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      const r = await fetch(url, { ...opts, signal: controller.signal, cache: 'no-store' });
      let body = {};
      try { body = await r.json(); } catch (_) {}
      if (!r.ok) throw new Error(body.error || `Request failed: ${r.status}`);
      return body;
    } catch (e) {
      if (e.name === 'AbortError') throw new Error(`Request timed out: ${url}`);
      throw e;
    } finally {
      clearTimeout(timeout);
      state.controllers.delete(controller);
    }
  }

  function addStyle() {
    if (document.getElementById('atomic-demo-style')) return;
    const s = document.createElement('style');
    s.id = 'atomic-demo-style';
    s.textContent = `
#atomic-demo-root{position:fixed;inset:0 0 0 265px;background:#090c12;color:#f4f6fb;z-index:8000;overflow:auto;padding:30px;font-family:inherit}
.atomic-demo-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}.atomic-demo-head h1{font-size:28px;margin:0}
.atomic-back,.atomic-btn{border:1px solid #2a3040;background:#151a25;color:#fff;border-radius:10px;padding:10px 15px;cursor:pointer}.atomic-btn.primary{background:#665cf6;border-color:#665cf6}
.atomic-home-wrap{min-height:calc(100vh - 150px);display:flex;align-items:center;justify-content:center}.atomic-home-grid{width:min(760px,100%);display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:24px}
.atomic-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:18px}.atomic-card{background:#121722;border:1px solid #242b3a;border-radius:16px;padding:20px;cursor:pointer;min-height:130px}.atomic-card:hover{border-color:#665cf6}.atomic-card h3{margin:8px 0;font-size:18px}.atomic-card p,.atomic-muted{color:#9ca5b7}
.atomic-video-card video{width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;object-fit:cover}.atomic-status{margin:14px 0;padding:12px 14px;border:1px solid #293044;border-radius:10px;background:#111620}.atomic-error{margin:14px 0;padding:12px 14px;border:1px solid rgba(239,68,68,.45);border-radius:10px;background:rgba(239,68,68,.08);color:#ff8b95}
.atomic-fire-workspace{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.7fr);gap:20px;margin:24px 0}.atomic-preview{background:#05070a;border:1px solid #242b3a;border-radius:16px;overflow:hidden;min-height:420px;display:flex;align-items:center;justify-content:center}.atomic-preview img{width:100%;max-height:68vh;object-fit:contain;display:block}.atomic-preview-empty{text-align:center;color:#8f98aa;padding:40px}.atomic-side{background:#111620;border:1px solid #242b3a;border-radius:16px;padding:18px}.atomic-section-title{display:flex;align-items:center;justify-content:space-between;margin:30px 0 14px}.atomic-section-title h2{margin:0;font-size:21px}.atomic-event{cursor:default}.atomic-event img{width:100%;height:180px;object-fit:contain;background:#05070a;border-radius:10px}
.atomic-modal{position:fixed;inset:0;background:rgba(0,0,0,.82);z-index:9000;display:flex;align-items:center;justify-content:center;padding:30px}.atomic-modal-box{width:min(1000px,92vw);background:#10141e;border:1px solid #2a3040;border-radius:16px;overflow:hidden}.atomic-modal-head{display:flex;justify-content:space-between;align-items:center;padding:16px 20px}.atomic-modal-head button{font-size:25px;background:none;border:0;color:white;cursor:pointer}.atomic-modal video{width:100%;max-height:75vh;background:#000;display:block}
@media(max-width:1000px){.atomic-fire-workspace{grid-template-columns:1fr}}@media(max-width:850px){#atomic-demo-root{left:0;padding:18px}.atomic-home-grid,.atomic-grid{grid-template-columns:1fr}}
`;
    document.head.appendChild(s);
  }

  function root() {
    let r = document.getElementById('atomic-demo-root');
    if (!r) {
      r = document.createElement('section');
      r.id = 'atomic-demo-root';
      document.body.appendChild(r);
    }
    return r;
  }

  function stopTimers() {
    clearInterval(state.poll);
    clearInterval(state.previewTimer);
    state.poll = null;
    state.previewTimer = null;
  }

  function abortRequests() {
    for (const c of state.controllers) {
      try { c.abort(); } catch (_) {}
    }
    state.controllers.clear();
  }

  function leaveScreen() {
    state.renderToken++;
    stopTimers();
    abortRequests();
  }

  function closeDemo() {
    leaveScreen();
    document.getElementById('atomic-demo-modal')?.remove();
    document.getElementById('atomic-demo-root')?.remove();
  }

  function header(title, back) {
    return `<div class="atomic-demo-head"><div><h1>${esc(title)}</h1><div class="atomic-muted">Atomic Vision demonstration</div></div>${back ? '<button class="atomic-back" data-action="home">← Back</button>' : '<button class="atomic-back" data-action="close">Close</button>'}</div>`;
  }

  function renderHome() {
    leaveScreen();
    state.screen = 'home';
    root().innerHTML = header('Demo', false) + `
      <div class="atomic-home-wrap"><div class="atomic-home-grid">
        <div class="atomic-card" data-action="videos"><h3>▶ Demo Videos</h3><p>Browse your demo videos and play any video in a popup.</p></div>
        <div class="atomic-card" data-action="fire"><h3>🔥 Live Fire Detection</h3><p>Select a video, watch detection live, and review fire events on the same page.</p></div>
      </div></div>`;
  }

  async function renderVideos() {
    leaveScreen();
    const token = ++state.renderToken;
    state.screen = 'videos';
    root().innerHTML = header('Demo Videos', true) + '<div class="atomic-status">Loading videos...</div>';
    try {
      const videos = await api('/api/demo/videos');
      if (token !== state.renderToken || state.screen !== 'videos') return;
      state.videos = videos;
      root().innerHTML = header('Demo Videos', true) + (videos.length
        ? `<div class="atomic-grid">${videos.map(v => `<div class="atomic-card atomic-video-card" data-action="play" data-url="${esc(v.url)}" data-name="${esc(v.name)}"><video src="${esc(v.url)}" muted preload="metadata"></video><h3>${esc(v.name)}</h3><p>Click to play</p></div>`).join('')}</div>`
        : '<div class="atomic-status">No demo videos found.</div>');
    } catch (e) {
      if (token === state.renderToken) root().innerHTML = header('Demo Videos', true) + `<div class="atomic-error">${esc(e.message)}</div>`;
    }
  }

  function statusHTML() {
    if (!state.job) return '<div class="atomic-muted">Select a video and press Start Detection.</div>';
    return `<div><b>Status:</b> <span id="atomic-job-status">${esc(state.job.status)}</span></div><div style="margin-top:8px"><b>Progress:</b> <span id="atomic-job-progress">${Number(state.job.progress || 0)}%</span></div>${state.job.error ? `<div class="atomic-error">${esc(state.job.error)}</div>` : ''}`;
  }

  function eventCards() {
    if (!state.events.length) return '<div class="atomic-status">No fire events detected yet.</div>';
    return `<div class="atomic-grid">${state.events.map(e => {
      const d = new Date(e.timestamp);
      return `<div class="atomic-card atomic-event"><img src="${esc(e.image_url)}?v=${encodeURIComponent(e.id || e.timestamp)}"><h3>🔥 FIRE</h3><p>${esc(e.video || '')}</p><p>${esc(d.toLocaleDateString())} • ${esc(d.toLocaleTimeString())}</p></div>`;
    }).join('')}</div>`;
  }

  function previewHTML() {
    if (!state.job?.preview_url) {
      return '<div class="atomic-preview-empty">Live detection preview will appear here after Start Detection.</div>';
    }
    return `<img id="atomic-live-preview" alt="Live fire detection preview" src="${esc(state.job.preview_url)}?t=${Date.now()}" onerror="this.style.visibility='hidden'" onload="this.style.visibility='visible'">`;
  }

  function firePageHTML() {
    const selected = state.fireVideos.find(v => v.name === state.selectedFire);
    return header('Live Fire Detection', true) + `
      <div class="atomic-section-title"><h2>Detection Videos</h2></div>
      ${state.fireVideos.length ? `<div class="atomic-grid">${state.fireVideos.map(v => `<div class="atomic-card atomic-video-card" data-action="select-fire" data-name="${esc(v.name)}"><video src="${esc(v.url)}" muted preload="metadata"></video><h3>${esc(v.name)}</h3><p>${state.selectedFire === v.name ? 'Selected' : 'Click to select'}</p></div>`).join('')}</div>` : '<div class="atomic-status">No fire videos found.</div>'}
      ${selected ? `<div class="atomic-fire-workspace"><div class="atomic-preview" id="atomic-preview-box">${previewHTML()}</div><div class="atomic-side"><h3>${esc(selected.name)}</h3><div class="atomic-status">${statusHTML()}</div><button class="atomic-btn primary" data-action="start-fire">Start Detection</button></div></div>` : ''}
      <div class="atomic-section-title"><h2>Detected Fire Events</h2><span class="atomic-muted" id="atomic-event-count">${state.events.length} events</span></div>
      <div id="atomic-fire-events">${eventCards()}</div>`;
  }

  async function renderFire() {
    leaveScreen();
    const token = ++state.renderToken;
    state.screen = 'fire';
    root().innerHTML = header('Live Fire Detection', true) + '<div class="atomic-status">Loading detection videos...</div>';
    try {
      const videos = await api('/api/demo/fire/videos');
      if (token !== state.renderToken || state.screen !== 'fire') return;
      state.fireVideos = videos;
      root().innerHTML = firePageHTML();
      refreshEvents();
      if (state.job && ['running','starting'].includes(state.job.status)) startPolling();
    } catch (e) {
      if (token === state.renderToken) root().innerHTML = header('Live Fire Detection', true) + `<div class="atomic-error">${esc(e.message)}</div>`;
    }
  }

  function refreshPreview() {
    if (state.screen !== 'fire' || !state.job?.preview_url) return;
    let img = document.getElementById('atomic-live-preview');
    const box = document.getElementById('atomic-preview-box');
    if (!box) return;
    if (!img) {
      box.innerHTML = '<img id="atomic-live-preview" alt="Live fire detection preview">';
      img = document.getElementById('atomic-live-preview');
    }
    img.onload = () => { img.style.visibility = 'visible'; };
    img.onerror = () => { img.style.visibility = 'hidden'; };
    img.src = `${state.job.preview_url}?t=${Date.now()}`;
  }

  function refreshJobDOM() {
    const st = document.getElementById('atomic-job-status');
    const pr = document.getElementById('atomic-job-progress');
    if (st) st.textContent = state.job?.status || '';
    if (pr) pr.textContent = `${Number(state.job?.progress || 0)}%`;
    refreshPreview();
  }

  async function refreshEvents() {
    if (state.screen !== 'fire') return;
    try {
      const events = await api('/api/demo/fire/events');
      if (state.screen !== 'fire') return;
      if (JSON.stringify(events.map(e => e.id)) !== JSON.stringify(state.events.map(e => e.id))) {
        state.events = events;
        const box = document.getElementById('atomic-fire-events');
        const count = document.getElementById('atomic-event-count');
        if (box) box.innerHTML = eventCards();
        if (count) count.textContent = `${events.length} events`;
      }
    } catch (_) {}
  }

  function startPolling() {
    stopTimers();
    state.poll = setInterval(async () => {
      if (state.screen !== 'fire' || !state.job) return;
      try {
        state.job = await api(`/api/demo/fire/jobs/${encodeURIComponent(state.job.id)}`);
        refreshJobDOM();
        refreshEvents();
        if (['completed','error','stopped'].includes(state.job.status)) stopTimers();
      } catch (_) {
        stopTimers();
      }
    }, 1000);
    state.previewTimer = setInterval(refreshPreview, 300);
  }

  async function startFire() {
    if (!state.selectedFire) return;
    try {
      state.job = await api('/api/demo/fire/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video: state.selectedFire }),
      });
      root().innerHTML = firePageHTML();
      // Preview endpoint may need one or two inference cycles before the first JPG
      // exists. Keep retrying immediately instead of waiting for a page re-render.
      refreshPreview();
      startPolling();
    } catch (e) {
      alert(e.message);
    }
  }

  function openVideo(url, name) {
    document.getElementById('atomic-demo-modal')?.remove();
    const m = document.createElement('div');
    m.className = 'atomic-modal';
    m.id = 'atomic-demo-modal';
    m.innerHTML = `<div class="atomic-modal-box"><div class="atomic-modal-head"><b>${esc(name)}</b><button data-action="modal-close">×</button></div><video src="${esc(url)}" controls autoplay></video></div>`;
    document.body.appendChild(m);
  }

  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const a = el.dataset.action;
    if (a === 'close') { e.preventDefault(); closeDemo(); }
    else if (a === 'home') { e.preventDefault(); renderHome(); }
    else if (a === 'videos') { e.preventDefault(); renderVideos(); }
    else if (a === 'fire') { e.preventDefault(); renderFire(); }
    else if (a === 'play') { e.preventDefault(); openVideo(el.dataset.url, el.dataset.name); }
    else if (a === 'select-fire') { e.preventDefault(); state.selectedFire = el.dataset.name; root().innerHTML = firePageHTML(); refreshEvents(); }
    else if (a === 'start-fire') { e.preventDefault(); e.stopPropagation(); startFire(); }
    else if (a === 'modal-close') { e.preventDefault(); document.getElementById('atomic-demo-modal')?.remove(); }
  });

  document.addEventListener('click', (e) => {
    const m = document.getElementById('atomic-demo-modal');
    if (m && e.target === m) m.remove();
  });

  function findSettings() {
    return [...document.querySelectorAll('a,button,[role="button"],li')].find(x => x.textContent.trim() === 'Settings');
  }

  function installNav() {
    if (document.querySelector('[data-atomic-demo-nav]')) return;
    const settings = findSettings();
    if (!settings) return;
    const item = settings.cloneNode(true);
    item.dataset.atomicDemoNav = '1';
    item.removeAttribute('href');
    item.querySelectorAll('[id]').forEach(x => x.removeAttribute('id'));
    const walker = document.createTreeWalker(item, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = walker.nextNode())) {
      if (n.nodeValue.trim() === 'Settings') n.nodeValue = n.nodeValue.replace('Settings', 'Demo');
    }
    const icon = item.querySelector('i');
    if (icon) icon.className = 'fa-solid fa-flask';
    item.style.cursor = 'pointer';
    item.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      addStyle();
      renderHome();
    });
    settings.parentNode.insertBefore(item, settings.nextSibling);
  }

  addStyle();
  installNav();
  new MutationObserver(installNav).observe(document.documentElement, { childList: true, subtree: true });
})();
