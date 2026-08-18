(() => {
  'use strict';

  const NORMAL_LABELS = new Set(['Live Monitor', 'Profiles', 'Event Gallery', 'Settings']);
  const saved = new Map();

  function text(el) {
    return String(el?.textContent || '').trim();
  }

  function candidates() {
    return [...document.querySelectorAll('a,button,[role="button"],li')]
      .filter(el => {
        const t = text(el);
        return NORMAL_LABELS.has(t) || t === 'Demo';
      });
  }

  function save(el) {
    if (!el || saved.has(el)) return;
    saved.set(el, el.getAttribute('style'));
  }

  function clearVisual(el) {
    save(el);
    el.style.setProperty('background', 'transparent', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
    el.style.setProperty('border-color', 'transparent', 'important');
    el.style.setProperty('box-shadow', 'none', 'important');
  }

  function selectDemo(el) {
    save(el);
    el.style.setProperty('background', 'rgba(99, 91, 255, 0.22)', 'important');
    el.style.setProperty('background-color', 'rgba(99, 91, 255, 0.22)', 'important');
    el.style.setProperty('border', '1px solid rgba(108, 99, 255, 0.48)', 'important');
    el.style.setProperty('box-shadow', 'none', 'important');
    el.style.setProperty('color', '#ffffff', 'important');
  }

  function restore() {
    for (const [el, oldStyle] of saved.entries()) {
      if (!el.isConnected) continue;
      if (oldStyle === null) el.removeAttribute('style');
      else el.setAttribute('style', oldStyle);
    }
    saved.clear();
  }

  function sync() {
    const demoOpen = !!document.getElementById('atomic-demo-root');
    if (!demoOpen) {
      restore();
      return;
    }

    const items = candidates();
    const demos = items.filter(el => text(el) === 'Demo');

    // Prefer the actual sidebar Demo item, not text that might appear inside content.
    const demo = demos.find(el => el.hasAttribute('data-atomic-demo-nav')) || demos[0];

    for (const el of items) {
      if (el === demo) selectDemo(el);
      else if (NORMAL_LABELS.has(text(el))) clearVisual(el);
    }
  }

  // When the user navigates from Demo to any normal sidebar page, remove the
  // overlay immediately and allow the application's original click handler to run.
  document.addEventListener('click', (event) => {
    const el = event.target.closest('a,button,[role="button"],li');
    if (!el || !document.getElementById('atomic-demo-root')) return;
    if (!NORMAL_LABELS.has(text(el))) return;

    const fireVideo = document.getElementById('atomic-fire-video');
    if (fireVideo) {
      try { fireVideo.pause(); } catch (_) {}
    }
    document.getElementById('atomic-demo-modal')?.remove();
    document.getElementById('atomic-demo-root')?.remove();
    restore();
  }, true);

  const observer = new MutationObserver(sync);
  observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });

  // Keep state correct when Demo itself is clicked or its inner page changes.
  document.addEventListener('click', () => setTimeout(sync, 0));
  sync();
})();
