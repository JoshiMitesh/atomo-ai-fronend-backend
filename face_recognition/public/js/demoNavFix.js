(() => {
  'use strict';

  const APP_TABS = ['Live Monitor', 'Profiles', 'Event Gallery', 'Settings'];
  const savedStyles = new Map();

  const label = el => String(el?.textContent || '').trim();

  function saveStyle(el) {
    if (el && !savedStyles.has(el)) savedStyles.set(el, el.getAttribute('style'));
  }

  function clearActiveVisual(el) {
    saveStyle(el);
    el.style.setProperty('background', 'transparent', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
    el.style.setProperty('border-color', 'transparent', 'important');
    el.style.setProperty('box-shadow', 'none', 'important');
  }

  function setDemoActive(el) {
    saveStyle(el);
    el.style.setProperty('background', 'rgba(99, 91, 255, 0.22)', 'important');
    el.style.setProperty('background-color', 'rgba(99, 91, 255, 0.22)', 'important');
    el.style.setProperty('border', '1px solid rgba(108, 99, 255, 0.48)', 'important');
    el.style.setProperty('box-shadow', 'none', 'important');
    el.style.setProperty('color', '#ffffff', 'important');
  }

  function restoreStyles() {
    for (const [el, oldStyle] of savedStyles) {
      if (!el.isConnected) continue;
      if (oldStyle === null) el.removeAttribute('style');
      else el.setAttribute('style', oldStyle);
    }
    savedStyles.clear();
  }

  function syncDemoActiveState() {
    if (!document.getElementById('atomic-demo-root')) {
      restoreStyles();
      return;
    }

    const nodes = [...document.querySelectorAll('a,button,[role="button"],li')];
    const demo = nodes.find(el => el.hasAttribute('data-atomic-demo-nav'));
    if (!demo) return;

    for (const el of nodes) {
      const text = label(el);
      if (APP_TABS.includes(text)) clearActiveVisual(el);
    }
    setDemoActive(demo);
  }

  function closeDemoForSidebarNavigation(e) {
    const root = document.getElementById('atomic-demo-root');
    if (!root) return;

    const nav = e.target.closest('a, button, [role="button"], li');
    if (!nav) return;

    // Demo's own controls continue to be handled by demo.js.
    if (nav.closest('#atomic-demo-root') || nav.closest('#atomic-demo-modal')) return;
    if (nav.matches('[data-atomic-demo-nav]') || nav.closest('[data-atomic-demo-nav]')) return;

    const text = label(nav);
    if (!APP_TABS.some(name => text === name || text.includes(name))) return;

    // Remove Demo before the application's original navigation handler runs.
    const fireVideo = document.getElementById('atomic-fire-video');
    if (fireVideo) {
      try { fireVideo.pause(); } catch (_) {}
    }
    document.getElementById('atomic-demo-modal')?.remove();
    root.remove();
    restoreStyles();
  }

  // Capture phase closes Demo first but does not block the application's click.
  document.addEventListener('click', closeDemoForSidebarNavigation, true);
  document.addEventListener('click', () => setTimeout(syncDemoActiveState, 0));

  new MutationObserver(syncDemoActiveState).observe(document.documentElement, {
    childList: true,
    subtree: true
  });

  syncDemoActiveState();
})();
