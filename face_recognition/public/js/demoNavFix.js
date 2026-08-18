(() => {
  'use strict';

  function closeDemoForSidebarNavigation(e) {
    const root = document.getElementById('atomic-demo-root');
    if (!root) return;

    const nav = e.target.closest('a, button, [role="button"], li');
    if (!nav) return;

    // Demo's own controls must continue to be handled by demo.js.
    if (nav.closest('#atomic-demo-root') || nav.closest('#atomic-demo-modal')) return;
    if (nav.matches('[data-atomic-demo-nav]') || nav.closest('[data-atomic-demo-nav]')) return;

    // Only react to the application's sidebar/navigation items.
    const text = (nav.textContent || '').trim();
    const appTabs = ['Live Monitor', 'Profiles', 'Event Gallery', 'Settings'];
    if (!appTabs.some(name => text === name || text.includes(name))) return;

    // Remove the Demo overlay immediately, then allow the original app click
    // handler to run normally and open the requested page.
    document.getElementById('atomic-demo-modal')?.remove();
    root.remove();
  }

  // Capture phase removes the overlay before the application's normal sidebar
  // click handler executes. We intentionally do not preventDefault/stopPropagation.
  document.addEventListener('click', closeDemoForSidebarNavigation, true);
})();
