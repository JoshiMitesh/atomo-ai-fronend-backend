const fs = require('fs');
const path = require('path');
const express = require('express');

// Keep the original frontend source untouched. This preload patches the served
// app.js so Live Monitor contains only the newest 100 events and updates cards
// without clearing/rebuilding the entire list.
const originalStatic = express.static;

function patchAppJs(source) {
  let patched = source;

  // Clear button is intentionally removed from the UI. Guard its old listener
  // so setupEventListeners() continues and Authorised/Unauthorised tabs work.
  patched = patched.replace(
`  // Clear Events logs
  btnClearEvents.addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear the entire events log history? This will also remove saved cropped faces.')) {
      try {
        await fetch('/api/events', { method: 'DELETE' });
      } catch (err) {
        console.error('Failed to clear events:', err);
      }
    }
  });`,
`  // Clear Events button is intentionally hidden/removed in Atomic Vision.
  if (btnClearEvents) {
    btnClearEvents.addEventListener('click', async () => {
      if (confirm('Are you sure you want to clear the entire events log history? This will also remove saved cropped faces.')) {
        try {
          await fetch('/api/events', { method: 'DELETE' });
        } catch (err) {
          console.error('Failed to clear events:', err);
        }
      }
    });
  }`
  );

  // Keep only latest 100 on initial WebSocket state too.
  patched = patched.replace(
    `    allEvents = msg.data.events;`,
    `    allEvents = Array.isArray(msg.data.events) ? msg.data.events.slice(0, 100) : [];`
  );

  // Regardless of how many events fetchEvents() receives, keep newest 100.
  patched = patched.replace(
    /allEvents\s*=\s*await\s+res\.json\(\);/g,
    `allEvents = (await res.json()).slice(0, 100);`
  );
  patched = patched.replace(
    /allEvents\s*=\s*await\s+response\.json\(\);/g,
    `allEvents = (await response.json()).slice(0, 100);`
  );

  patched = patched.replace(
`  } else if (msg.event === 'database_updated') {
    fetchPersons();
    fetchEvents();
    if (currentPersonId && modalPersonDetails && !modalPersonDetails.classList.contains('hidden')) {`,
`  } else if (msg.event === 'database_updated') {
    fetchPersons();
    // Recognition events already arrive through WebSocket; avoid full re-render.
    if (currentPersonId && modalPersonDetails && !modalPersonDetails.classList.contains('hidden')) {`
  );

  patched = patched.replace(
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();
    fetchEvents();`,
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();
    // Avoid rebuilding all event cards when clusters change.`
  );

  patched = patched.replace(
`function prependRecognitionEvent(ev) {
  allEvents.unshift(ev);
  if (allEvents.length > 100) allEvents.pop();
  filterAndRenderEvents();
}`,
`function prependRecognitionEvent(ev) {
  allEvents.unshift(ev);
  let removedEvent = null;
  if (allEvents.length > 100) removedEvent = allEvents.pop();

  const shouldShow = currentEventFilter === 'all'
    || (currentEventFilter === 'known' && ev.is_known === true)
    || (currentEventFilter === 'unknown' && ev.is_known === false);

  if (shouldShow) {
    const empty = eventsList.querySelector('.empty-state');
    if (empty) empty.remove();
    appendEventHTML(ev, true);
  }

  if (removedEvent) {
    const oldItem = eventsList.querySelector(\`[data-event-id="\${removedEvent.id}"]\`);
    if (oldItem) oldItem.remove();
  }
}`
  );

  patched = patched.replace(
`function updateRecognitionEvent(ev) {
  const index = allEvents.findIndex(e => e.id === ev.id);
  if (index !== -1) {
    allEvents[index] = { ...allEvents[index], ...ev };
  }
  filterAndRenderEvents();
}`,
`function updateRecognitionEvent(ev) {
  const index = allEvents.findIndex(e => e.id === ev.id);
  if (index === -1) return;

  allEvents[index] = { ...allEvents[index], ...ev };
  const updated = allEvents[index];
  const shouldShow = currentEventFilter === 'all'
    || (currentEventFilter === 'known' && updated.is_known === true)
    || (currentEventFilter === 'unknown' && updated.is_known === false);

  const existingItem = eventsList.querySelector(\`[data-event-id="\${updated.id}"]\`);

  if (!shouldShow) {
    if (existingItem) existingItem.remove();
    if (!eventsList.querySelector('.event-item')) {
      eventsList.innerHTML = \`
        <div class="empty-state" id="events-empty-state">
          <i class="fa-solid fa-bell-slash"></i>
          <p>No matching face events found.</p>
        </div>
      \`;
    }
    return;
  }

  if (!existingItem) {
    const empty = eventsList.querySelector('.empty-state');
    if (empty) empty.remove();
    appendEventHTML(updated, index === 0);
    return;
  }

  appendEventHTML(updated, false);
  const replacement = eventsList.lastElementChild;
  existingItem.replaceWith(replacement);
}`
  );

  // LAN-only live streaming: the recovery helper replaces the original WHEP
  // connection routine after app.js loads. No public STUN/TURN service is used.
  patched += `\n;(() => {\n  const s=document.createElement('script');\n  s.src='/js/offlineRtspRecovery.js?v=1';\n  s.async=false;\n  document.head.appendChild(s);\n})();\n`;

  return patched;
}

express.static = function patchedStatic(root, options) {
  const normalStatic = originalStatic(root, options);
  const isPublicRoot = path.basename(path.resolve(root)) === 'public';
  if (!isPublicRoot) return normalStatic;
  return function noFlickerStatic(req, res, next) {
    if (req.path === '/js/app.js') {
      try {
        const appJsPath = path.join(root, 'js', 'app.js');
        const source = fs.readFileSync(appJsPath, 'utf8');
        res.type('application/javascript').send(patchAppJs(source));
        return;
      } catch (err) {
        console.error('[UI] Failed to apply no-flicker event patch:', err.message);
      }
    }
    normalStatic(req, res, next);
  };
};

console.log('[UI] No-flicker live events + LAN-only RTSP/WebRTC auto recovery enabled.');
