const fs = require('fs');
const path = require('path');
const express = require('express');

// Keep the original frontend source untouched. This preload only patches the
// served app.js in memory so live recognition updates change one event card
// instead of clearing and rebuilding the entire events list.
const originalStatic = express.static;

function patchAppJs(source) {
  let patched = source;

  patched = patched.replace(
`  } else if (msg.event === 'database_updated') {
    fetchPersons();
    fetchEvents();
    if (currentPersonId && modalPersonDetails && !modalPersonDetails.classList.contains('hidden')) {`,
`  } else if (msg.event === 'database_updated') {
    fetchPersons();
    // Do not re-fetch/re-render the live events list here. Recognition events
    // already arrive through recognition_event / recognition_update WebSockets.
    if (currentPersonId && modalPersonDetails && !modalPersonDetails.classList.contains('hidden')) {`
  );

  patched = patched.replace(
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();
    fetchEvents();`,
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();
    // Avoid rebuilding all event cards when an unknown cluster changes.`
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

  // Build the updated card using the existing renderer, then move that new
  // node into the old card's position. All operations happen synchronously,
  // so the rest of the event list never disappears or flickers.
  appendEventHTML(updated, false);
  const replacement = eventsList.lastElementChild;
  existingItem.replaceWith(replacement);
}`
  );

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

console.log('[UI] No-flicker live event updates enabled.');
