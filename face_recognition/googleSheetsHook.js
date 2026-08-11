// Google Sheets integration loaded before server.js.
// It wraps the existing DB event methods without changing face detection,
// recognition, clustering, tracking, or the original Python worker.
const db = require('./db');
const sheetsLogger = require('./googleSheetsLogger');

const originalAddEvent = db.addEvent.bind(db);
const originalUpdateEvent = db.updateEvent.bind(db);
const pendingTimers = new Map();

function scheduleUnknownFallback(event) {
  if (!event || !event.id) return;
  const timer = setTimeout(async () => {
    pendingTimers.delete(event.id);
    try {
      const latest = db.getEvents(1000).find(e => e.id === event.id) || event;
      await sheetsLogger.logEvent({ ...latest, recognition_finalized: true });
    } catch (err) {
      console.error('[Google Sheets] Failed to export event:', err.message);
    }
  }, 5000);
  pendingTimers.set(event.id, timer);
}

db.addEvent = function (...args) {
  const event = originalAddEvent(...args);
  // Recognition normally follows stream_detect. This fallback ensures an event
  // is still exported if the recognition response never arrives.
  scheduleUnknownFallback(event);
  return event;
};

db.updateEvent = function (eventId, updates) {
  const event = originalUpdateEvent(eventId, updates);
  if (event) {
    const timer = pendingTimers.get(eventId);
    if (timer) {
      clearTimeout(timer);
      pendingTimers.delete(eventId);
    }
    sheetsLogger.logEvent({ ...event, recognition_finalized: true }).catch(err => {
      console.error('[Google Sheets] Failed to export event:', err.message);
    });
  }
  return event;
};
