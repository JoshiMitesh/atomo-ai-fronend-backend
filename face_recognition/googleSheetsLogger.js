const { google } = require('googleapis');

const enabled = String(process.env.GOOGLE_SHEETS_ENABLED || '').toLowerCase() === 'true';
const spreadsheetId = process.env.GOOGLE_SHEETS_ID || '';
const sheetName = process.env.GOOGLE_SHEETS_TAB || 'Events';
const publicBaseUrl = (process.env.PUBLIC_BASE_URL || '').replace(/\/$/, '');

let sheets = null;
let initialized = false;
const loggedEventIds = new Set();

async function getSheets() {
  if (!enabled || !spreadsheetId) return null;
  if (sheets) return sheets;

  const auth = new google.auth.GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/spreadsheets']
  });
  const client = await auth.getClient();
  sheets = google.sheets({ version: 'v4', auth: client });
  return sheets;
}

async function ensureHeader(api) {
  if (initialized) return;
  const range = `'${sheetName}'!A1:H1`;
  const current = await api.spreadsheets.values.get({ spreadsheetId, range });
  const values = current.data.values || [];
  if (!values.length) {
    await api.spreadsheets.values.update({
      spreadsheetId,
      range,
      valueInputOption: 'RAW',
      requestBody: {
        values: [[
          'Timestamp', 'Person', 'Status', 'Camera', 'Zone',
          'Confidence', 'Photo', 'Event ID'
        ]]
      }
    });
  }
  initialized = true;
}

function photoUrl(event) {
  if (!event.crop_filename) return '';
  if (!publicBaseUrl) return event.crop_filename;
  return `${publicBaseUrl}/crops/${encodeURIComponent(event.crop_filename)}`;
}

async function logEvent(event) {
  if (!enabled || !spreadsheetId || !event || !event.id) return;
  if (loggedEventIds.has(event.id)) return;

  // Only export after recognition has finalized the event.
  if (event.recognition_finalized !== true) return;

  const api = await getSheets();
  if (!api) return;
  await ensureHeader(api);

  const status = event.is_known ? 'Authorized' : 'Unauthorized';
  const person = event.is_known ? (event.person_name || 'Known') : 'Unknown';
  const zone = event.zone || event.camera_name || 'Unknown';
  const confidence = Number.isFinite(Number(event.score)) ? Number(event.score) : 0;

  await api.spreadsheets.values.append({
    spreadsheetId,
    range: `'${sheetName}'!A:H`,
    valueInputOption: 'USER_ENTERED',
    insertDataOption: 'INSERT_ROWS',
    requestBody: {
      values: [[
        event.timestamp || new Date().toISOString(),
        person,
        status,
        event.camera_name || 'Unknown',
        zone,
        confidence,
        photoUrl(event),
        event.id
      ]]
    }
  });

  loggedEventIds.add(event.id);
}

module.exports = { logEvent };
