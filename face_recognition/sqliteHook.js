const fs = require('fs');
const path = require('path');
const Module = require('module');
const Database = require('better-sqlite3');

const DATA_DIR = path.join(__dirname, 'data');
const JSON_FILE = path.join(DATA_DIR, 'database.json');
const BACKUP_FILE = path.join(DATA_DIR, 'database.json.backup');
const SQLITE_FILE = path.join(DATA_DIR, 'face_recognition.db');
const DB_MODULE = path.join(__dirname, 'db.js');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const sql = new Database(SQLITE_FILE);
sql.pragma('journal_mode = WAL');
sql.pragma('synchronous = NORMAL');
sql.pragma('foreign_keys = ON');
sql.pragma('busy_timeout = 5000');

sql.exec(`
CREATE TABLE IF NOT EXISTS app_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  data TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, gender TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY, person_id TEXT, filename TEXT, embedding TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, timestamp TEXT, person_id TEXT, person_name TEXT,
  score REAL, crop_filename TEXT, is_known INTEGER, camera_id TEXT, camera_name TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY, name TEXT, rtsp_url TEXT, is_active INTEGER,
  line_crossing_enabled INTEGER, line_y REAL, line_direction TEXT,
  line_x_start REAL, line_x_end REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY, name TEXT, gender TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS cluster_photos (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT, cluster_id TEXT, photo_id TEXT,
  filename TEXT, embedding TEXT, gender TEXT
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY, value TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
CREATE INDEX IF NOT EXISTS idx_cluster_photos_cluster ON cluster_photos(cluster_id);
`);

function emptyState() {
  return { persons: [], photos: [], events: [], cameras: [], clusters: [], cluster_counter: 0 };
}

function normalize(state) {
  const s = state && typeof state === 'object' ? state : {};
  return {
    ...s,
    persons: Array.isArray(s.persons) ? s.persons : [],
    photos: Array.isArray(s.photos) ? s.photos : [],
    events: Array.isArray(s.events) ? s.events : [],
    cameras: Array.isArray(s.cameras) ? s.cameras : [],
    clusters: Array.isArray(s.clusters) ? s.clusters : [],
    cluster_counter: Number.isFinite(s.cluster_counter) ? s.cluster_counter : (Array.isArray(s.clusters) ? s.clusters.length : 0)
  };
}

const replaceMirrors = sql.transaction(state => {
  sql.prepare('DELETE FROM persons').run();
  sql.prepare('DELETE FROM photos').run();
  sql.prepare('DELETE FROM events').run();
  sql.prepare('DELETE FROM cameras').run();
  sql.prepare('DELETE FROM clusters').run();
  sql.prepare('DELETE FROM cluster_photos').run();
  sql.prepare('DELETE FROM settings').run();

  const personStmt = sql.prepare('INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)');
  for (const p of state.persons) personStmt.run(p.id, p.name || '', p.gender || 'Unknown', p.created_at || null);

  const photoStmt = sql.prepare('INSERT INTO photos(id,person_id,filename,embedding) VALUES(?,?,?,?)');
  for (const p of state.photos) photoStmt.run(p.id, p.person_id || null, p.filename || null, JSON.stringify(p.embedding || null));

  const eventStmt = sql.prepare('INSERT INTO events(id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name) VALUES(?,?,?,?,?,?,?,?,?)');
  for (const e of state.events) eventStmt.run(e.id, e.timestamp || null, e.person_id || null, e.person_name || null, Number(e.score || 0), e.crop_filename || null, e.is_known ? 1 : 0, e.camera_id || null, e.camera_name || null);

  const cameraStmt = sql.prepare('INSERT INTO cameras(id,name,rtsp_url,is_active,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)');
  for (const c of state.cameras) cameraStmt.run(c.id, c.name || '', c.rtsp_url || '', c.is_active ? 1 : 0, c.line_crossing_enabled ? 1 : 0, Number(c.line_y ?? 0.6), c.line_direction || 'in', Number(c.line_x_start ?? 0), Number(c.line_x_end ?? 1), c.created_at || null);

  const clusterStmt = sql.prepare('INSERT INTO clusters(id,name,gender,created_at) VALUES(?,?,?,?)');
  const clusterPhotoStmt = sql.prepare('INSERT INTO cluster_photos(cluster_id,photo_id,filename,embedding,gender) VALUES(?,?,?,?,?)');
  for (const c of state.clusters) {
    clusterStmt.run(c.id, c.name || '', c.gender || 'Unknown', c.created_at || null);
    for (const p of (c.photos || [])) clusterPhotoStmt.run(c.id, p.id || null, p.filename || null, JSON.stringify(p.embedding || null), p.gender || 'Unknown');
  }

  const settingsStmt = sql.prepare('INSERT INTO settings(key,value) VALUES(?,?)');
  if (state.settings && typeof state.settings === 'object') {
    for (const [key, value] of Object.entries(state.settings)) settingsStmt.run(key, JSON.stringify(value));
  }
});

function saveState(input) {
  const state = normalize(input);
  const text = JSON.stringify(state);
  const now = new Date().toISOString();
  sql.prepare(`INSERT INTO app_state(id,data,updated_at) VALUES(1,?,?)
               ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at`).run(text, now);
  replaceMirrors(state);
}

function loadState() {
  const row = sql.prepare('SELECT data FROM app_state WHERE id=1').get();
  if (!row) return null;
  try { return normalize(JSON.parse(row.data)); }
  catch (err) { throw new Error(`SQLite app_state is invalid: ${err.message}`); }
}

function migrateIfNeeded() {
  if (loadState()) return;
  let state = emptyState();
  if (fs.existsSync(JSON_FILE)) {
    const raw = fs.readFileSync(JSON_FILE, 'utf8');
    state = normalize(JSON.parse(raw));
    if (!fs.existsSync(BACKUP_FILE)) fs.copyFileSync(JSON_FILE, BACKUP_FILE);
    console.log(`[SQLite] Migrating existing database.json (${state.persons.length} persons, ${state.events.length} events, ${state.cameras.length} cameras, ${state.clusters.length} clusters)...`);
  }
  saveState(state);
  console.log(`[SQLite] Database ready: ${SQLITE_FILE}`);
  if (fs.existsSync(BACKUP_FILE)) console.log(`[SQLite] Original JSON backup: ${BACKUP_FILE}`);
}

migrateIfNeeded();

// db.js is intentionally left unchanged. Intercept only its database.json reads/writes
// so every existing method and recognition behavior remains exactly the same.
const originalReadFileSync = fs.readFileSync.bind(fs);
const originalWriteFileSync = fs.writeFileSync.bind(fs);
const originalExistsSync = fs.existsSync.bind(fs);

function isJsonDb(file) {
  try { return path.resolve(String(file)) === path.resolve(JSON_FILE); }
  catch (_) { return false; }
}

fs.existsSync = function(file) {
  if (isJsonDb(file)) return true;
  return originalExistsSync(file);
};
fs.readFileSync = function(file, options) {
  if (isJsonDb(file)) return JSON.stringify(loadState() || emptyState(), null, 2);
  return originalReadFileSync(file, options);
};
fs.writeFileSync = function(file, data, options) {
  if (isJsonDb(file)) {
    const text = Buffer.isBuffer(data) ? data.toString('utf8') : String(data);
    saveState(JSON.parse(text));
    return;
  }
  return originalWriteFileSync(file, data, options);
};

process.on('exit', () => { try { sql.close(); } catch (_) {} });
console.log('[SQLite] Persistence layer enabled (WAL mode).');
