const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const express = require('express');
const originalListen = express.application.listen;

const ROOT = __dirname;
const DEMO_ROOT = path.join(ROOT, 'demo_media');
const DEMO_VIDEOS = path.join(DEMO_ROOT, 'videos');
const FIRE_VIDEOS = path.join(DEMO_ROOT, 'fire_videos');
const FIRE_EVENTS = path.join(DEMO_ROOT, 'fire_events');
const EVENTS_JSON = path.join(FIRE_EVENTS, 'events.json');

for (const dir of [DEMO_VIDEOS, FIRE_VIDEOS, FIRE_EVENTS]) fs.mkdirSync(dir, { recursive: true });

const VIDEO_EXT = new Set(['.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v']);
const jobs = new Map();

function safeName(name) {
  const base = path.basename(String(name || ''));
  if (!base || base !== String(name || '') || base.includes('..')) return null;
  return base;
}
function listVideos(dir, routePrefix) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter(x => x.isFile() && VIDEO_EXT.has(path.extname(x.name).toLowerCase()))
    .map(x => ({ name: x.name, url: `${routePrefix}/${encodeURIComponent(x.name)}` }))
    .sort((a, b) => a.name.localeCompare(b.name));
}
function readEvents() {
  try { return JSON.parse(fs.readFileSync(EVENTS_JSON, 'utf8')); }
  catch (_) { return []; }
}
function writeEvents(events) {
  fs.writeFileSync(EVENTS_JSON, JSON.stringify(events.slice(0, 1000), null, 2));
}
function resolveFireConfig() {
  return {
    python: process.env.FIRE_PYTHON || 'python3',
    script: process.env.FIRE_SCRIPT || path.join(ROOT, 'fire_final3.py'),
    model: process.env.FIRE_MODEL || path.join(ROOT, 'stage2_best_3class_1024_fp16.nb'),
    library: process.env.FIRE_LIBRARY || path.join(ROOT, 'libnn_fire_smoke_3class_1024_fp16.fast.so'),
  };
}
function attach(app) {
  if (app.__atomicDemoBackendInstalled) return;
  app.__atomicDemoBackendInstalled = true;
  app.use('/demo-media/videos', express.static(DEMO_VIDEOS));
  app.use('/demo-media/fire-videos', express.static(FIRE_VIDEOS));
  app.use('/demo-media/fire-events', express.static(FIRE_EVENTS));

  app.get('/api/demo/videos', (_req, res) => res.json(listVideos(DEMO_VIDEOS, '/demo-media/videos')));
  app.get('/api/demo/fire/videos', (_req, res) => res.json(listVideos(FIRE_VIDEOS, '/demo-media/fire-videos')));
  app.get('/api/demo/fire/events', (_req, res) => res.json(readEvents()));
  app.get('/api/demo/fire/jobs/:id', (req, res) => {
    const job = jobs.get(req.params.id);
    if (!job) return res.status(404).json({ error: 'Detection job not found.' });
    res.json(job);
  });
  app.post('/api/demo/fire/start', express.json(), (req, res) => {
    const name = safeName(req.body && req.body.video);
    if (!name) return res.status(400).json({ error: 'Valid video name is required.' });
    const input = path.join(FIRE_VIDEOS, name);
    if (!fs.existsSync(input)) return res.status(404).json({ error: 'Fire demo video not found.' });

    const cfg = resolveFireConfig();
    for (const [label, file] of Object.entries({ detector: cfg.script, model: cfg.model, library: cfg.library })) {
      if (!fs.existsSync(file)) return res.status(500).json({ error: `${label} file not found: ${file}` });
    }

    const id = `fire_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const job = { id, video: name, status: 'starting', progress: 0, error: null, started_at: new Date().toISOString() };
    jobs.set(id, job);
    const worker = path.join(ROOT, 'fire_demo_worker.py');
    const child = spawn(cfg.python, [worker, '--input', input, '--script', cfg.script, '--model', cfg.model, '--library', cfg.library, '--events-dir', FIRE_EVENTS], {
      cwd: ROOT,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    job.pid = child.pid;
    job.status = 'running';
    let stdoutBuf = '';
    child.stdout.on('data', chunk => {
      stdoutBuf += chunk.toString();
      const lines = stdoutBuf.split(/\r?\n/); stdoutBuf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'progress') job.progress = msg.progress;
          if (msg.type === 'event') {
            const events = readEvents();
            const event = { ...msg, video: name, image_url: `/demo-media/fire-events/${encodeURIComponent(msg.image)}` };
            events.unshift(event); writeEvents(events); job.last_event = event;
          }
        } catch (_) { console.log(`[FireDemo] ${line}`); }
      }
    });
    child.stderr.on('data', chunk => console.error(`[FireDemo] ${chunk.toString().trim()}`));
    child.on('error', err => { job.status = 'error'; job.error = err.message; });
    child.on('close', code => {
      job.finished_at = new Date().toISOString();
      job.exit_code = code;
      job.status = code === 0 ? 'completed' : 'error';
      if (code === 0) job.progress = 100;
      else if (!job.error) job.error = `Detector exited with code ${code}`;
    });
    res.status(202).json(job);
  });
}

express.application.listen = function (...args) {
  attach(this);
  return originalListen.apply(this, args);
};
console.log('[Demo] Demo Videos + Fire Detection APIs enabled.');
