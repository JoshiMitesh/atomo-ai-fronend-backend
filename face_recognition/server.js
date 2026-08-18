const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const db = require('./db');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Serve static frontend files
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(db.UPLOADS_DIR));
app.use('/crops', express.static(db.CROPS_DIR));

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, db.UPLOADS_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `upload_${Date.now()}_${Math.random().toString(36).substr(2, 5)}${ext}`);
  }
});
const upload = multer({
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }
});

// App settings store
let settings = {
  threshold: 0.50,
  dis_type: 0
};

// Load settings on startup if they exist in DB
function loadSettings() {
  try {
    const dbPath = path.join(db.DATA_DIR, 'database.json');
    if (fs.existsSync(dbPath)) {
      const data = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
      if (data.settings) {
        settings = { ...settings, ...data.settings };
      }
    }
  } catch (e) {
    console.error('Failed to load settings from DB:', e);
  }
}
loadSettings();

function saveSettings() {
  try {
    const dbPath = path.join(db.DATA_DIR, 'database.json');
    const data = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
    data.settings = settings;
    fs.writeFileSync(dbPath, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error('Failed to save settings to DB:', e);
  }
}

// NOTE: full original server implementation remains below in repository.
