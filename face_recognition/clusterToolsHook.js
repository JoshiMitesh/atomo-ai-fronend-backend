const express = require('express');
const db = require('./db');

const originalUse = express.application.use;
let installed = false;

function middleware(req, res, next) {
  if (req.method !== 'DELETE' || req.path !== '/api/clusters') return next();
  try {
    const clusters = db.getClusters();
    let deleted = 0;
    for (const cluster of clusters) {
      if (db.deleteCluster(cluster.id)) deleted++;
    }
    res.json({ success: true, deleted });
  } catch (err) {
    console.error('[Clusters] Failed to clear discovered profiles:', err);
    res.status(500).json({ success: false, error: 'Failed to clear discovered profiles.' });
  }
}

express.application.use = function(...args) {
  if (!installed) {
    installed = true;
    originalUse.call(this, middleware);
  }
  return originalUse.apply(this, args);
};

console.log('[Clusters] Clear-all discovered profiles API enabled.');
