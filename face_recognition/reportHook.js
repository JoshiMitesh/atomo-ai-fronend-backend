const fs = require('fs');
const path = require('path');
const express = require('express');
const ExcelJS = require('exceljs');
const db = require('./db');

// -------------------------------------------------------------
// Report API middleware
// -------------------------------------------------------------
const originalUse = express.application.use;
let reportMiddlewareInstalled = false;

async function generateEventsWorkbook() {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'AURA Face Recognition';
  workbook.created = new Date();

  const sheet = workbook.addWorksheet('Event Report', {
    views: [{ state: 'frozen', ySplit: 1 }]
  });

  sheet.columns = [
    { header: 'Date', key: 'date', width: 14 },
    { header: 'Time', key: 'time', width: 14 },
    { header: 'Name', key: 'name', width: 24 },
    { header: 'Status', key: 'status', width: 18 },
    { header: 'Camera', key: 'camera', width: 24 },
    { header: 'Confidence', key: 'confidence', width: 14 },
    { header: 'Photo', key: 'photo', width: 20 }
  ];

  const header = sheet.getRow(1);
  header.font = { bold: true };
  header.alignment = { vertical: 'middle', horizontal: 'center' };
  header.height = 24;

  const events = db.getEvents(1000).slice().sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  for (const event of events) {
    const dt = new Date(event.timestamp);
    const status = event.is_known ? 'Authorized' : 'Unauthorized';
    const name = event.is_known ? (event.person_name || 'Known') : 'Unknown';
    const score = Number(event.score || 0);

    const row = sheet.addRow({
      date: Number.isNaN(dt.getTime()) ? '' : dt.toLocaleDateString('en-IN'),
      time: Number.isNaN(dt.getTime()) ? '' : dt.toLocaleTimeString('en-IN', { hour12: true }),
      name,
      status,
      camera: event.camera_name || 'Manual Upload',
      confidence: score
    });

    row.height = 82;
    row.alignment = { vertical: 'middle' };
    row.getCell('confidence').numFmt = '0%';

    if (event.crop_filename) {
      const cropPath = path.join(db.CROPS_DIR, path.basename(event.crop_filename));
      if (fs.existsSync(cropPath)) {
        try {
          const ext = path.extname(cropPath).toLowerCase();
          const extension = ext === '.png' ? 'png' : 'jpeg';
          const imageId = workbook.addImage({ filename: cropPath, extension });
          sheet.addImage(imageId, {
            tl: { col: 6.12, row: row.number - 0.88 },
            ext: { width: 88, height: 88 },
            editAs: 'oneCell'
          });
        } catch (err) {
          console.warn(`[Report] Could not embed ${event.crop_filename}: ${err.message}`);
        }
      }
    }
  }

  sheet.autoFilter = { from: 'A1', to: 'G1' };
  return workbook;
}

function reportMiddleware(req, res, next) {
  if (req.method !== 'GET' || req.path !== '/api/reports/events.xlsx') {
    return next();
  }

  generateEventsWorkbook()
    .then(async workbook => {
      const now = new Date();
      const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      const filename = `face-recognition-report-${stamp}.xlsx`;
      res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      await workbook.xlsx.write(res);
      res.end();
    })
    .catch(err => {
      console.error('[Report] Failed to generate report:', err);
      if (!res.headersSent) res.status(500).json({ error: 'Failed to generate Excel report.' });
      else res.end();
    });
}

express.application.use = function patchedUse(...args) {
  if (!reportMiddlewareInstalled) {
    reportMiddlewareInstalled = true;
    originalUse.call(this, reportMiddleware);
  }
  return originalUse.apply(this, args);
};

// -------------------------------------------------------------
// Settings page UI injection
// -------------------------------------------------------------
const previousStatic = express.static;

function injectReportUI(html) {
  const reportBlock = `
              <div class="setting-group" id="report-settings-group">
                <div class="setting-info">
                  <label class="setting-title">Event Report</label>
                  <p class="setting-desc">Download all saved recognition events as an Excel report with date, time, name, authorization status, camera, confidence and the actual captured face photo embedded in the sheet.</p>
                </div>
                <div class="setting-controls">
                  <button class="btn btn-primary" id="btn-generate-report">
                    <i class="fa-solid fa-file-excel"></i> Generate Report
                  </button>
                </div>
              </div>
`;

  if (!html.includes('id="btn-generate-report"')) {
    html = html.replace(
      '              <div class="settings-save-row">',
      reportBlock + '              <div class="settings-save-row">'
    );
  }

  const script = `
<script>
document.addEventListener('DOMContentLoaded', function () {
  const btn = document.getElementById('btn-generate-report');
  if (!btn) return;
  btn.addEventListener('click', function () {
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
    const iframe = document.createElement('iframe');
    iframe.style.display = 'none';
    iframe.src = '/api/reports/events.xlsx?t=' + Date.now();
    document.body.appendChild(iframe);
    setTimeout(function () {
      btn.disabled = false;
      btn.innerHTML = original;
      iframe.remove();
    }, 2500);
  });
});
</script>
`;

  if (!html.includes("'/api/reports/events.xlsx?t='")) {
    html = html.replace('</body>', script + '</body>');
  }
  return html;
}

express.static = function reportAwareStatic(root, options) {
  const normalStatic = previousStatic(root, options);
  const isPublicRoot = path.basename(path.resolve(root)) === 'public';
  if (!isPublicRoot) return normalStatic;

  return function reportStatic(req, res, next) {
    if (req.path === '/' || req.path === '/index.html') {
      try {
        const indexPath = path.join(root, 'index.html');
        const html = fs.readFileSync(indexPath, 'utf8');
        res.type('html').send(injectReportUI(html));
        return;
      } catch (err) {
        console.error('[Report] Failed to inject Settings report UI:', err.message);
      }
    }
    return normalStatic(req, res, next);
  };
};

console.log('[Report] Excel event reporting enabled.');
