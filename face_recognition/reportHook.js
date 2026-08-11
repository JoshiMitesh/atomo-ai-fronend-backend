const fs = require('fs');
const path = require('path');
const express = require('express');
const ExcelJS = require('exceljs');
const db = require('./db');

const originalUse = express.application.use;
let reportMiddlewareInstalled = false;
const REPORT_LIMIT = 5000;

async function generateEventsWorkbook() {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'Atomic Vision';
  workbook.created = new Date();
  const sheet = workbook.addWorksheet('Event Report', { views: [{ state: 'frozen', ySplit: 1 }] });
  sheet.columns = [
    { header: 'Date', key: 'date', width: 14 },
    { header: 'Time', key: 'time', width: 14 },
    { header: 'Name', key: 'name', width: 24 },
    { header: 'Status', key: 'status', width: 18 },
    { header: 'Camera', key: 'camera', width: 24 },
    { header: 'Photo', key: 'photo', width: 20 }
  ];
  const header = sheet.getRow(1);
  header.font = { bold: true };
  header.alignment = { vertical: 'middle', horizontal: 'center' };
  header.height = 24;

  const events = db.getEvents(REPORT_LIMIT).slice(0, REPORT_LIMIT).sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp));
  for (const event of events) {
    const dt = new Date(event.timestamp);
    const row = sheet.addRow({
      date: Number.isNaN(dt.getTime()) ? '' : dt.toLocaleDateString('en-IN'),
      time: Number.isNaN(dt.getTime()) ? '' : dt.toLocaleTimeString('en-IN', { hour12: true }),
      name: event.is_known ? (event.person_name || 'Known') : 'Unknown',
      status: event.is_known ? 'Authorized' : 'Unauthorized',
      camera: event.camera_name || 'Manual Upload'
    });
    row.height = 82;
    row.alignment = { vertical: 'middle' };

    if (event.crop_filename) {
      const cropPath = path.join(db.CROPS_DIR, path.basename(event.crop_filename));
      if (fs.existsSync(cropPath)) {
        try {
          const imageId = workbook.addImage({
            filename: cropPath,
            extension: path.extname(cropPath).toLowerCase() === '.png' ? 'png' : 'jpeg'
          });
          sheet.addImage(imageId, {
            tl: { col: 5.12, row: row.number - 0.88 },
            ext: { width: 88, height: 88 },
            editAs: 'oneCell'
          });
        } catch (err) {
          console.warn(`[Report] Could not embed ${event.crop_filename}: ${err.message}`);
        }
      }
    }
  }
  sheet.autoFilter = { from: 'A1', to: 'F1' };
  return workbook;
}

function reportMiddleware(req,res,next) {
  if (req.method !== 'GET' || req.path !== '/api/reports/events.xlsx') return next();
  generateEventsWorkbook().then(async workbook => {
    const now = new Date();
    const stamp = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
    res.setHeader('Content-Type','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition',`attachment; filename="atomic-vision-report-${stamp}.xlsx"`);
    await workbook.xlsx.write(res);
    res.end();
  }).catch(err => {
    console.error('[Report] Failed:',err);
    if(!res.headersSent) res.status(500).json({error:'Failed to generate Excel report.'});
    else res.end();
  });
}

express.application.use = function(...args) {
  if (!reportMiddlewareInstalled) {
    reportMiddlewareInstalled = true;
    originalUse.call(this,reportMiddleware);
  }
  return originalUse.apply(this,args);
};

const previousStatic = express.static;
function injectReportUI(html) {
  const block = `<div class="setting-group" id="report-settings-group"><div class="setting-info"><label class="setting-title">Event Report</label><p class="setting-desc">Download the newest 5,000 recognition events as Excel with date, time, name, authorization status, camera and the actual captured face photo embedded in the sheet.</p></div><div class="setting-controls"><button class="btn btn-primary" id="btn-generate-report"><i class="fa-solid fa-file-excel"></i> Generate Excel Report</button></div></div>`;
  if(!html.includes('id="btn-generate-report"')) html=html.replace('<div class="settings-save-row">',block+'<div class="settings-save-row">');
  const script=`<script>document.addEventListener('DOMContentLoaded',function(){const b=document.getElementById('btn-generate-report');if(!b)return;b.addEventListener('click',function(){const o=b.innerHTML;b.disabled=true;b.innerHTML='<i class="fa-solid fa-spinner fa-spin"></i> Generating...';const f=document.createElement('iframe');f.style.display='none';f.src='/api/reports/events.xlsx?t='+Date.now();document.body.appendChild(f);setTimeout(function(){b.disabled=false;b.innerHTML=o;f.remove();},3500);});});</script>`;
  if(!html.includes("'/api/reports/events.xlsx?t='")) html=html.replace('</body>',script+'</body>');
  return html;
}
express.static=function(root,options){const normal=previousStatic(root,options);if(path.basename(path.resolve(root))!=='public')return normal;return function(req,res,next){if(req.path==='/'||req.path==='/index.html'){try{res.type('html').send(injectReportUI(fs.readFileSync(path.join(root,'index.html'),'utf8')));return;}catch(e){console.error('[Report] UI injection failed:',e.message);}}return normal(req,res,next);};};
console.log('[Report] Excel event reporting enabled (maximum 5000 events, no confidence column).');
