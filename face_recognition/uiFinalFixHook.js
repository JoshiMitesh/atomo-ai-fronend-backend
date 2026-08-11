const fs=require('fs');const path=require('path');const express=require('express');const previousStatic=express.static;
function patch(html){
  // Use the supplied Atomic Vision mark in the sidebar.
  html=html.replace(/<div class="logo-icon">[\s\S]*?<\/div>\s*<span class="logo-text atomic-brand">Atomic Vision<\/span>/,'<div class="logo-icon atomic-logo-wrap"><img src="/img/atomic-vision-logo.svg" class="atomic-logo-img" alt="Atomic Vision"></div><span class="logo-text atomic-brand">Atomic Vision</span>');

  // The database is a sibling section, so it needs the same sub-navigation.
  // This makes Discovered Profiles reachable after opening Face Database.
  if(!html.includes('database-profile-subtabs')){
    html=html.replace('<section id="tab-content-database" class="tab-content">','<section id="tab-content-database" class="tab-content"><div class="profiles-subtabs database-profile-subtabs"><button class="profile-subtab" data-profile-view="clusters"><i class="fa-solid fa-folder-open"></i> Discovered Profiles</button><button class="profile-subtab active" data-profile-view="database"><i class="fa-solid fa-users"></i> Face Database</button></div>');
  }

  // Report description must match the actual export: confidence is intentionally omitted.
  html=html.replace(/Export the newest 5,000 events to Excel with date, time, name, authorization status, camera, confidence and captured photo\./g,'Export the newest 5,000 events to Excel with date, time, name, authorization status, camera and captured photo.');
  html=html.replace(/Download the newest 5,000 recognition events as Excel with date, time, name, authorization status, camera, confidence and the actual captured face photo embedded in the sheet\./g,'Download the newest 5,000 recognition events as Excel with date, time, name, authorization status, camera and the actual captured face photo embedded in the sheet.');

  const css='<style>.atomic-logo-wrap{display:flex!important;align-items:center!important;justify-content:center!important;background:transparent!important;box-shadow:none!important;flex:0 0 42px!important;width:42px!important;height:42px!important}.atomic-logo-img{display:block;width:42px;height:42px;object-fit:contain;filter:invert(1);opacity:.96}.logo-container{gap:10px!important}.atomic-brand{white-space:nowrap!important}.database-profile-subtabs{margin-bottom:20px!important}@media(max-width:760px){.atomic-logo-wrap{width:38px!important;height:38px!important;flex-basis:38px!important}.atomic-logo-img{width:38px;height:38px}}</style>';
  html=html.replace('</head>',css+'</head>');
  return html;
}
express.static=function(root,options){const normal=previousStatic(root,options);if(path.basename(path.resolve(root))!=='public')return normal;return function(req,res,next){if(req.path==='/'||req.path==='/index.html'){try{res.type('html').send(patch(fs.readFileSync(path.join(root,'index.html'),'utf8')));return}catch(e){console.error('[UI Final Fix]',e.message)}}return normal(req,res,next)}};
console.log('[UI] Atomic Vision logo and persistent Profiles navigation enabled.');