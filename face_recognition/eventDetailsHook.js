const fs = require('fs');
const path = require('path');
const express = require('express');

// IMPORTANT: patch only /js/app.js. Never intercept index.html here.
const previousStatic = express.static;

function appendEventDetailsJs(source) {
  if (source.includes('function atomicOpenEventDetails(')) return source;

  return source + `

// -------------------------------------------------------------
// Atomic Vision - Live Event Details + Reliable Event Filters
// -------------------------------------------------------------
function atomicEnsureEventDetailsModal() {
  if (document.getElementById('atomic-event-details-modal')) return;

  const style = document.createElement('style');
  style.textContent = \`
    #events-list .event-item{cursor:pointer;transition:border-color .18s ease,transform .18s ease,background .18s ease}
    #events-list .event-item:hover{border-color:rgba(124,108,255,.55);background:rgba(124,108,255,.035);transform:translateY(-1px)}
    .atomic-event-modal{position:fixed;inset:0;z-index:99999;background:rgba(2,5,12,.76);display:flex;align-items:center;justify-content:center;padding:22px;backdrop-filter:blur(5px)}
    .atomic-event-modal.hidden{display:none!important}
    .atomic-event-dialog{width:min(720px,96vw);max-height:92vh;overflow:auto;background:#0f1420;border:1px solid var(--border-color);border-radius:18px;box-shadow:0 28px 90px rgba(0,0,0,.48)}
    .atomic-event-modal-header{display:flex;align-items:center;justify-content:space-between;padding:20px 22px;border-bottom:1px solid var(--border-color)}
    .atomic-event-modal-kicker{font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-muted);font-weight:700;margin-bottom:4px}
    .atomic-event-modal-header h3{margin:0;font-size:1.2rem}
    .atomic-event-close{width:38px;height:38px;border-radius:9px;border:1px solid var(--border-color);background:transparent;color:var(--text-muted);cursor:pointer;font-size:1rem}
    .atomic-event-close:hover{color:#fff;background:rgba(255,255,255,.05)}
    .atomic-event-modal-body{display:grid;grid-template-columns:minmax(220px,.9fr) minmax(270px,1.1fr);gap:22px;padding:22px}
    .atomic-event-photo-panel img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:14px;background:#080b12;border:1px solid var(--border-color)}
    .atomic-event-info-panel{display:flex;flex-direction:column}
    .atomic-event-status{display:inline-flex;align-self:flex-start;padding:7px 11px;border-radius:999px;font-size:.75rem;font-weight:800;letter-spacing:.04em;margin-bottom:12px}
    .atomic-event-status.authorised{color:#35e0aa;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.32)}
    .atomic-event-status.unauthorised{color:#ff6674;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.30)}
    .atomic-event-info-row{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;padding:14px 0;border-bottom:1px solid var(--border-color)}
    .atomic-event-info-row span{color:var(--text-muted);font-size:.82rem}
    .atomic-event-info-row strong{text-align:right;color:var(--text-light);font-size:.9rem;word-break:break-word}
    @media(max-width:620px){.atomic-event-modal{padding:10px}.atomic-event-modal-body{grid-template-columns:1fr}.atomic-event-dialog{max-height:96vh}.atomic-event-photo-panel img{max-height:330px}}
  \`;
  document.head.appendChild(style);

  const wrap = document.createElement('div');
  wrap.innerHTML = \`
    <div class="atomic-event-modal hidden" id="atomic-event-details-modal">
      <div class="atomic-event-dialog" role="dialog" aria-modal="true">
        <div class="atomic-event-modal-header">
          <div><div class="atomic-event-modal-kicker">Event Details</div><h3 id="atomic-event-title">Face Event</h3></div>
          <button class="atomic-event-close" id="atomic-event-close"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="atomic-event-modal-body">
          <div class="atomic-event-photo-panel"><img id="atomic-event-photo" src="" alt="Captured face event"></div>
          <div class="atomic-event-info-panel">
            <div class="atomic-event-status" id="atomic-event-status"></div>
            <div class="atomic-event-info-row"><span>Person</span><strong id="atomic-event-person">—</strong></div>
            <div class="atomic-event-info-row"><span>Zone / Camera</span><strong id="atomic-event-zone">—</strong></div>
            <div class="atomic-event-info-row"><span>Date</span><strong id="atomic-event-date">—</strong></div>
            <div class="atomic-event-info-row"><span>Time</span><strong id="atomic-event-time">—</strong></div>
          </div>
        </div>
      </div>
    </div>\`;
  document.body.appendChild(wrap.firstElementChild);
  document.getElementById('atomic-event-close').addEventListener('click', atomicCloseEventDetails);
  document.getElementById('atomic-event-details-modal').addEventListener('click', function(e){ if(e.target===this) atomicCloseEventDetails(); });
}

function atomicOpenEventDetails(ev) {
  if (!ev) return;
  atomicEnsureEventDetailsModal();
  if (typeof modalImagePreview !== 'undefined' && modalImagePreview) modalImagePreview.classList.add('hidden');
  const modal=document.getElementById('atomic-event-details-modal');
  const dt=ev.timestamp?new Date(ev.timestamp):null;
  const valid=dt&&!isNaN(dt.getTime());
  const known=ev.is_known===true;
  const photo=ev.crop_filename?'/crops/'+encodeURIComponent(ev.crop_filename):'https://placehold.co/500x500?text=Face';
  document.getElementById('atomic-event-photo').src=photo;
  document.getElementById('atomic-event-title').textContent=known?(ev.person_name||'Authorised Person'):(ev.person_name||'Unauthorised Profile');
  const status=document.getElementById('atomic-event-status');
  status.textContent=known?'AUTHORISED':'UNAUTHORISED';
  status.className='atomic-event-status '+(known?'authorised':'unauthorised');
  document.getElementById('atomic-event-person').textContent=ev.person_name||(known?'Known':'Unknown');
  document.getElementById('atomic-event-zone').textContent=ev.zone||ev.camera_name||'Manual Upload';
  document.getElementById('atomic-event-date').textContent=valid?dt.toLocaleDateString([], {year:'numeric',month:'short',day:'2-digit'}):'—';
  document.getElementById('atomic-event-time').textContent=valid?dt.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';
  modal.classList.remove('hidden');
}

function atomicCloseEventDetails(){
  const modal=document.getElementById('atomic-event-details-modal');
  if(modal)modal.classList.add('hidden');
  if(typeof modalImagePreview!=='undefined'&&modalImagePreview)modalImagePreview.classList.add('hidden');
}

function atomicApplyEventFilter(filterValue){
  currentEventFilter=filterValue;
  const buttons=document.querySelectorAll('.event-tab-btn');
  buttons.forEach(function(btn){
    const active=btn.getAttribute('data-event-filter')===filterValue;
    btn.classList.toggle('active',active);
    btn.style.color=active?'var(--text-light)':'var(--text-muted)';
  });
  filterAndRenderEvents();
}

document.addEventListener('DOMContentLoaded',function(){
  atomicEnsureEventDetailsModal();

  // Attach these independently from setupEventListeners(). This keeps the tabs
  // working even if another optional UI element is absent.
  document.querySelectorAll('.event-tab-btn').forEach(function(btn){
    btn.addEventListener('click',function(e){
      e.preventDefault();
      e.stopPropagation();
      atomicApplyEventFilter(btn.getAttribute('data-event-filter'));
    });
  });

  const list=document.getElementById('events-list');
  if(list){
    list.addEventListener('click',function(e){
      if(e.target.closest('.event-actions-dropdown,.btn-event-dots,.dropdown-menu,.dropdown-item'))return;
      const item=e.target.closest('.event-item');
      if(!item)return;
      e.preventDefault();e.stopPropagation();e.stopImmediatePropagation();
      const id=item.getAttribute('data-event-id');
      const ev=Array.isArray(allEvents)?allEvents.find(x=>String(x.id)===String(id)):null;
      if(ev)atomicOpenEventDetails(ev);
    },true);
  }
  document.addEventListener('keydown',function(e){if(e.key==='Escape')atomicCloseEventDetails();});
});
`;
}

express.static = function(root, options) {
  const normal = previousStatic(root, options);
  if (path.basename(path.resolve(root)) !== 'public') return normal;
  return function(req,res,next){
    if(req.path==='/js/app.js'){
      try{
        const source=fs.readFileSync(path.join(root,'js','app.js'),'utf8');
        res.type('application/javascript').send(appendEventDetailsJs(source));
        return;
      }catch(err){console.error('[UI] Event details/filter JS patch failed:',err.message);}
    }
    return normal(req,res,next);
  };
};

console.log('[UI] Live event popup + Authorised/Unauthorised filters fixed.');
