const fs = require('fs');
const path = require('path');
const express = require('express');

// Final /js/app.js patch in the preload chain. It contains BOTH the incremental
// live-event update logic and Event Details popup so later hooks cannot bypass it.
const previousStatic = express.static;

function patchBaseApp(source) {
  let s = source;

  // Clear button is removed from HTML. Guard the old listener so the rest of
  // setupEventListeners(), including event filters, still initializes.
  s = s.replace(
`  // Clear Events logs
  btnClearEvents.addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear the entire events log history? This will also remove saved cropped faces.')) {
      try {
        await fetch('/api/events', { method: 'DELETE' });
      } catch (err) {
        console.error('Failed to clear events:', err);
      }
    }
  });`,
`  // Atomic Vision: live Clear button removed.
  if (btnClearEvents) {
    btnClearEvents.addEventListener('click', async () => {
      if (confirm('Are you sure you want to clear the entire events log history?')) {
        try { await fetch('/api/events', { method: 'DELETE' }); }
        catch (err) { console.error('Failed to clear events:', err); }
      }
    });
  }`
  );

  // Initial state and explicit fetches: keep only newest 100.
  s = s.replace('    allEvents = msg.data.events;', '    allEvents = Array.isArray(msg.data.events) ? msg.data.events.slice(0, 100) : [];');
  s = s.replace(/allEvents\s*=\s*await\s+res\.json\(\);/g, 'allEvents = (await res.json()).slice(0, 100);');
  s = s.replace(/allEvents\s*=\s*await\s+response\.json\(\);/g, 'allEvents = (await response.json()).slice(0, 100);');

  // Do not rebuild the event list because unrelated database/cluster state changed.
  s = s.replace(
`  } else if (msg.event === 'database_updated') {
    fetchPersons();
    fetchEvents();`,
`  } else if (msg.event === 'database_updated') {
    fetchPersons();`
  );
  s = s.replace(
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();
    fetchEvents();`,
`  } else if (msg.event === 'clusters_updated') {
    fetchClusters();`
  );

  // New event: add ONE card at the top instead of filterAndRenderEvents(),
  // which clears and recreates every card and causes the visible refresh/flicker.
  s = s.replace(
`function prependRecognitionEvent(ev) {
  allEvents.unshift(ev);
  if (allEvents.length > 100) allEvents.pop();
  filterAndRenderEvents();
}`,
`function prependRecognitionEvent(ev) {
  allEvents.unshift(ev);
  let removed = null;
  if (allEvents.length > 100) removed = allEvents.pop();

  const visible = currentEventFilter === 'all'
    || (currentEventFilter === 'known' && ev.is_known === true)
    || (currentEventFilter === 'unknown' && ev.is_known === false);

  if (visible) {
    const empty = eventsList.querySelector('.empty-state');
    if (empty) empty.remove();
    appendEventHTML(ev, true);
  }

  if (removed) {
    const old = eventsList.querySelector('[data-event-id="' + removed.id + '"]');
    if (old) old.remove();
  }
}`
  );

  // Recognition update: replace only that specific event card. This is usually
  // the Unknown -> Authorised update after SFace finishes.
  s = s.replace(
`function updateRecognitionEvent(ev) {
  const index = allEvents.findIndex(e => e.id === ev.id);
  if (index !== -1) {
    allEvents[index] = { ...allEvents[index], ...ev };
  }
  filterAndRenderEvents();
}`,
`function updateRecognitionEvent(ev) {
  const index = allEvents.findIndex(e => e.id === ev.id);
  if (index === -1) return;

  allEvents[index] = { ...allEvents[index], ...ev };
  const updated = allEvents[index];
  const visible = currentEventFilter === 'all'
    || (currentEventFilter === 'known' && updated.is_known === true)
    || (currentEventFilter === 'unknown' && updated.is_known === false);
  const existing = eventsList.querySelector('[data-event-id="' + updated.id + '"]');

  if (!visible) {
    if (existing) existing.remove();
    if (!eventsList.querySelector('.event-item')) {
      eventsList.innerHTML = '<div class="empty-state" id="events-empty-state"><i class="fa-solid fa-bell-slash"></i><p>No matching face events found.</p></div>';
    }
    return;
  }

  // Build one fresh card at the end, then move it into the existing card's
  // exact position. Other event DOM nodes are left completely untouched.
  appendEventHTML(updated, false);
  const replacement = eventsList.lastElementChild;
  if (existing && replacement && replacement !== existing) {
    existing.replaceWith(replacement);
  } else if (!existing && replacement) {
    eventsList.insertBefore(replacement, eventsList.firstChild);
  }
}`
  );

  return s;
}

function appendAtomicUi(source) {
  if (source.includes('function atomicOpenEventDetails(')) return source;
  return source + `

// Atomic Vision - Event Details + independent filter handlers
function atomicEnsureEventDetailsModal(){
  if(document.getElementById('atomic-event-details-modal'))return;
  const style=document.createElement('style');
  style.textContent='#events-list .event-item{cursor:pointer}.atomic-event-modal{position:fixed;inset:0;z-index:99999;background:rgba(2,5,12,.76);display:flex;align-items:center;justify-content:center;padding:22px;backdrop-filter:blur(5px)}.atomic-event-modal.hidden{display:none!important}.atomic-event-dialog{width:min(720px,96vw);max-height:92vh;overflow:auto;background:#0f1420;border:1px solid var(--border-color);border-radius:18px;box-shadow:0 28px 90px rgba(0,0,0,.48)}.atomic-event-modal-header{display:flex;justify-content:space-between;align-items:center;padding:20px 22px;border-bottom:1px solid var(--border-color)}.atomic-event-modal-kicker{font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-muted);font-weight:700;margin-bottom:4px}.atomic-event-modal-header h3{margin:0}.atomic-event-close{width:38px;height:38px;border-radius:9px;border:1px solid var(--border-color);background:transparent;color:var(--text-muted);cursor:pointer}.atomic-event-modal-body{display:grid;grid-template-columns:minmax(220px,.9fr) minmax(270px,1.1fr);gap:22px;padding:22px}.atomic-event-photo-panel img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:14px}.atomic-event-status{display:inline-flex;padding:7px 11px;border-radius:999px;font-size:.75rem;font-weight:800;margin-bottom:12px}.atomic-event-status.authorised{color:#35e0aa;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.32)}.atomic-event-status.unauthorised{color:#ff6674;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3)}.atomic-event-info-row{display:flex;justify-content:space-between;gap:18px;padding:14px 0;border-bottom:1px solid var(--border-color)}.atomic-event-info-row span{color:var(--text-muted);font-size:.82rem}.atomic-event-info-row strong{text-align:right}@media(max-width:620px){.atomic-event-modal-body{grid-template-columns:1fr}}';
  document.head.appendChild(style);
  const wrap=document.createElement('div');
  wrap.innerHTML='<div class="atomic-event-modal hidden" id="atomic-event-details-modal"><div class="atomic-event-dialog"><div class="atomic-event-modal-header"><div><div class="atomic-event-modal-kicker">Event Details</div><h3 id="atomic-event-title">Face Event</h3></div><button class="atomic-event-close" id="atomic-event-close"><i class="fa-solid fa-xmark"></i></button></div><div class="atomic-event-modal-body"><div class="atomic-event-photo-panel"><img id="atomic-event-photo"></div><div><div class="atomic-event-status" id="atomic-event-status"></div><div class="atomic-event-info-row"><span>Person</span><strong id="atomic-event-person">—</strong></div><div class="atomic-event-info-row"><span>Zone / Camera</span><strong id="atomic-event-zone">—</strong></div><div class="atomic-event-info-row"><span>Date</span><strong id="atomic-event-date">—</strong></div><div class="atomic-event-info-row"><span>Time</span><strong id="atomic-event-time">—</strong></div></div></div></div></div>';
  document.body.appendChild(wrap.firstElementChild);
  document.getElementById('atomic-event-close').onclick=atomicCloseEventDetails;
  document.getElementById('atomic-event-details-modal').onclick=function(e){if(e.target===this)atomicCloseEventDetails()};
}
function atomicOpenEventDetails(ev){
  if(!ev)return;atomicEnsureEventDetailsModal();
  if(typeof modalImagePreview!=='undefined'&&modalImagePreview)modalImagePreview.classList.add('hidden');
  const d=ev.timestamp?new Date(ev.timestamp):null,k=ev.is_known===true,valid=d&&!isNaN(d.getTime());
  document.getElementById('atomic-event-photo').src=ev.crop_filename?'/crops/'+encodeURIComponent(ev.crop_filename):'https://placehold.co/500x500?text=Face';
  document.getElementById('atomic-event-title').textContent=k?(ev.person_name||'Authorised Person'):(ev.person_name||'Unauthorised Profile');
  const st=document.getElementById('atomic-event-status');st.textContent=k?'AUTHORISED':'UNAUTHORISED';st.className='atomic-event-status '+(k?'authorised':'unauthorised');
  document.getElementById('atomic-event-person').textContent=ev.person_name||(k?'Known':'Unknown');
  document.getElementById('atomic-event-zone').textContent=ev.zone||ev.camera_name||'Manual Upload';
  document.getElementById('atomic-event-date').textContent=valid?d.toLocaleDateString([], {year:'numeric',month:'short',day:'2-digit'}):'—';
  document.getElementById('atomic-event-time').textContent=valid?d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}):'—';
  document.getElementById('atomic-event-details-modal').classList.remove('hidden');
}
function atomicCloseEventDetails(){const m=document.getElementById('atomic-event-details-modal');if(m)m.classList.add('hidden');if(typeof modalImagePreview!=='undefined'&&modalImagePreview)modalImagePreview.classList.add('hidden')}
function atomicApplyEventFilter(v){currentEventFilter=v;document.querySelectorAll('.event-tab-btn').forEach(b=>{const a=b.getAttribute('data-event-filter')===v;b.classList.toggle('active',a);b.style.color=a?'var(--text-light)':'var(--text-muted)'});filterAndRenderEvents()}
document.addEventListener('DOMContentLoaded',function(){
  atomicEnsureEventDetailsModal();
  document.querySelectorAll('.event-tab-btn').forEach(b=>b.addEventListener('click',function(e){e.preventDefault();e.stopPropagation();atomicApplyEventFilter(b.getAttribute('data-event-filter'))}));
  const list=document.getElementById('events-list');if(list)list.addEventListener('click',function(e){if(e.target.closest('.event-actions-dropdown,.btn-event-dots,.dropdown-menu,.dropdown-item'))return;const item=e.target.closest('.event-item');if(!item)return;e.preventDefault();e.stopImmediatePropagation();const ev=Array.isArray(allEvents)?allEvents.find(x=>String(x.id)===String(item.getAttribute('data-event-id'))):null;if(ev)atomicOpenEventDetails(ev)},true);
  document.addEventListener('keydown',e=>{if(e.key==='Escape')atomicCloseEventDetails()});
});
`;
}

express.static = function(root, options) {
  const normal = previousStatic(root, options);
  if (path.basename(path.resolve(root)) !== 'public') return normal;
  return function(req,res,next){
    if(req.path==='/js/app.js'){
      try{
        let source=fs.readFileSync(path.join(root,'js','app.js'),'utf8');
        source=patchBaseApp(source);
        source=appendAtomicUi(source);
        res.type('application/javascript').send(source);
        return;
      }catch(err){console.error('[UI] Final Atomic Vision app.js patch failed:',err.message);}
    }
    return normal(req,res,next);
  };
};

console.log('[UI] Live events incremental updates enabled: no full-list refresh/flicker.');
