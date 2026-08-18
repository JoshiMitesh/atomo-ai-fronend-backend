const path = require('path');
const express = require('express');
const originalStatic = express.static;

const browserScript = String.raw`(() => {
'use strict';
const state={screen:'home',videos:[],fireVideos:[],events:[],selectedFire:null,job:null,poll:null};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opts){const r=await fetch(url,opts);let b={};try{b=await r.json()}catch(_){b={}}if(!r.ok)throw new Error(b.error||('Request failed: '+r.status));return b}
function style(){if(document.getElementById('atomic-demo-style'))return;const s=document.createElement('style');s.id='atomic-demo-style';s.textContent=`
#atomic-demo-root{position:fixed;inset:0 0 0 265px;background:#090c12;color:#f4f6fb;z-index:8000;overflow:auto;padding:30px;font-family:inherit}.atomic-demo-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}.atomic-demo-head h1{font-size:28px;margin:0}.atomic-back,.atomic-btn{border:1px solid #2a3040;background:#151a25;color:#fff;border-radius:10px;padding:10px 15px;cursor:pointer}.atomic-btn.primary{background:#665cf6;border-color:#665cf6}.atomic-home-wrap{min-height:calc(100vh - 150px);display:flex;align-items:center;justify-content:center}.atomic-home-grid{width:min(760px,100%);display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:24px}.atomic-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px}.atomic-card{background:#121722;border:1px solid #242b3a;border-radius:16px;padding:20px;cursor:pointer;min-height:130px;transition:.18s ease}.atomic-card:hover{transform:translateY(-2px);border-color:#665cf6}.atomic-card h3{margin:8px 0;font-size:18px}.atomic-card p,.atomic-muted{color:#9ca5b7}.atomic-video-card video{width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;object-fit:cover}.atomic-tabs{display:flex;gap:10px;margin:0 0 20px}.atomic-tabs button{border:1px solid #2a3040;background:#151a25;color:#aeb5c4;padding:10px 16px;border-radius:10px;cursor:pointer}.atomic-tabs button.active{background:#665cf6;color:#fff}.atomic-event img{width:100%;aspect-ratio:16/10;object-fit:cover;border-radius:10px}.atomic-status{margin:14px 0;padding:12px 14px;border:1px solid #293044;border-radius:10px;background:#111620}.atomic-error{margin:14px 0;padding:12px 14px;border:1px solid rgba(239,68,68,.45);border-radius:10px;background:rgba(239,68,68,.08);color:#ff8b95}.atomic-modal{position:fixed;inset:0;background:rgba(0,0,0,.82);z-index:9000;display:flex;align-items:center;justify-content:center;padding:30px}.atomic-modal-box{width:min(1000px,92vw);background:#10141e;border:1px solid #2a3040;border-radius:16px;overflow:hidden}.atomic-modal-head{display:flex;justify-content:space-between;align-items:center;padding:16px 20px}.atomic-modal-head button{font-size:25px;background:none;border:0;color:white;cursor:pointer}.atomic-modal video{width:100%;max-height:75vh;background:#000;display:block}@media(max-width:850px){#atomic-demo-root{left:0;padding:18px}.atomic-home-grid,.atomic-grid{grid-template-columns:1fr}}
`;document.head.appendChild(s)}
function root(){let r=document.getElementById('atomic-demo-root');if(!r){r=document.createElement('section');r.id='atomic-demo-root';document.body.appendChild(r)}return r}
function closeDemo(){document.getElementById('atomic-demo-root')?.remove();clearInterval(state.poll);state.poll=null}
function header(title,back){return '<div class="atomic-demo-head"><div><h1>'+esc(title)+'</h1><div class="atomic-muted">Atomic Vision demonstration</div></div>'+(back?'<button class="atomic-back" data-action="home">← Back</button>':'<button class="atomic-back" data-action="close">Close</button>')+'</div>'}
function errorPage(title,msg){root().innerHTML=header(title,true)+'<div class="atomic-error">'+esc(msg)+'</div>'}
function renderHome(){state.screen='home';root().innerHTML=header('Demo',false)+'<div class="atomic-home-wrap"><div class="atomic-home-grid"><div class="atomic-card" data-action="videos"><h3>▶ Demo Videos</h3><p>Browse your demo videos and play any video in a popup.</p></div><div class="atomic-card" data-action="fire"><h3>🔥 Live Fire Detection</h3><p>Select a fire demo video, run NPU detection, and review detected events.</p></div></div></div>'}
async function renderVideos(){state.screen='videos';root().innerHTML=header('Demo Videos',true)+'<div class="atomic-status">Loading videos...</div>';try{state.videos=await api('/api/demo/videos');root().innerHTML=header('Demo Videos',true)+(state.videos.length?'<div class="atomic-grid">'+state.videos.map(v=>'<div class="atomic-card atomic-video-card" data-action="play" data-url="'+esc(v.url)+'" data-name="'+esc(v.name)+'"><video src="'+esc(v.url)+'" muted preload="metadata"></video><h3>'+esc(v.name)+'</h3><p>Click to play</p></div>').join('')+'</div>':'<div class="atomic-status">No demo videos found.<br><br>Put files in <b>face_recognition/demo_media/videos/</b> and refresh this page.</div>')}catch(e){errorPage('Demo Videos',e.message)}}
async function renderFire(tab='detect'){state.screen='fire';root().innerHTML=header('Live Fire Detection',true)+'<div class="atomic-status">Loading...</div>';try{[state.fireVideos,state.events]=await Promise.all([api('/api/demo/fire/videos'),api('/api/demo/fire/events')]);root().innerHTML=header('Live Fire Detection',true)+'<div class="atomic-tabs"><button data-action="fire-tab" data-tab="detect" class="'+(tab==='detect'?'active':'')+'">Fire Detection</button><button data-action="fire-tab" data-tab="events" class="'+(tab==='events'?'active':'')+'">Events ('+state.events.length+')</button></div>'+(tab==='events'?eventsHTML():fireHTML())}catch(e){errorPage('Live Fire Detection',e.message)}}
function fireHTML(){return (state.job?'<div class="atomic-status"><b>Status:</b> '+esc(state.job.status)+' &nbsp; <b>Progress:</b> '+Number(state.job.progress||0)+'%'+(state.job.error?'<br>'+esc(state.job.error):'')+'</div>':'')+(state.fireVideos.length?'<div class="atomic-grid">'+state.fireVideos.map(v=>'<div class="atomic-card atomic-video-card" data-action="select-fire" data-name="'+esc(v.name)+'"><video src="'+esc(v.url)+'" muted preload="metadata"></video><h3>'+esc(v.name)+'</h3><p>'+(state.selectedFire===v.name?'Selected':'Click to select')+'</p>'+(state.selectedFire===v.name?'<button class="atomic-btn primary" data-action="start-fire">Start Detection</button>':'')+'</div>').join('')+'</div>':'<div class="atomic-status">No fire videos found.<br><br>Put files in <b>face_recognition/demo_media/fire_videos/</b> and refresh this page.</div>')}
function eventsHTML(){if(!state.events.length)return '<div class="atomic-status">No fire/smoke events detected yet.</div>';return '<div class="atomic-grid">'+state.events.map(e=>{const d=new Date(e.timestamp);return '<div class="atomic-card atomic-event"><img src="'+esc(e.image_url)+'"><h3>'+esc(String(e.label||'fire').toUpperCase())+'</h3><p>'+esc(e.video||'')+'</p><p>'+esc(d.toLocaleDateString())+' • '+esc(d.toLocaleTimeString())+'</p></div>'}).join('')+'</div>'}
function openVideo(url,name){document.getElementById('atomic-demo-modal')?.remove();const m=document.createElement('div');m.className='atomic-modal';m.id='atomic-demo-modal';m.innerHTML='<div class="atomic-modal-box"><div class="atomic-modal-head"><b>'+esc(name)+'</b><button data-action="modal-close">×</button></div><video src="'+esc(url)+'" controls autoplay></video></div>';document.body.appendChild(m)}
async function startFire(){if(!state.selectedFire)return;try{state.job=await api('/api/demo/fire/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video:state.selectedFire})});await renderFire('detect');clearInterval(state.poll);state.poll=setInterval(async()=>{try{state.job=await api('/api/demo/fire/jobs/'+encodeURIComponent(state.job.id));if(['completed','error'].includes(state.job.status)){clearInterval(state.poll);state.poll=null}await renderFire('detect')}catch(e){clearInterval(state.poll);state.poll=null}},1000)}catch(e){alert(e.message)}}

document.addEventListener('click',function(e){
 const el=e.target.closest('[data-action]');if(!el)return;const action=el.dataset.action;
 if(action==='close'){e.preventDefault();closeDemo();return}
 if(action==='home'){e.preventDefault();renderHome();return}
 if(action==='videos'){e.preventDefault();renderVideos();return}
 if(action==='fire'){e.preventDefault();renderFire('detect');return}
 if(action==='play'){e.preventDefault();openVideo(el.dataset.url,el.dataset.name);return}
 if(action==='select-fire'){if(e.target.closest('[data-action="start-fire"]'))return;state.selectedFire=el.dataset.name;renderFire('detect');return}
 if(action==='start-fire'){e.preventDefault();e.stopPropagation();startFire();return}
 if(action==='fire-tab'){e.preventDefault();renderFire(el.dataset.tab);return}
 if(action==='modal-close'){e.preventDefault();document.getElementById('atomic-demo-modal')?.remove();return}
});
document.addEventListener('click',e=>{const m=document.getElementById('atomic-demo-modal');if(m&&e.target===m)m.remove()});
function findSettings(){return [...document.querySelectorAll('a,button,[role="button"],li')].find(x=>x.textContent.trim()==='Settings')}
function installNav(){if(document.querySelector('[data-atomic-demo-nav]'))return;const settings=findSettings();if(!settings)return;const item=settings.cloneNode(true);item.dataset.atomicDemoNav='1';item.dataset.action='open-demo';item.removeAttribute('href');item.querySelectorAll('[id]').forEach(x=>x.removeAttribute('id'));const walker=document.createTreeWalker(item,NodeFilter.SHOW_TEXT);let n;while(n=walker.nextNode()){if(n.nodeValue.trim()==='Settings')n.nodeValue=n.nodeValue.replace('Settings','Demo')}const icon=item.querySelector('i');if(icon)icon.className='fa-solid fa-flask';item.style.cursor='pointer';item.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();style();renderHome()});settings.parentNode.insertBefore(item,settings.nextSibling)}
style();installNav();new MutationObserver(installNav).observe(document.documentElement,{childList:true,subtree:true});
})();`;

express.static = function demoUiStatic(root, options) {
  const normalStatic = originalStatic(root, options);
  const isPublicRoot = path.basename(path.resolve(root)) === 'public';
  if (!isPublicRoot) return normalStatic;
  return function demoStatic(req, res, next) {
    if (req.path === '/demo-ui.js') {
      res.set('Cache-Control', 'no-store');
      res.type('application/javascript').send(browserScript);
      return;
    }
    normalStatic(req, res, next);
  };
};

const originalSend = express.response.send;
express.response.send = function patchedSend(body) {
  if (typeof body === 'string' && body.includes('</body>') && !body.includes('/demo-ui.js')) {
    body = body.replace('</body>', '<script src="/demo-ui.js?v=2"></script>\n</body>');
  }
  return originalSend.call(this, body);
};

console.log('[Demo] Centered Demo UI + reliable card navigation enabled.');
