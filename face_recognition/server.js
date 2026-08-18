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

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(db.UPLOADS_DIR));
app.use('/crops', express.static(db.CROPS_DIR));

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, db.UPLOADS_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `upload_${Date.now()}_${Math.random().toString(36).substr(2, 5)}${ext}`);
  }
});
const upload = multer({ storage, limits: { fileSize: 100 * 1024 * 1024 } });

let settings = { threshold: 0.50, dis_type: 0 };
function loadSettings() {
  try {
    const dbPath = path.join(db.DATA_DIR, 'database.json');
    if (fs.existsSync(dbPath)) {
      const data = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
      if (data.settings) settings = { ...settings, ...data.settings };
    }
  } catch (e) { console.error('Failed to load settings from DB:', e); }
}
loadSettings();
function saveSettings() {
  try {
    const dbPath = path.join(db.DATA_DIR, 'database.json');
    const data = JSON.parse(fs.readFileSync(dbPath, 'utf8'));
    data.settings = settings;
    fs.writeFileSync(dbPath, JSON.stringify(data, null, 2));
  } catch (e) { console.error('Failed to save settings to DB:', e); }
}

let mediamtxProcess = null;
function startMediaMTX() {
  console.log('Starting MediaMTX RTSP Server...');
  mediamtxProcess = spawn('./mediamtx', [], { cwd: __dirname });
  mediamtxProcess.stdout.on('data', data => {
    const logStr = data.toString().trim();
    if (logStr.includes('ERR') || logStr.includes('warn')) console.log(`[MediaMTX] ${logStr}`);
  });
  mediamtxProcess.stderr.on('data', data => console.error(`[MediaMTX Err] ${data.toString().trim()}`));
  mediamtxProcess.on('close', code => {
    if (code !== 0 && code !== null) {
      console.error(`MediaMTX process closed with code ${code}. Re-spawning in 5 seconds...`);
      setTimeout(startMediaMTX, 5000);
    }
  });
}
startMediaMTX();

const ffmpegProcesses = new Map();
function startMockFFmpegStream(cameraId, videoPath) {
  stopMockFFmpegStream(cameraId);
  console.log(`Starting FFmpeg loop stream for camera [${cameraId}] using file: ${videoPath}`);
  const rtspUrl = `rtsp://127.0.0.1:8554/${cameraId}`;
  const ffmpegProcess = spawn('ffmpeg', ['-re','-stream_loop','-1','-i',videoPath,'-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-an','-f','rtsp','-rtsp_transport','tcp',rtspUrl]);
  ffmpegProcess.on('close', code => { console.log(`FFmpeg loop stream for camera [${cameraId}] exited with code ${code}`); ffmpegProcesses.delete(cameraId); });
  ffmpegProcesses.set(cameraId, ffmpegProcess);
  return rtspUrl;
}
function stopMockFFmpegStream(cameraId) {
  if (ffmpegProcesses.has(cameraId)) {
    console.log(`Stopping FFmpeg loop stream for camera [${cameraId}]`);
    const proc = ffmpegProcesses.get(cameraId); proc.kill('SIGKILL'); ffmpegProcesses.delete(cameraId);
  }
}

const transcoderProcesses = new Map();
function getStreamCodec(rtspUrl) {
  return new Promise(resolve => {
    const cmd = `ffprobe -v error -rtsp_transport tcp -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${rtspUrl}"`;
    exec(cmd, { timeout: 5000 }, (err, stdout) => {
      if (err) { console.warn(`[Codec Detector] ffprobe failed for URL: ${rtspUrl}, error: ${err.message}`); resolve('unknown'); }
      else resolve(stdout.trim().toLowerCase());
    });
  });
}
function startTranscoder(cameraId, rtspUrl) {
  stopTranscoder(cameraId);
  console.log(`[Transcoder] Starting RTSP transcoding to H.264 for camera ${cameraId}: ${rtspUrl}`);
  const targetRtspUrl = `rtsp://127.0.0.1:8554/${cameraId}`;
  const ffmpegProcess = spawn('ffmpeg', ['-allowed_media_types','video','-rtsp_transport','tcp','-fflags','nobuffer','-probesize','100000','-analyzeduration','0','-i',rtspUrl,'-vf','scale=960:-2,fps=8','-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-bf','0','-g','8','-crf','22','-threads','1','-an','-f','rtsp','-rtsp_transport','tcp',targetRtspUrl]);
  ffmpegProcess.stderr.on('data', data => console.error(`[Transcoder-${cameraId} Err] ${data.toString().trim()}`));
  ffmpegProcess.on('close', code => { console.log(`[Transcoder] FFmpeg transcoding for camera ${cameraId} exited with code ${code}`); transcoderProcesses.delete(cameraId); });
  transcoderProcesses.set(cameraId, ffmpegProcess);
}
function stopTranscoder(cameraId) {
  if (transcoderProcesses.has(cameraId)) {
    console.log(`[Transcoder] Stopping transcoder process for camera ${cameraId}`);
    const proc = transcoderProcesses.get(cameraId);
    try { proc.kill('SIGKILL'); } catch (e) { console.error(`[Transcoder] Failed to kill process for camera ${cameraId}:`, e.message); }
    transcoderProcesses.delete(cameraId);
  }
}

async function startCameraStream(camera) {
  const isLocal = camera.rtsp_url.startsWith('rtsp://localhost') || camera.rtsp_url.startsWith('rtsp://127.0.0.1');
  if (!isLocal) {
    await unregisterMediaMtxPath(camera.id);
    const codec = await getStreamCodec(camera.rtsp_url);
    console.log(`[Server] Detected codec for camera ${camera.name} (${camera.id}): ${codec}`);
    if (codec === 'hevc' || codec === 'h265') {
      console.log('[Server] Stream uses H.265. Starting on-the-fly H.264 transcoding...');
      await registerMediaMtxPublisherPath(camera.id);
      startTranscoder(camera.id, camera.rtsp_url);
      await new Promise(r => setTimeout(r, 1500));
    } else {
      console.log(`[Server] Stream uses H.264 or unknown (${codec}). Proxying directly through MediaMTX...`);
      await registerMediaMtxPath(camera.id, camera.rtsp_url);
    }
  }
  const targetRtspUrl = isLocal ? camera.rtsp_url : `rtsp://127.0.0.1:8554/${camera.id}`;
  console.log(`[Server] Starting stream thread in Python worker for ${camera.name} using URL: ${targetRtspUrl}`);
  return await sendPythonCommand({cmd:'start_stream',camera_id:camera.id,camera_name:camera.name,rtsp_url:targetRtspUrl,candidates:db.getAllCandidates(),threshold:settings.threshold,dis_type:settings.dis_type,crops_dir:db.CROPS_DIR,line_crossing_enabled:!!camera.line_crossing_enabled,line_y:camera.line_y!==undefined?camera.line_y:0.6,line_direction:camera.line_direction||'in',line_x_start:camera.line_x_start!==undefined?camera.line_x_start:0.0,line_x_end:camera.line_x_end!==undefined?camera.line_x_end:1.0});
}
async function stopCameraStream(camera) {
  try { await sendPythonCommand({cmd:'stop_stream',camera_id:camera.id}); }
  catch (err) { console.warn(`[Server] Python stop_stream error for camera ${camera.id}:`, err.message); }
  stopMockFFmpegStream(camera.id); stopTranscoder(camera.id);
  const isLocal = camera.rtsp_url.startsWith('rtsp://localhost') || camera.rtsp_url.startsWith('rtsp://127.0.0.1');
  if (!isLocal) await unregisterMediaMtxPath(camera.id);
}

let pythonProcess=null,isPythonReady=false; const pendingRequests=[]; let stdoutBuffer='';
function startPythonWorker(){
  console.log('Starting Python face worker process...'); isPythonReady=false;
  pythonProcess=spawn('python3',['-u','face_worker.py'],{cwd:__dirname,env:{...process.env,PYTHONUNBUFFERED:'1'}});
  pythonProcess.stdout.on('data',data=>{stdoutBuffer+=data.toString();let boundary=stdoutBuffer.indexOf('\n');while(boundary!==-1){const line=stdoutBuffer.slice(0,boundary).trim();stdoutBuffer=stdoutBuffer.slice(boundary+1);boundary=stdoutBuffer.indexOf('\n');if(!line)continue;try{handlePythonMessage(JSON.parse(line));}catch(err){console.error('Error parsing Python JSON:',err);console.error('Line was:',line);}}});
  pythonProcess.stderr.on('data',data=>process.stderr.write(`[Python] ${data.toString()}`));
  pythonProcess.on('close',code=>{console.error(`Python process exited with code ${code}. Re-spawning in 3 seconds...`);isPythonReady=false;while(pendingRequests.length>0){const req=pendingRequests.shift();req.reject(new Error('Python worker process exited.'));}setTimeout(startPythonWorker,3000);});
}
function sendPythonCommand(cmdObj){return new Promise((resolve,reject)=>{if(!pythonProcess||pythonProcess.killed)return reject(new Error('Python process is not running.'));pendingRequests.push({resolve,reject,cmd:cmdObj.cmd});pythonProcess.stdin.write(JSON.stringify(cmdObj)+'\n');});}

async function registerMediaMtxPath(cameraId,rtspUrl){if(rtspUrl.startsWith('rtsp://localhost')||rtspUrl.startsWith('rtsp://127.0.0.1')){console.log(`[MediaMTX] Camera ${cameraId} is a localhost/mock stream. No proxy configuration needed.`);return;}console.log(`[MediaMTX] Registering remote RTSP camera ${cameraId} with URL: ${rtspUrl}`);try{const response=await fetch(`http://localhost:9997/v3/config/paths/add/${cameraId}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:rtspUrl,sourceOnDemand:false,sourceProtocol:'tcp'})});if(response.ok)console.log(`[MediaMTX] Successfully configured remote path for ${cameraId}`);else console.warn(`[MediaMTX] Path add API returned status ${response.status}`);}catch(err){console.error('[MediaMTX] Error registering path:',err.message);}}
async function registerMediaMtxPublisherPath(cameraId){console.log(`[MediaMTX] Registering publisher path config for camera ${cameraId}`);try{const response=await fetch(`http://localhost:9997/v3/config/paths/add/${cameraId}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:'publisher'})});if(response.ok)console.log(`[MediaMTX] Successfully configured publisher path for ${cameraId}`);else console.warn(`[MediaMTX] Publisher path add API returned status ${response.status}`);}catch(err){console.error('[MediaMTX] Error registering publisher path:',err.message);}}
async function unregisterMediaMtxPath(cameraId){console.log(`[MediaMTX] Unregistering path config for camera ${cameraId}`);try{const response=await fetch(`http://localhost:9997/v3/config/paths/delete/${cameraId}`,{method:'POST'});if(response.ok)console.log(`[MediaMTX] Successfully deleted path for ${cameraId}`);else console.warn(`[MediaMTX] Path delete API returned status ${response.status}`);}catch(err){console.error('[MediaMTX] Error deleting path:',err.message);}}

async function resumeActiveStreams(){const activeCameras=db.getCameras().filter(c=>c.is_active);if(activeCameras.length===0)return;console.log(`[Server] Python worker ready. Auto-resuming ${activeCameras.length} active camera streams...`);for(const cam of activeCameras){try{await startCameraStream(cam);}catch(err){console.error(`[Server] Failed to auto-resume camera ${cam.name}:`,err.message);}}}

const activeEvents=new Map(),lastAutoLearnTimes=new Map();
setInterval(()=>{const now=Date.now();for(const [uuid] of activeEvents.entries()){const parts=uuid.split('_');if(parts.length>=2){const ts=parseInt(parts[1],10);if(ts&&(now-ts)>30000)activeEvents.delete(uuid);}}},60000);
function tryAutoLearnPerson(personId,personName,cropFilename,embedding,score){if(!personId||personId==='UNKNOWN'||!embedding||!cropFilename)return;const now=Date.now(),lastTime=lastAutoLearnTimes.get(personId)||0;if(now-lastTime<60000)return;const person=db.getPerson(personId);if(!person)return;const autoPhotos=person.photos.filter(p=>p.filename.startsWith('auto_'));if(autoPhotos.length>=10)return;const candidates=db.getAllCandidates().find(c=>c.person_id===personId);if(candidates&&candidates.embeddings){for(const refEmb of candidates.embeddings){let dot=0,normA=0,normB=0;for(let i=0;i<embedding.length;i++){dot+=embedding[i]*refEmb[i];normA+=embedding[i]*embedding[i];normB+=refEmb[i]*refEmb[i];}const sim=(normA===0||normB===0)?0:dot/(Math.sqrt(normA)*Math.sqrt(normB));if(sim>=0.88)return;}}const cropPath=path.join(db.CROPS_DIR,cropFilename),uploadFilename=`auto_${Date.now()}_${path.basename(cropFilename)}`,uploadPath=path.join(db.UPLOADS_DIR,uploadFilename);try{if(fs.existsSync(cropPath)){fs.copyFileSync(cropPath,uploadPath);db.addPhoto(personId,uploadFilename,embedding);lastAutoLearnTimes.set(personId,Date.now());console.log(`[Auto-Learning] Enrolled new distinct template for ${personName} (Score: ${score.toFixed(3)})`);sendPythonCommand({cmd:'add_template',person_id:personId,embedding,name:personName}).catch(err=>console.error('Error sending template to Python:',err.message));broadcastDatabaseUpdate();}}catch(err){console.error('[Auto-Learning] Error saving template:',err.message);}}

function handlePythonMessage(msg){
  if(msg.event){
    if(msg.event==='ready'){console.log('Python face worker is ready.');db.deduplicateAutoPhotos();isPythonReady=true;sendCandidatesToPython();resumeActiveStreams();}
    else if(msg.event==='video_enroll_face'){try{const photo=db.addPhoto(msg.person_id,msg.filename,msg.embedding);broadcast({event:'video_face_enrolled',person_id:msg.person_id,photo:{id:photo.id,filename:photo.filename}});sendCandidatesToPython();console.log(`[Video Enrollment] Dynamically enrolled face ${msg.filename} for person ${msg.person_id}`);}catch(err){console.error('[Video Enrollment] Failed to dynamically enroll face:',err);}}
    else if(msg.event==='stream_frame')broadcast({event:'live_frame',camera_id:msg.camera_id,frame:msg.frame});
    else if(msg.event==='stream_detect'){const isKnown=!!msg.is_known;let personId='UNKNOWN',personName='UNKNOWN',score=msg.score||0.0;if(isKnown&&msg.match){personId=msg.match.person_id;personName=msg.match.name;}const savedEvent=db.addEvent(personId,personName,score,msg.crop_filename,isKnown,msg.camera_id,msg.camera_name);activeEvents.set(msg.event_uuid,savedEvent.id);broadcast({event:'recognition_event',data:savedEvent});}
    else if(msg.event==='stream_recognize'){const eventId=activeEvents.get(msg.event_uuid);console.log(`[Recognition] Received stream_recognize: is_known=${msg.is_known}, score=${msg.score?msg.score.toFixed(3):0.0}, match=${msg.match?JSON.stringify(msg.match):'None'}`);if(eventId){activeEvents.delete(msg.event_uuid);let personId='UNKNOWN',personName='UNKNOWN',isKnownEvent=msg.is_known;if(msg.is_known){personId=msg.match.person_id;personName=msg.match.name;}else if(msg.embedding){try{const clusterResult=db.addFaceToCluster(msg.embedding,msg.crop_filename,null,settings);personId=clusterResult.id;personName=clusterResult.name;broadcast({event:'clusters_updated'});}catch(err){console.error('[Clustering] Error clustering face:',err);}}const updatedEvent=db.updateEvent(eventId,{person_id:personId,person_name:personName,score:msg.score,is_known:isKnownEvent});if(updatedEvent){broadcast({event:'recognition_update',data:updatedEvent});if(msg.is_known&&msg.score>=Math.max(0.52,settings.threshold)&&msg.embedding)tryAutoLearnPerson(msg.match.person_id,msg.match.name,msg.crop_filename,msg.embedding,msg.score);}}}
  } else if(msg.cmd){const index=pendingRequests.findIndex(r=>r.cmd===msg.cmd);if(index!==-1){const req=pendingRequests.splice(index,1)[0];req.resolve(msg.response!==undefined?msg.response:msg);}}
}
function sendCandidatesToPython(){if(!isPythonReady)return;sendPythonCommand({cmd:'update_candidates',candidates:db.getAllCandidates()}).catch(err=>console.error('Failed to update candidate embeddings in Python:',err.message));}
startPythonWorker();

function broadcast(data){const message=JSON.stringify(data);wss.clients.forEach(client=>{if(client.readyState===WebSocket.OPEN)client.send(message);});}
function broadcastDatabaseUpdate(){broadcast({event:'database_updated'});}
wss.on('connection',ws=>{ws.send(JSON.stringify({event:'connection_init',data:{status:{threshold:settings.threshold,dis_type:settings.dis_type},cameras:db.getCameras(),events:db.getEvents(30)}}));});

app.get('/api/persons',(req,res)=>res.json(db.getPersons()));
app.post('/api/persons',(req,res)=>{const{name,gender}=req.body;if(!name||name.trim()==='')return res.status(400).json({error:'Person name is required.'});const person=db.addPerson(name.trim(),gender);broadcastDatabaseUpdate();res.status(201).json(person);});
app.delete('/api/persons/:id',(req,res)=>{const success=db.deletePerson(req.params.id);if(success){sendCandidatesToPython();broadcastDatabaseUpdate();res.json({success:true,message:'Person deleted.'});}else res.status(404).json({error:'Person not found.'});});
app.post('/api/persons/:id/photos',upload.array('photos'),async(req,res)=>{const personId=req.params.id,files=req.files;if(!files||files.length===0)return res.status(400).json({error:'No photos uploaded.'});const person=db.getPerson(personId);if(!person){files.forEach(f=>{try{fs.unlinkSync(f.path);}catch(e){}});return res.status(404).json({error:'Person not found.'});}const addedPhotos=[],errors=[];for(const file of files){try{const response=await sendPythonCommand({cmd:'extract_embedding',img_path:file.path});if(response.status==='success'){const filename=path.basename(file.path),photo=db.addPhoto(personId,filename,response.embedding);addedPhotos.push({id:photo.id,filename:photo.filename});}else{errors.push(`File ${file.originalname}: ${response.message}`);try{fs.unlinkSync(file.path);}catch(e){}}}catch(err){errors.push(`File ${file.originalname}: ${err.message}`);try{fs.unlinkSync(file.path);}catch(e){}}}sendCandidatesToPython();broadcastDatabaseUpdate();res.json({success:addedPhotos.length>0,added:addedPhotos,errors});});
app.post('/api/persons/:id/video',upload.single('video'),async(req,res)=>{const personId=req.params.id,file=req.file;if(!file)return res.status(400).json({error:'No video file uploaded.'});const person=db.getPerson(personId);if(!person){try{fs.unlinkSync(file.path);}catch(e){}return res.status(404).json({error:'Person not found.'});}try{const response=await sendPythonCommand({cmd:'process_video_enrollment',person_id:personId,video_path:file.path,crops_dir:db.UPLOADS_DIR});try{fs.unlinkSync(file.path);}catch(e){}if(response.status==='success'){const addedPhotos=[];for(const face of response.faces||[]){const p=db.getPerson(personId),alreadyExists=p&&p.photos.some(ph=>ph.filename===face.filename);if(!alreadyExists){const photo=db.addPhoto(personId,face.filename,face.embedding);addedPhotos.push({id:photo.id,filename:photo.filename});}else{const existingPhoto=p.photos.find(ph=>ph.filename===face.filename);if(existingPhoto)addedPhotos.push({id:existingPhoto.id,filename:existingPhoto.filename});}}sendCandidatesToPython();broadcastDatabaseUpdate();res.json({success:true,addedCount:addedPhotos.length,added:addedPhotos});}else res.status(500).json({error:response.message||'Failed to process video.'});}catch(err){try{fs.unlinkSync(file.path);}catch(e){}res.status(500).json({error:err.message});}});
app.delete('/api/photos/:id',(req,res)=>{const success=db.deletePhoto(req.params.id);if(success){sendCandidatesToPython();broadcastDatabaseUpdate();res.json({success:true,message:'Photo deleted.'});}else res.status(404).json({error:'Photo not found.'});});

app.get('/api/clusters',(req,res)=>res.json(db.getClusters()));
app.post('/api/clusters/:id/enroll',async(req,res)=>{const clusterId=req.params.id,{name}=req.body;if(!name||name.trim()==='')return res.status(400).json({error:'Person name is required.'});try{const person=db.enrollCluster(clusterId,name);sendCandidatesToPython();broadcastDatabaseUpdate();broadcast({event:'clusters_updated'});res.json({success:true,person});}catch(err){res.status(500).json({error:err.message});}});
app.post('/api/clusters/:clusterId/photos/:photoId/move',async(req,res)=>{const{clusterId,photoId}=req.params,{personId,name}=req.body;if(!personId&&(!name||name.trim()===''))return res.status(400).json({error:'Person ID or Name is required.'});try{const photo=db.movePhotoFromClusterToPerson(clusterId,photoId,personId,name);sendCandidatesToPython();broadcastDatabaseUpdate();broadcast({event:'clusters_updated'});res.json({success:true,photo});}catch(err){res.status(500).json({error:err.message});}});
app.delete('/api/clusters/:id',(req,res)=>{const success=db.deleteCluster(req.params.id);if(success){broadcast({event:'clusters_updated'});res.json({success:true});}else res.status(404).json({error:'Cluster not found.'});});

app.post('/api/recognize',upload.single('photo'),async(req,res)=>{if(!req.file)return res.status(400).json({error:'No photo uploaded.'});try{const response=await sendPythonCommand({cmd:'recognize_image',img_path:req.file.path,candidates:db.getAllCandidates(),threshold:settings.threshold,dis_type:settings.dis_type,crops_dir:db.CROPS_DIR});if(response.status==='success'){response.faces.forEach(face=>{let personId='UNKNOWN',personName=face.gender?`UNKNOWN (${face.gender})`:'UNKNOWN',isKnownEvent=face.is_known;if(face.is_known){const person=db.getPerson(face.match.person_id),gender=(person&&person.gender&&person.gender!=='Unknown')?person.gender:(face.gender||'Unknown'),genderSuffix=gender!=='Unknown'?` (${gender})`:'';personId=face.match.person_id;personName=`${face.match.name}${genderSuffix}`;face.match.name=personName;}else if(face.embedding){try{const clusterResult=db.addFaceToCluster(face.embedding,face.crop_filename,face.gender,settings);personId=clusterResult.id;personName=clusterResult.name;broadcast({event:'clusters_updated'});face.cluster_id=clusterResult.id;face.cluster_name=clusterResult.name;}catch(err){console.error('[Clustering] Error clustering face:',err);}}const savedEvent=db.addEvent(personId,personName,face.score,face.crop_filename,isKnownEvent,null,'Manual Upload');broadcast({event:'recognition_event',data:savedEvent});if(face.is_known&&face.score>=Math.max(0.52,settings.threshold)&&face.embedding)tryAutoLearnPerson(face.match.person_id,face.match.name,face.crop_filename,face.embedding,face.score);});try{fs.unlinkSync(req.file.path);}catch(e){}res.json({success:true,faces:response.faces});}else{try{fs.unlinkSync(req.file.path);}catch(e){}res.status(500).json({error:response.message});}}catch(err){try{fs.unlinkSync(req.file.path);}catch(e){}res.status(500).json({error:err.message});}});

app.get('/api/cameras',(req,res)=>res.json(db.getCameras()));
app.post('/api/cameras',(req,res)=>{const{name,rtsp_url}=req.body;if(!name||!rtsp_url)return res.status(400).json({error:'Name and RTSP URL are required.'});const camera=db.addCamera(name.trim(),rtsp_url.trim());broadcast({event:'cameras_updated',data:db.getCameras()});res.status(201).json(camera);});
app.delete('/api/cameras/:id',async(req,res)=>{const cameraId=req.params.id,camera=db.getCamera(cameraId);if(camera){if(camera.is_active)await stopCameraStream(camera);db.deleteCamera(cameraId);broadcast({event:'cameras_updated',data:db.getCameras()});res.json({success:true,message:'Camera deleted.'});}else res.status(404).json({error:'Camera not found.'});});
app.post('/api/cameras/:id/toggle',async(req,res)=>{const cameraId=req.params.id,camera=db.getCamera(cameraId);if(!camera)return res.status(404).json({error:'Camera not found.'});const activate=!camera.is_active;try{if(activate){const response=await startCameraStream(camera);if(response.status==='success'){db.updateCameraStatus(cameraId,true);broadcast({event:'cameras_updated',data:db.getCameras()});res.json({success:true,message:'Camera stream started.'});}else{await stopCameraStream(camera);res.status(500).json({error:response.message});}}else{await stopCameraStream(camera);db.updateCameraStatus(cameraId,false);broadcast({event:'cameras_updated',data:db.getCameras()});res.json({success:true,message:'Camera stream stopped.'});}}catch(err){res.status(500).json({error:err.message});}});
app.post('/api/cameras/:id/line-settings',async(req,res)=>{const cameraId=req.params.id,{line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end}=req.body;const camera=db.updateCameraLineSettings(cameraId,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end);if(!camera)return res.status(404).json({error:'Camera not found.'});if(camera.is_active){try{const isLocal=camera.rtsp_url.startsWith('rtsp://localhost')||camera.rtsp_url.startsWith('rtsp://127.0.0.1'),targetRtspUrl=isLocal?camera.rtsp_url:`rtsp://127.0.0.1:8554/${camera.id}`;await sendPythonCommand({cmd:'start_stream',camera_id:camera.id,camera_name:camera.name,rtsp_url:targetRtspUrl,candidates:db.getAllCandidates(),threshold:settings.threshold,dis_type:settings.dis_type,crops_dir:db.CROPS_DIR,line_crossing_enabled:camera.line_crossing_enabled,line_y:camera.line_y,line_direction:camera.line_direction,line_x_start:camera.line_x_start!==undefined?camera.line_x_start:0.0,line_x_end:camera.line_x_end!==undefined?camera.line_x_end:1.0});}catch(e){console.error(`Failed to update settings for active camera ${camera.id}:`,e.message);}}broadcast({event:'cameras_updated',data:db.getCameras()});res.json({success:true,camera});});
app.post('/api/cameras/upload-mock',upload.single('video'),async(req,res)=>{if(!req.file)return res.status(400).json({error:'No video file uploaded.'});try{const originalName=path.basename(req.file.originalname,path.extname(req.file.originalname)),name=`Mock Stream (${originalName})`,camera=db.addCamera(name,'temp'),rtspUrl=startMockFFmpegStream(camera.id,req.file.path);camera.rtsp_url=rtspUrl;const dbData=JSON.parse(fs.readFileSync(path.join(db.DATA_DIR,'database.json'),'utf8')),dbCam=dbData.cameras.find(c=>c.id===camera.id);if(dbCam){dbCam.rtsp_url=rtspUrl;fs.writeFileSync(path.join(db.DATA_DIR,'database.json'),JSON.stringify(dbData,null,2));}const response=await sendPythonCommand({cmd:'start_stream',camera_id:camera.id,camera_name:camera.name,rtsp_url:rtspUrl,candidates:db.getAllCandidates(),threshold:settings.threshold,dis_type:settings.dis_type,crops_dir:db.CROPS_DIR});if(response.status==='success'){db.updateCameraStatus(camera.id,true);broadcast({event:'cameras_updated',data:db.getCameras()});res.status(201).json(camera);}else{stopMockFFmpegStream(camera.id);db.deleteCamera(camera.id);res.status(500).json({error:response.message});}}catch(err){console.error('Failed to start mock loop stream:',err);res.status(500).json({error:err.message});}});

app.get('/api/events',(req,res)=>{const limit=parseInt(req.query.limit)||100;res.json(db.getEvents(limit));});
app.post('/api/events/:eventId/move',async(req,res)=>{const{eventId}=req.params,{personId,name}=req.body;if(!personId&&(!name||name.trim()===''))return res.status(400).json({error:'Person ID or Name is required.'});try{const event=db.getEvents(1000).find(e=>e.id===eventId);if(!event)return res.status(404).json({error:'Event not found.'});let embedding=null,foundInCluster=false;const clusterId=event.person_id;if(clusterId&&clusterId!=='UNKNOWN'){const cluster=db.getClusters().find(c=>c.id===clusterId);if(cluster){const photo=cluster.photos.find(p=>p.filename===event.crop_filename);if(photo){embedding=photo.embedding;foundInCluster=true;}}}if(!foundInCluster){const cropPath=path.join(db.CROPS_DIR,event.crop_filename);if(fs.existsSync(cropPath)){console.log(`[Event Promotion] Extracting embedding on-the-fly from crop: ${event.crop_filename}`);const response=await sendPythonCommand({cmd:'extract_embedding',img_path:cropPath,enforce_quality:false});if(response.status==='success')embedding=response.embedding;else console.warn(`[Event Promotion] Failed to extract embedding: ${response.message}`);}}const result=db.moveEventPhotoToPerson(eventId,personId,name,embedding);sendCandidatesToPython();broadcastDatabaseUpdate();broadcast({event:'clusters_updated'});res.json({success:true,event:result});}catch(err){res.status(500).json({error:err.message});}});
app.delete('/api/events',(req,res)=>{const success=db.clearEvents();broadcast({event:'events_cleared'});res.json({success});});
app.delete('/api/reset-all',async(req,res)=>{try{for(const cam of db.getCameras())if(cam.is_active)await stopCameraStream(cam);db.resetAll();sendCandidatesToPython();broadcastDatabaseUpdate();broadcast({event:'clusters_updated'});broadcast({event:'events_cleared'});broadcast({event:'cameras_updated',data:[]});res.json({success:true,message:'All cameras, clusters, persons, photos, and events have been completely reset.'});}catch(err){res.status(500).json({error:err.message});}});
app.post('/api/settings',async(req,res)=>{const{threshold,dis_type}=req.body;if(threshold!==undefined)settings.threshold=parseFloat(threshold);if(dis_type!==undefined)settings.dis_type=parseInt(dis_type);saveSettings();sendCandidatesToPython();for(const cam of db.getCameras().filter(c=>c.is_active)){try{const isLocal=cam.rtsp_url.startsWith('rtsp://localhost')||cam.rtsp_url.startsWith('rtsp://127.0.0.1'),targetRtspUrl=isLocal?cam.rtsp_url:`rtsp://127.0.0.1:8554/${cam.id}`;await sendPythonCommand({cmd:'start_stream',camera_id:cam.id,camera_name:cam.name,rtsp_url:targetRtspUrl,candidates:db.getAllCandidates(),threshold:settings.threshold,dis_type:settings.dis_type,crops_dir:db.CROPS_DIR,line_crossing_enabled:!!cam.line_crossing_enabled,line_y:cam.line_y!==undefined?cam.line_y:0.6,line_direction:cam.line_direction||'in',line_x_start:cam.line_x_start!==undefined?cam.line_x_start:0.0,line_x_end:cam.line_x_end!==undefined?cam.line_x_end:1.0});}catch(e){console.error(`Failed to apply updated settings on running camera ${cam.id}:`,e.message);}}broadcast({event:'stream_status',data:{threshold:settings.threshold,dis_type:settings.dis_type}});res.json({success:true,settings});});

function cleanup(){console.log('Cleaning up sub-processes...');if(pythonProcess)try{pythonProcess.kill('SIGKILL');}catch(e){}if(mediamtxProcess)try{mediamtxProcess.kill('SIGKILL');}catch(e){}ffmpegProcesses.forEach(proc=>{try{proc.kill('SIGKILL');}catch(e){}});transcoderProcesses.forEach(proc=>{try{proc.kill('SIGKILL');}catch(e){}});}
process.on('exit',cleanup);process.on('SIGINT',()=>{cleanup();process.exit();});process.on('SIGTERM',()=>{cleanup();process.exit();});
server.listen(PORT,()=>console.log(`Face Recognition Server is listening on http://localhost:${PORT}`));
