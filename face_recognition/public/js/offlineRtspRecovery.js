(() => {
  'use strict';

  const retryTimers = new Map();
  const health = new Map();
  const STALL_MS = 7000;
  const RETRY_MS = 2000;
  const HEALTH_MS = 2000;

  function log(cameraId, msg) {
    console.log(`[OFFLINE-RTSP][${cameraId}] ${msg}`);
  }

  function isCameraActive(cameraId) {
    try {
      return Array.isArray(cameras) && cameras.some(c => c.id === cameraId && c.is_active);
    } catch (_) {
      return !!document.getElementById(`stream-tile-${cameraId}`);
    }
  }

  function clearRetry(cameraId) {
    const timer = retryTimers.get(cameraId);
    if (timer) clearTimeout(timer);
    retryTimers.delete(cameraId);
  }

  function scheduleRetry(cameraId, reason) {
    if (!isCameraActive(cameraId)) return;
    if (retryTimers.has(cameraId)) return;
    log(cameraId, `Reconnect scheduled in ${RETRY_MS / 1000}s (${reason})`);
    const timer = setTimeout(() => {
      retryTimers.delete(cameraId);
      robustStartWhepStream(cameraId, true);
    }, RETRY_MS);
    retryTimers.set(cameraId, timer);
  }

  function closeConnection(cameraId) {
    try {
      if (typeof whepPeerConnections !== 'undefined' && whepPeerConnections.has(cameraId)) {
        const pc = whepPeerConnections.get(cameraId);
        whepPeerConnections.delete(cameraId);
        try { pc.oniceconnectionstatechange = null; } catch (_) {}
        try { pc.onconnectionstatechange = null; } catch (_) {}
        try { pc.close(); } catch (_) {}
      }
    } catch (_) {}

    const video = document.getElementById(`stream-video-${cameraId}`);
    if (video) {
      try { video.pause(); } catch (_) {}
      try { video.srcObject = null; } catch (_) {}
      video.classList.add('hidden');
    }

    const spinner = document.getElementById(`stream-spinner-${cameraId}`);
    if (spinner) {
      spinner.classList.remove('hidden');
      const p = spinner.querySelector('p');
      if (p) p.textContent = 'Reconnecting local stream...';
    }
  }

  async function waitForIceGatheringComplete(pc, timeoutMs = 1500) {
    if (pc.iceGatheringState === 'complete') return;
    await new Promise(resolve => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        pc.removeEventListener('icegatheringstatechange', check);
        clearTimeout(timer);
        resolve();
      };
      const check = () => {
        if (pc.iceGatheringState === 'complete') finish();
      };
      const timer = setTimeout(finish, timeoutMs);
      pc.addEventListener('icegatheringstatechange', check);
    });
  }

  async function robustStartWhepStream(cameraId, force = false) {
    if (!isCameraActive(cameraId)) return;

    if (!force) {
      try {
        if (typeof whepPeerConnections !== 'undefined' && whepPeerConnections.has(cameraId)) return;
      } catch (_) {}
    }

    clearRetry(cameraId);
    closeConnection(cameraId);

    const video = document.getElementById(`stream-video-${cameraId}`);
    const spinner = document.getElementById(`stream-spinner-${cameraId}`);
    if (!video) return;

    log(cameraId, 'Starting LAN-only WHEP connection...');

    // No STUN/TURN servers: this is intentionally LAN-only and requires no internet.
    const pc = new RTCPeerConnection({ iceServers: [] });
    try {
      whepPeerConnections.set(cameraId, pc);
    } catch (_) {
      log(cameraId, 'Could not access WHEP connection map.');
    }

    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = event => {
      const stream = event.streams && event.streams[0] ? event.streams[0] : new MediaStream([event.track]);
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.classList.remove('hidden');
      if (spinner) spinner.classList.add('hidden');
      video.play().catch(() => {});

      const now = Date.now();
      health.set(cameraId, {
        currentTime: Number(video.currentTime || 0),
        lastAdvance: now,
        lastTrack: now,
      });
      log(cameraId, 'Video track received.');
    };

    const failed = reason => {
      if (!isCameraActive(cameraId)) return;
      closeConnection(cameraId);
      scheduleRetry(cameraId, reason);
    };

    pc.oniceconnectionstatechange = () => {
      const s = pc.iceConnectionState;
      log(cameraId, `ICE=${s}`);
      if (s === 'failed' || s === 'disconnected' || s === 'closed') failed(`ICE ${s}`);
    };

    pc.onconnectionstatechange = () => {
      const s = pc.connectionState;
      log(cameraId, `PC=${s}`);
      if (s === 'failed' || s === 'disconnected' || s === 'closed') failed(`peer ${s}`);
    };

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      let response;
      try {
        response = await fetch(`http://${location.hostname}:8889/${cameraId}/whep`, {
          method: 'POST',
          body: pc.localDescription.sdp,
          headers: { 'Content-Type': 'application/sdp' },
          cache: 'no-store',
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
      }

      if (!response.ok) throw new Error(`MediaMTX WHEP HTTP ${response.status}`);
      const answer = await response.text();
      await pc.setRemoteDescription({ type: 'answer', sdp: answer });
      log(cameraId, 'WHEP session established.');
    } catch (err) {
      log(cameraId, `WHEP start failed: ${err.message}`);
      failed(err.message || 'WHEP setup error');
    }
  }

  // Replace the original start function. renderCameras() will call this version.
  window.startWhepStream = robustStartWhepStream;
  try { startWhepStream = robustStartWhepStream; } catch (_) {}

  // Detect freezes where WebRTC still claims to be connected but video time no longer advances.
  setInterval(() => {
    let active = [];
    try { active = (cameras || []).filter(c => c.is_active); } catch (_) {}

    for (const cam of active) {
      const cameraId = cam.id;
      const video = document.getElementById(`stream-video-${cameraId}`);
      if (!video) continue;

      const now = Date.now();
      let h = health.get(cameraId);
      if (!h) {
        h = { currentTime: Number(video.currentTime || 0), lastAdvance: now, lastTrack: now };
        health.set(cameraId, h);
      }

      const cur = Number(video.currentTime || 0);
      if (cur > h.currentTime + 0.02) {
        h.currentTime = cur;
        h.lastAdvance = now;
      }

      const hasTrack = !!(video.srcObject && video.srcObject.getVideoTracks && video.srcObject.getVideoTracks().some(t => t.readyState === 'live'));
      const connected = (() => {
        try {
          const pc = whepPeerConnections.get(cameraId);
          return !!pc && (pc.connectionState === 'connected' || pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed');
        } catch (_) { return false; }
      })();

      // Reconnect if no playable track, or the stream has stopped advancing for 7s.
      if (!hasTrack) {
        scheduleRetry(cameraId, 'no live video track');
      } else if (connected && now - h.lastAdvance > STALL_MS) {
        log(cameraId, `Video frozen for ${Math.round((now - h.lastAdvance) / 1000)}s; forcing reconnect.`);
        h.lastAdvance = now;
        closeConnection(cameraId);
        scheduleRetry(cameraId, 'video frame watchdog');
      }
    }

    // Clean stale camera state.
    for (const cameraId of [...health.keys()]) {
      if (!isCameraActive(cameraId)) {
        health.delete(cameraId);
        clearRetry(cameraId);
      }
    }
  }, HEALTH_MS);

  // If the browser tab sleeps/backgrounds and comes back, validate streams immediately.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    setTimeout(() => {
      let active = [];
      try { active = (cameras || []).filter(c => c.is_active); } catch (_) {}
      for (const cam of active) {
        const video = document.getElementById(`stream-video-${cam.id}`);
        if (!video || !video.srcObject) robustStartWhepStream(cam.id, true);
      }
    }, 250);
  });

  window.addEventListener('online', () => {
    // 'online' here also fires when LAN connectivity returns. No WAN is required.
    let active = [];
    try { active = (cameras || []).filter(c => c.is_active); } catch (_) {}
    for (const cam of active) robustStartWhepStream(cam.id, true);
  });

  console.log('[OFFLINE-RTSP] LAN-only WHEP + frozen-video auto recovery enabled.');
})();
