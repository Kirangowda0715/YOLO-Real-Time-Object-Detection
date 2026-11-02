/**
 * NEURALVISION // JAVASCRIPT CONTROLLER
 * Real-Time Telemetry Polling, Source Switching & Interactive Controls
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Metrics
  const valFps = document.getElementById('val-fps');
  const valLatency = document.getElementById('val-latency');
  const valObjects = document.getElementById('val-objects');
  const valSourceName = document.getElementById('val-source-name');
  const valHardware = document.getElementById('val-hardware');
  const valActiveModel = document.getElementById('val-active-model');
  const fillFps = document.getElementById('fill-fps');
  const fillLatency = document.getElementById('fill-latency');
  const fillObjects = document.getElementById('fill-objects');

  // DOM Elements - Controls
  const sourceTabs = document.querySelectorAll('.source-tab:not(.upload-btn)');
  const selectModel = document.getElementById('select-model');
  const rangeConf = document.getElementById('range-conf');
  const valConfTxt = document.getElementById('val-conf-txt');
  const rangeIou = document.getElementById('range-iou');
  const valIouTxt = document.getElementById('val-iou-txt');
  const checkTracking = document.getElementById('check-tracking');
  const btnPause = document.getElementById('btn-pause');
  const txtPause = document.getElementById('txt-pause');
  const btnSnapshot = document.getElementById('btn-snapshot');
  const btnLoop = document.getElementById('btn-loop');
  const btnFullscreen = document.getElementById('btn-fullscreen');
  const streamImg = document.getElementById('stream-img');
  const classBarsList = document.getElementById('class-bars-list');
  const detectionsFeed = document.getElementById('detections-feed');
  const badgeDetectionsCount = document.getElementById('badge-detections-count');

  // Upload elements
  const fileUploadInput = document.getElementById('file-upload-input');
  const videoWrapper = document.getElementById('video-wrapper');
  const dropZone = document.getElementById('drop-zone');

  let isPaused = false;
  let isLooping = true;
  let debounceTimeout = null;

  // =================================================================
  // 1. TELEMETRY POLLING (Every 300ms)
  // =================================================================
  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      if (!res.ok) return;
      const data = await res.json();

      // Update Top Metrics
      if (valFps) {
        valFps.textContent = data.fps.toFixed(1);
        const fpsPercent = Math.min(100, (data.fps / 90) * 100);
        if (fillFps) fillFps.style.width = `${fpsPercent}%`;
      }

      if (valLatency) {
        valLatency.textContent = data.latency_ms.toFixed(1);
        const latPercent = Math.min(100, (data.latency_ms / 50) * 100);
        if (fillLatency) fillLatency.style.width = `${latPercent}%`;
      }

      if (valObjects) {
        valObjects.textContent = data.object_count;
        const objPercent = Math.min(100, (data.object_count / 15) * 100);
        if (fillObjects) fillObjects.style.width = `${objPercent}%`;
      }

      if (valSourceName) valSourceName.textContent = data.source_name;
      if (valHardware) valHardware.textContent = data.hardware;
      if (valActiveModel) valActiveModel.textContent = data.active_model.toUpperCase();
      if (badgeDetectionsCount) badgeDetectionsCount.textContent = data.object_count;

      // Update Pause Button state
      if (data.paused !== isPaused) {
        isPaused = data.paused;
        updatePauseUI();
      }

      // Update Class Distribution Bars
      renderClassBars(data.class_counts, data.object_count);

      // Update Live Detections Feed
      renderDetectionsFeed(data.detections);

    } catch (err) {
      // Quiet fail if connection drops
    }
  }

  setInterval(fetchTelemetry, 350);

  // =================================================================
  // 2. RENDER WIDGETS
  // =================================================================
  function renderClassBars(classCounts, totalObjects) {
    if (!classBarsList) return;
    const entries = Object.entries(classCounts || {});

    if (entries.length === 0) {
      classBarsList.innerHTML = '<div class="empty-state">No objects detected.</div>';
      return;
    }

    entries.sort((a, b) => b[1] - a[1]);
    const maxVal = Math.max(...entries.map(e => e[1])) || 1;

    let html = '';
    for (const [cls, count] of entries) {
      const pct = Math.round((count / maxVal) * 100);
      html += `
        <div class="class-bar-row">
          <div class="class-bar-meta">
            <span class="class-bar-name">${cls}</span>
            <span class="class-bar-count">${count}</span>
          </div>
          <div class="class-bar-track">
            <div class="class-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }
    classBarsList.innerHTML = html;
  }

  function renderDetectionsFeed(detections) {
    if (!detectionsFeed) return;
    if (!detections || detections.length === 0) {
      detectionsFeed.innerHTML = '<div class="empty-state">No active objects in frame.</div>';
      return;
    }

    let html = '';
    for (const d of detections) {
      const trackText = d.track_id !== null ? `<span class="feed-item-track">#${d.track_id}</span>` : '';
      html += `
        <div class="feed-item">
          <div>
            <span class="feed-item-label">${d.label}</span>
            ${trackText}
          </div>
          <span class="feed-item-conf">${d.confidence}%</span>
        </div>
      `;
    }
    detectionsFeed.innerHTML = html;
  }

  // =================================================================
  // 3. SOURCE SWITCHING
  // =================================================================
  sourceTabs.forEach(tab => {
    tab.addEventListener('click', async () => {
      sourceTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const stype = tab.getAttribute('data-type');
      const spath = tab.getAttribute('data-path');

      try {
        await fetch('/api/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: stype, path: spath })
        });
        // Reload stream image
        reloadStream();
      } catch (err) {
        console.error('Failed to change source:', err);
      }
    });
  });

  function reloadStream() {
    if (streamImg) {
      streamImg.src = `/video_feed?t=${new Date().getTime()}`;
    }
  }

  // Auto-reconnect stream image if dropped or during server restart
  if (streamImg) {
    streamImg.onerror = () => {
      console.warn('Stream feed interrupted, retrying in 1.5s...');
      setTimeout(() => {
        reloadStream();
      }, 1500);
    };
  }

  // =================================================================
  // 4. CONFIG CONTROLS (Model, Sliders, Tracking)
  // =================================================================
  if (selectModel) {
    selectModel.addEventListener('change', async (e) => {
      const model = e.target.value;
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      reloadStream();
    });
  }

  if (rangeConf) {
    rangeConf.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (valConfTxt) valConfTxt.textContent = `${Math.round(val * 100)}%`;

      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => {
        fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ confidence: val })
        });
      }, 150);
    });
  }

  if (rangeIou) {
    rangeIou.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (valIouTxt) valIouTxt.textContent = `${Math.round(val * 100)}%`;

      clearTimeout(debounceTimeout);
      debounceTimeout = setTimeout(() => {
        fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ iou: val })
        });
      }, 150);
    });
  }

  if (checkTracking) {
    checkTracking.addEventListener('change', async (e) => {
      const tracking = e.target.checked;
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tracking })
      });
    });
  }

  // Pause / Resume Button
  if (btnPause) {
    btnPause.addEventListener('click', async () => {
      isPaused = !isPaused;
      updatePauseUI();
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paused: isPaused })
      });
    });
  }

  function updatePauseUI() {
    if (!txtPause) return;
    if (isPaused) {
      txtPause.textContent = 'Resume';
      btnPause.classList.add('active');
    } else {
      txtPause.textContent = 'Pause';
      btnPause.classList.remove('active');
    }
  }

  // Loop Toggle
  if (btnLoop) {
    btnLoop.addEventListener('click', async () => {
      isLooping = !isLooping;
      btnLoop.classList.toggle('active', isLooping);
      btnLoop.innerHTML = isLooping
        ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14M7 23l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg> Loop ON`
        : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14M7 23l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg> Loop OFF`;

      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ loop: isLooping })
      });
    });
  }

  // Reset Tracking IDs to #1
  const btnResetTrack = document.getElementById('btn-reset-track');
  if (btnResetTrack) {
    btnResetTrack.addEventListener('click', async () => {
      try {
        await fetch('/api/reset_tracks', { method: 'POST' });
        btnResetTrack.style.borderColor = '#00ff88';
        setTimeout(() => { btnResetTrack.style.borderColor = ''; }, 600);
      } catch (err) {
        console.error('Failed to reset tracks:', err);
      }
    });
  }

  // Snapshot Download
  if (btnSnapshot) {
    btnSnapshot.addEventListener('click', () => {
      window.open('/api/snapshot', '_blank');
    });
  }

  // Fullscreen Viewport Toggle
  if (btnFullscreen && videoWrapper) {
    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        videoWrapper.requestFullscreen().catch(err => console.log(err));
      } else {
        document.exitFullscreen().catch(err => console.log(err));
      }
    });
  }

  // =================================================================
  // 5. FILE UPLOAD & DRAG-AND-DROP
  // =================================================================
  if (fileUploadInput) {
    fileUploadInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (file) await handleFileUpload(file);
    });
  }

  // Drag and Drop support
  if (videoWrapper && dropZone) {
    ['dragenter', 'dragover'].forEach(name => {
      videoWrapper.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('active');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      videoWrapper.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('active');
      });
    });

    videoWrapper.addEventListener('drop', async (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        await handleFileUpload(files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show temporary loading indicator
    if (valSourceName) valSourceName.textContent = `Uploading ${file.name}...`;

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.status === 'success') {
        sourceTabs.forEach(t => t.classList.remove('active'));
        reloadStream();
      }
    } catch (err) {
      console.error('File upload failed:', err);
      alert('Upload failed. Please try again.');
    }
  }

});
