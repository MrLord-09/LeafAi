/* ============================
   LeafAI — App Logic
   ============================ */

// ==============================
// STATE
// ==============================
let currentFile = null;
let cameraStream = null;
let ipCameraTimer = null;
let ipCameraUrl = '';
let ipCameraPreviewUrl = '';
let diagnosticSnapshotUrl = '';
let ipCameraRefreshing = false;
let activeTab = 'symptoms';
const DEFAULT_API_URL = (() => {
  const origin = window.location.origin;
  if (origin.includes(':5500')) {
    return 'http://127.0.0.1:8000/api/v1/scan';
  }
  return '/api/v1/scan';
})();

async function loadCropNews() {
  const status = document.getElementById('newsStatus');
  const grid = document.getElementById('newsGrid');
  const retry = document.getElementById('newsRetry');
  if (!status || !grid) return;
  const base = DEFAULT_API_URL.replace('/api/v1/scan', '');
  if (retry) retry.hidden = true;
  status.textContent = 'Loading the latest news…';

  try {
    const response = await fetch(`${base}/api/v1/news`, {
      signal: AbortSignal.timeout(12000),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'News is temporarily unavailable.');
    grid.replaceChildren();

    for (const article of data.articles || []) {
      const card = document.createElement('article');
      card.className = 'news-card';
      if (article.image_url) {
        const image = document.createElement('img');
        image.className = 'news-thumbnail';
        const thumbnailUrl = new URL(`${base}/api/v1/news/image`, window.location.href);
        thumbnailUrl.searchParams.set('url', article.image_url);
        image.src = thumbnailUrl.href;
        image.alt = '';
        image.loading = 'lazy';
        image.decoding = 'async';
        image.addEventListener('error', () => {
          image.replaceWith(createNewsThumbnail(article.title));
        }, { once: true });
        card.appendChild(image);
      } else {
        card.appendChild(createNewsThumbnail(article.title));
      }

      const link = document.createElement('a');
      link.className = 'news-title';
      link.href = article.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = article.title;
      const metadata = document.createElement('p');
      metadata.className = 'news-meta';
      const published = article.published_at ? new Date(article.published_at) : null;
      const dateText = published && !Number.isNaN(published.getTime())
        ? published.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
        : '';
      metadata.textContent = [article.source, dateText].filter(Boolean).join(' · ');
      const summary = document.createElement('p');
      summary.className = 'news-summary';
      summary.textContent = article.summary || 'Read the full report from the publisher.';
      card.append(link, metadata, summary);
      grid.appendChild(card);
    }

    status.textContent = grid.childElementCount
      ? 'Latest India-focused agriculture headlines.'
      : 'No recent crop news is available right now.';
  } catch (error) {
    grid.replaceChildren();
    status.textContent = 'Could not load crop news. Check your internet connection and try again.';
    if (retry) retry.hidden = false;
    console.error('Could not load crop news', error);
  }
}

function createNewsThumbnail(title) {
  const thumbnail = document.createElement('div');
  const topic = /scheme|government|annadatas/i.test(title) ? 'public'
    : /wheat|rice|millet|bajra|jowar|ragi/i.test(title) ? 'grain'
    : /herbal|organic|natural farming/i.test(title) ? 'nature'
    : /farmer|farm/i.test(title) ? 'farm'
    : 'crop';
  thumbnail.className = `news-thumbnail news-thumbnail-placeholder news-thumbnail-${topic}`;
  thumbnail.setAttribute('role', 'img');
  thumbnail.setAttribute('aria-label', `Agriculture news thumbnail for ${title}`);
  const icon = document.createElement('span');
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = topic === 'public' ? '🇮🇳'
    : topic === 'grain' ? '🌾'
    : topic === 'nature' ? '🌱'
    : topic === 'farm' ? '🚜'
    : '🌿';
  thumbnail.appendChild(icon);
  return thumbnail;
}

// ==============================
// INIT
// ==============================
document.addEventListener('DOMContentLoaded', () => {
  const apiInput = document.getElementById('apiUrl');
  if (apiInput && !apiInput.value) {
    apiInput.value = DEFAULT_API_URL;
  }
  initParticles();
  initNavScroll();
  initPwaInstall();
  initDragDrop();
  initFileInput();
  initRevealObserver();
  document.getElementById('errorClose')?.addEventListener('click', hideError);
  document.getElementById('newsRetry')?.addEventListener('click', loadCropNews);
  loadCropNews();
});

// ==============================
// BG PARTICLES
// ==============================
function initParticles() {
  const container = document.getElementById('bgParticles');
  if (!container) return;
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 300 + 80;
    p.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      left: ${Math.random() * 100}%;
      animation-duration: ${Math.random() * 18 + 12}s;
      animation-delay: ${Math.random() * -20}s;
      opacity: ${Math.random() * 0.3 + 0.05};
    `;
    container.appendChild(p);
  }
}

// ==============================
// NAVBAR SCROLL
// ==============================
function initNavScroll() {
  const nav = document.getElementById('navbar');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 40);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  const toggle = document.getElementById('navToggle');
  const links = document.getElementById('navLinks');
  if (!toggle || !links) return;
  const closeMenu = () => {
    links.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
  };
  toggle.setAttribute('aria-expanded', 'false');
  toggle.addEventListener('click', () => {
    const isOpen = links.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(isOpen));
  });
  links.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeMenu();
  });
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) closeMenu();
  });
}

function initPwaInstall() {
  const installButton = document.getElementById('installAppBtn');
  const helpDialog = document.getElementById('installHelpDialog');
  const closeHelpButton = document.getElementById('closeInstallHelpBtn');
  if (!installButton || !helpDialog || !closeHelpButton) return;

  let installPrompt = null;
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches
    || window.navigator.standalone === true;
  installButton.hidden = isStandalone;

  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault();
    installPrompt = event;
    installButton.hidden = false;
  });
  window.addEventListener('appinstalled', () => {
    installPrompt = null;
    installButton.hidden = true;
  });
  installButton.addEventListener('click', async () => {
    if (!installPrompt) {
      helpDialog.showModal();
      return;
    }

    try {
      await installPrompt.prompt();
      const { outcome } = await installPrompt.userChoice;
      if (outcome === 'accepted') installButton.hidden = true;
    } catch (error) {
      console.error('Could not open the app install prompt.', error);
      helpDialog.showModal();
    } finally {
      installPrompt = null;
    }
  });
  closeHelpButton.addEventListener('click', () => helpDialog.close());
  helpDialog.addEventListener('click', (event) => {
    if (event.target === helpDialog) helpDialog.close();
  });

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js')
      .catch(error => console.error('Could not register the app service worker.', error));
  }
}

// ==============================
// DRAG & DROP
// ==============================
function initDragDrop() {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
}

// ==============================
// FILE INPUT
// ==============================
function initFileInput() {
  const input = document.getElementById('fileInput');
  if (!input) return;
  input.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

function handleFile(file, fromCamera = false) {
  if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
    showError('Invalid file type. Please upload a JPEG or PNG image.');
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showError('File too large. Maximum size is 5MB.');
    return;
  }

  if (!fromCamera && cameraStream) {
    stopCameraStream();
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('retakePhotoBtn').style.display = cameraStream ? 'inline-flex' : 'none';
    document.getElementById('stopCapturedCameraBtn').style.display = cameraStream ? 'inline-flex' : 'none';
    showState('preview');
    hideResults();
  };
  reader.readAsDataURL(file);
}

// ==============================
// STATE MANAGEMENT
// ==============================
function showState(state) {
  document.getElementById('uploadContent').style.display  = state === 'upload'  ? 'flex' : 'none';
  document.getElementById('previewContent').style.display = state === 'preview' ? 'flex' : 'none';
  document.getElementById('cameraContent').style.display  = state === 'camera'  ? 'flex' : 'none';
  document.getElementById('ipCameraSetup').style.display  = state === 'ipCameraSetup' ? 'flex' : 'none';
  document.getElementById('ipCameraContent').style.display = state === 'ipCamera' ? 'flex' : 'none';
}

function resetUpload() {
  currentFile = null;
  document.getElementById('fileInput').value = '';
  stopCameraStream();
  document.getElementById('retakePhotoBtn').style.display = 'none';
  showState('upload');
  hideResults();
}

function hideResults() {
  document.getElementById('resultsPanel').style.display = 'none';
  clearDiagnosticSnapshot();
}

function closeResults() {
  hideResults();
}

function clearDiagnosticSnapshot() {
  document.getElementById('diagnosticSnapshot').style.display = 'none';
  document.getElementById('diagnosticSnapshotImg').removeAttribute('src');
  if (diagnosticSnapshotUrl) {
    URL.revokeObjectURL(diagnosticSnapshotUrl);
    diagnosticSnapshotUrl = '';
  }
}

// ==============================
// CAMERA
// ==============================
function openCamera() {
  hideError();
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    .then((stream) => {
      stopCameraStream();
      cameraStream = stream;
      document.getElementById('cameraVideo').srcObject = stream;
      showState('camera');
    })
    .catch(() => showError('Could not access the camera. Please upload an image instead.'));
}

function capturePhoto() {
  const video = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  if (!video.videoWidth || !video.videoHeight) {
    showError('The camera is not ready yet. Please wait a moment and try again.');
    return;
  }
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob((blob) => {
    if (!blob) {
      showError('Could not capture an image. Please try again.');
      return;
    }
    const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
    handleFile(file, true);
  }, 'image/jpeg', 0.9);
}

function retakePhoto() {
  if (!cameraStream) {
    openCamera();
    return;
  }
  showState('camera');
  hideError();
}

function stopCameraCapture() {
  stopCameraStream();
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  document.getElementById('cameraVideo').srcObject = null;
  document.getElementById('retakePhotoBtn').style.display = 'none';
  document.getElementById('stopCapturedCameraBtn').style.display = 'none';
}

function closeCamera() {
  stopCameraStream();
  currentFile = null;
  document.getElementById('fileInput').value = '';
  hideResults();
  showState('upload');
}

function openIpCameraSetup() {
  showState('ipCameraSetup');
  hideError();
}

function getApiBaseUrl() {
  return document.getElementById('apiUrl').value.trim().replace(/\/api\/v1\/scan\/?$/, '').replace(/\/+$/, '');
}

function normalizeIpCameraUrl(value) {
  const enteredUrl = value.trim();
  const url = new URL(/^[a-z][a-z\d+.-]*:\/\//i.test(enteredUrl) ? enteredUrl : `http://${enteredUrl}`);
  if (url.protocol !== 'http:' || (url.pathname !== '/' && url.pathname !== '/shot.jpg') || url.search || url.hash) {
    throw new Error('Enter the IP Webcam server address, such as http://192.168.1.25:8080.');
  }
  if (!url.port) url.port = '8080';
  url.pathname = '/shot.jpg';
  return url.toString();
}

async function fetchIpCameraFrame() {
  const snapshotUrl = `${getApiBaseUrl()}/api/v1/ip-camera/snapshot?url=${encodeURIComponent(ipCameraUrl)}&t=${Date.now()}`;
  const response = await fetch(snapshotUrl);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `IP Webcam request failed (${response.status}).`);
  }
  return response.blob();
}

function updateIpCameraPreview(blob) {
  const imageUrl = URL.createObjectURL(blob);
  document.getElementById('ipCameraPreview').src = imageUrl;
  if (ipCameraPreviewUrl) URL.revokeObjectURL(ipCameraPreviewUrl);
  ipCameraPreviewUrl = imageUrl;
}

async function connectIpCamera() {
  const urlInput = document.getElementById('ipCameraUrl');
  const connectButton = document.getElementById('connectIpCameraBtn');
  connectButton.disabled = true;
  hideError();

  try {
    ipCameraUrl = normalizeIpCameraUrl(urlInput.value);
    const firstFrame = await fetchIpCameraFrame();
    updateIpCameraPreview(firstFrame);
    showState('ipCamera');
    document.getElementById('ipCameraStatus').textContent = 'Connected. Keep the crop in view, then capture five frames for diagnosis.';
    ipCameraTimer = window.setInterval(refreshIpCameraPreview, 1200);
  } catch (error) {
    showError(`Could not connect to the IP Webcam. Confirm the phone and computer are on the same Wi-Fi, the IP Webcam server is started, and the address/port match the app. ${error.message}`);
  } finally {
    connectButton.disabled = false;
  }
}

async function refreshIpCameraPreview() {
  if (ipCameraRefreshing || !ipCameraUrl) return;
  ipCameraRefreshing = true;
  try {
    updateIpCameraPreview(await fetchIpCameraFrame());
  } catch (error) {
    document.getElementById('ipCameraStatus').textContent = `Preview unavailable: ${error.message}`;
  } finally {
    ipCameraRefreshing = false;
  }
}

function closeIpCamera() {
  if (ipCameraTimer !== null) {
    window.clearInterval(ipCameraTimer);
    ipCameraTimer = null;
  }
  ipCameraUrl = '';
  ipCameraRefreshing = false;
  if (ipCameraPreviewUrl) {
    URL.revokeObjectURL(ipCameraPreviewUrl);
    ipCameraPreviewUrl = '';
  }
  document.getElementById('ipCameraPreview').removeAttribute('src');
  showState('upload');
}

async function runIpCameraDiagnostic() {
  const captureButton = document.getElementById('scanIpCameraBtn');
  const closeButton = document.getElementById('closeIpCameraPreviewBtn');
  const status = document.getElementById('ipCameraStatus');
  const frames = [];
  if (ipCameraTimer !== null) {
    window.clearInterval(ipCameraTimer);
    ipCameraTimer = null;
  }
  captureButton.disabled = true;
  closeButton.disabled = true;
  hideError();

  try {
    for (let index = 0; index < 5; index += 1) {
      status.textContent = `Capturing frame ${index + 1} of 5…`;
      const blob = await fetchIpCameraFrame();
      frames.push(new File([blob], `ip-webcam-${index + 1}.jpg`, { type: blob.type || 'image/jpeg' }));
      if (index < 4) await new Promise(resolve => window.setTimeout(resolve, 500));
    }
    status.textContent = 'Analyzing five frames…';
    const result = await runBatchDiagnostic(frames);
    const selectedFrame = frames[result.selected_snapshot_index];
    if (!selectedFrame) {
      throw new Error('The diagnosis response did not include a valid selected snapshot.');
    }
    showResults(result, selectedFrame);
    status.textContent = 'Analysis complete. Camera still connected — capture again when ready.';
  } catch (error) {
    status.textContent = 'Capture or analysis failed.';
    showError(error.message);
  } finally {
    captureButton.disabled = false;
    closeButton.disabled = false;
    if (ipCameraUrl && ipCameraTimer === null) {
      ipCameraTimer = window.setInterval(refreshIpCameraPreview, 1200);
    }
  }
}

async function runBatchDiagnostic(files) {
  const apiUrl = document.getElementById('apiUrl').value.trim().replace(/\/+$/, '');
  const formData = new FormData();
  files.forEach(file => formData.append('files', file, file.name));

  const response = await fetch(`${apiUrl}/batch`, { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(`API Error (${response.status}): ${error.detail || response.statusText}`);
  }
  return response.json();
}

// ==============================
// DIAGNOSTIC SCAN
// ==============================
async function runDiagnostic() {
  if (!currentFile) return;

  const btn = document.getElementById('runScanBtn');
  const btnText = btn.querySelector('.scan-btn-text');
  const btnLoader = btn.querySelector('.scan-btn-loader');
  btnText.style.display = 'none';
  btnLoader.style.display = 'flex';
  btn.disabled = true;
  hideError();

  const apiUrl = document.getElementById('apiUrl').value.trim();
  const formData = new FormData();
  formData.append('file', currentFile, currentFile.name || 'leaf.jpg');

  try {
    const response = await fetch(apiUrl, { method: 'POST', body: formData });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`API Error (${response.status}): ${err.detail || response.statusText}`);
    }

    const data = await response.json();
    showResults(data, currentFile);
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      showError('❌ Cannot reach the FastAPI backend. Make sure Uvicorn is running on port 8000.');
    } else {
      showError(err.message);
    }
  } finally {
    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
    btn.disabled = false;
  }
}

// ==============================
// RESULTS DISPLAY
// ==============================
function showResults(data, snapshotFile = null) {
  const { confidence, report } = data;
  if (data.sample_count) {
    document.getElementById('scanIpCameraBtn').textContent = '↻ Capture 5 & Diagnose Again';
  } else {
    document.querySelector('#runScanBtn .scan-btn-text').textContent = '↻ Diagnose Again';
  }
  const panel = document.getElementById('resultsPanel');
  panel.style.display = 'block';
  clearDiagnosticSnapshot();
  if (snapshotFile) {
    diagnosticSnapshotUrl = URL.createObjectURL(snapshotFile);
    document.getElementById('diagnosticSnapshotImg').src = diagnosticSnapshotUrl;
    document.getElementById('diagnosticSnapshot').style.display = 'block';
  }
  document.getElementById('resultsTitle').textContent = data.sample_count
    ? `📊 Diagnostic Report · ${data.sample_count}-frame consensus`
    : '📊 Diagnostic Report';

  // Scroll to results
  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Disease name
  const name = (report.disease_name || 'Unknown').replace(/_/g, ' ');
  document.getElementById('diseaseLabel').textContent = name;

  // Confidence
  document.getElementById('confidencePct').textContent = `${parseFloat(confidence).toFixed(1)}%`;
  setTimeout(() => {
    document.getElementById('confidenceBar').style.width = `${Math.min(100, confidence)}%`;
  }, 100);

  // Caption
  const pct = parseFloat(confidence);
  const caption = pct >= 90 ? '✅ High confidence — result is reliable.' :
                  pct >= 70 ? '⚠️ Moderate confidence — consider retaking a clearer photo.' :
                              '🔴 Low confidence — please retake with better lighting.';
  const consensus = data.vote_count
    ? ` ${data.vote_count} of ${data.sample_count} frames agreed.`
    : '';
  document.getElementById('confidenceCaption').textContent = `${caption}${consensus}`;

  // Tab content
  document.getElementById('textSymptoms').textContent  = report.symptoms        || 'No data available.';
  document.getElementById('textOrganic').textContent   = report.organic_control  || 'No data available.';
  document.getElementById('textChemical').textContent  = report.chemical_control || 'No data available.';

  // Citation
  const cit = document.getElementById('resultCitation');
  cit.textContent = report.source ? `📚 Source: ${report.source}` : '';

  // Reset tabs
  switchTab('symptoms');
}

// ==============================
// TABS
// ==============================
function switchTab(tab) {
  activeTab = tab;
  ['symptoms', 'organic', 'chemical'].forEach(t => {
    const btn = document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const content = document.getElementById(`tabContent${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn)     btn.classList.toggle('active', t === tab);
    if (content) content.style.display = t === tab ? 'block' : 'none';
  });
}

// ==============================
// API TEST
// ==============================
async function testApiConnection() {
  const statusEl = document.getElementById('apiStatus');
  const url = document.getElementById('apiUrl').value.trim();
  const baseUrl = url.replace('/api/v1/scan', '').replace(/\/+$/, '');

  statusEl.className = 'api-status checking';
  statusEl.textContent = '⏳ Testing connection...';

  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (res.ok && data.status === 'online') {
      statusEl.className = 'api-status online';
      statusEl.textContent = '✅ Backend is online and ready.';
    } else {
      throw new Error('Unexpected response');
    }
  } catch {
    statusEl.className = 'api-status offline';
    statusEl.textContent = '❌ Backend unreachable. Start Uvicorn on port 8000.';
  }
}

// ==============================
// ERROR
// ==============================
function showError(msg) {
  const banner = document.getElementById('errorBanner');
  document.getElementById('errorText').textContent = msg;
  if (!banner.open) banner.showModal();
  banner.focus({ preventScroll: true });
}
function hideError() {
  const banner = document.getElementById('errorBanner');
  if (banner.open) banner.close();
}

// ==============================
// SCROLL REVEAL
// ==============================
function initRevealObserver() {
  const targets = document.querySelectorAll('.step-card, .section-header, .news-card, .scanner-card, .api-config');
  targets.forEach(el => el.classList.add('reveal'));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 60);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  targets.forEach(el => observer.observe(el));
}
