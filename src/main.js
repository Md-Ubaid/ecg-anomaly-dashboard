import { ApiService } from './APIservice.js';

const $ = (id) => document.getElementById(id);
const DISPLAY_SAMPLES = 720;
const ECG_SAMPLES = 4800;
const ECG_LEADS = 12;
const HEATMAP_BINS = 60;
const LEAD_MAP = Object.freeze({ I: 0, II: 1, III: 2, aVR: 3, aVL: 4, aVF: 5, V1: 6, V2: 7, V3: 8, V4: 9, V5: 10, V6: 11 });
const LEAD_SCALE = Object.freeze({ I: 100, II: 110, III: 100, aVR: 90, aVL: 90, aVF: 100, V1: 90, V2: 100, V3: 100, V4: 100, V5: 100, V6: 100 });

let patients = [
  { id: 'PT-90412', name: 'Aarav Mehta', info: '54 y · Male · Demonstration record', hr: 72, diagnosis: 'Research record', signal: null },
  { id: 'PT-90377', name: 'Sunita Verma', info: '61 y · Female · Demonstration record', hr: 64, diagnosis: 'Research record', signal: null },
  { id: 'PT-90503', name: 'Rohan Iyer', info: '38 y · Male · Demonstration record', hr: 88, diagnosis: 'Research record', signal: null }
];
let patient = null, signal = null, reconstruction = null, residual = null, mse = null, analysisTimestamp = null, modelMetadata = null;
let lead = 'II', cutoff = 0.035, anomaly = false, revision = 0, analyzing = false, audioEnabled = false, acknowledgedRevision = null;
let fallbackSignalPromise = null;
const patientSignalPromises = new Map();
const events = [];
let observed = new Float32Array(DISPLAY_SAMPLES).fill(NaN);
let baseline = new Float32Array(DISPLAY_SAMPLES).fill(NaN);
const canvas = $('ecg');
const ctx = canvas.getContext('2d');
document.documentElement.dataset.theme = 'light';

function requireFinite(value, message) {
  if (!Number.isFinite(value)) throw new Error(message);
  return value;
}

function validateWindow(input, label = 'ECG signal') {
  if (!Array.isArray(input) || input.length < ECG_SAMPLES) throw new Error(`${label} must contain at least ${ECG_SAMPLES} samples.`);
  return input.slice(0, ECG_SAMPLES).map((row, index) => {
    if (!Array.isArray(row) || row.length !== ECG_LEADS) throw new Error(`${label} sample ${index} must contain exactly ${ECG_LEADS} leads.`);
    if (!row.every(Number.isFinite)) throw new Error(`${label} contains a non-finite value at sample ${index}.`);
    return row;
  });
}

function parseResult(result) {
  if (!result || typeof result !== 'object' || typeof result.is_anomaly !== 'boolean') throw new Error('Invalid inference response.');
  const matrix = Array.isArray(result.reconstructed?.[0]?.[0]) ? result.reconstructed[0] : result.reconstructed;
  const metadata = result.metadata ?? result.model_metadata ?? null;
  if (metadata !== null && (typeof metadata !== 'object' || Array.isArray(metadata))) throw new Error('Inference response has invalid model metadata.');
  if (metadata?.samplingRateHz !== undefined) requireFinite(Number(metadata.samplingRateHz), 'Inference response has an invalid sampling rate.');
  return { reconstruction: validateWindow(matrix, 'Model reconstruction'), mse: requireFinite(Number(result.mse_error), 'Inference response has an invalid MSE.'), anomaly: result.is_anomaly, metadata };
}

function currentLeadIndex() {
  const index = LEAD_MAP[lead];
  if (!Number.isInteger(index)) throw new Error(`Unsupported lead: ${lead}`);
  return index;
}

function displaySamples(values) {
  return Float32Array.from({ length: DISPLAY_SAMPLES }, (_, i) => values[Math.round(i * (values.length - 1) / (DISPLAY_SAMPLES - 1))]);
}

function updateLeadData() {
  if (!signal) { observed.fill(NaN); baseline.fill(NaN); residual = null; return; }
  const index = currentLeadIndex();
  observed = displaySamples(signal.map((row) => row[index]));
  baseline = reconstruction ? displaySamples(reconstruction.map((row) => row[index])) : new Float32Array(DISPLAY_SAMPLES).fill(NaN);
  residual = reconstruction ? Float32Array.from({ length: ECG_SAMPLES }, (_, i) => (signal[i][index] - reconstruction[i][index]) ** 2) : null;
}

function drawGrid(width, height) {
  const style = getComputedStyle(document.documentElement);
  ctx.fillStyle = style.getPropertyValue('--screen-bg').trim() || '#f8fbfd';
  ctx.fillRect(0, 0, width, height);
  const major = style.getPropertyValue('--grid-major').trim() || '#d5e0e6';
  const minor = style.getPropertyValue('--grid-minor').trim() || '#e9eff2';
  for (let x = 0; x <= width; x += 10) { ctx.strokeStyle = x % 50 ? minor : major; ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, height); ctx.stroke(); }
  for (let y = 0; y <= height; y += 10) { ctx.strokeStyle = y % 50 ? minor : major; ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(width, y + .5); ctx.stroke(); }
}

function drawPath(values, x, y, color, width, dash = []) {
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  let penDown = false;
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) { penDown = false; return; }
    if (penDown) ctx.lineTo(x(index), y(value)); else ctx.moveTo(x(index), y(value));
    penDown = true;
  });
  ctx.stroke(); ctx.setLineDash([]);
}

function drawEcg() {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width)), height = Math.max(1, Math.round(rect.height));
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) { canvas.width = width * dpr; canvas.height = height * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); drawGrid(width, height);
  const x = (i) => i / (DISPLAY_SAMPLES - 1) * width;
  const y = (value) => height / 2 - value * (LEAD_SCALE[lead] ?? 100);
  drawPath(baseline, x, y, '#587080', 1.5, [6, 4]);
  drawPath(observed, x, y, '#0e7c86', 2.2);
  if (!residual) return;
  ctx.strokeStyle = '#d62839'; ctx.lineWidth = 3; ctx.lineCap = 'round';
  for (let i = 0; i < DISPLAY_SAMPLES - 1; i += 1) {
    const sourceIndex = Math.round(i * (ECG_SAMPLES - 1) / (DISPLAY_SAMPLES - 1));
    if (residual[sourceIndex] <= cutoff || !Number.isFinite(observed[i]) || !Number.isFinite(observed[i + 1])) continue;
    ctx.beginPath(); ctx.moveTo(x(i), y(observed[i])); ctx.lineTo(x(i + 1), y(observed[i + 1])); ctx.stroke();
  }
}

function updateHeatmap() {
  const heatmap = $('xai-heatmap');
  heatmap.replaceChildren();
  if (!residual) return;
  const fragment = document.createDocumentFragment();
  for (let bin = 0; bin < HEATMAP_BINS; bin += 1) {
    const start = Math.floor(bin * ECG_SAMPLES / HEATMAP_BINS), end = Math.floor((bin + 1) * ECG_SAMPLES / HEATMAP_BINS);
    let total = 0; for (let i = start; i < end; i += 1) total += residual[i];
    const score = total / (end - start);
    const cell = document.createElement('div');
    cell.className = `heatmap-cell${score > cutoff ? ' elevated' : ''}`;
    cell.title = `Mean reconstruction residual: ${score.toExponential(2)}`;
    fragment.append(cell);
  }
  heatmap.append(fragment);
}

function renderSignal() { updateLeadData(); drawEcg(); updateHeatmap(); }
function setText(id, value) { $(id).textContent = String(value); }
function escapeHtml(value) { const node = document.createElement('span'); node.textContent = String(value ?? ''); return node.innerHTML; }

function updateFindings() {
  const state = mse === null ? 'Not analyzed' : anomaly ? 'Elevated residual' : 'No model anomaly flag';
  const sourceMetadata = modelMetadata?.samplingRateHz ? `${modelMetadata.samplingRateHz} Hz · ${modelMetadata.signalUnit ?? 'unit not reported'}` : 'Not reported by the backend';
  const modelVersion = modelMetadata?.modelVersion ?? modelMetadata?.model_version ?? 'Not reported by the backend';
  $('t-find').innerHTML = `<div class="row ${anomaly ? 'bad' : ''}"><i></i><div><span>Model assessment</span><b>${escapeHtml(state)}</b></div><div><span>Window</span><b>${ECG_SAMPLES} samples × ${ECG_LEADS} leads</b></div><div><span>Lead shown</span><b>${escapeHtml(lead)}</b></div><div><span>Interpretation</span>Reconstruction residual only; not a clinical diagnosis.</div><span class="chip">Research only</span></div><div class="row"><i></i><div><span>Reconstruction MSE</span><b>${mse === null ? '—' : mse.toExponential(3)}</b></div><div><span>Display cutoff</span><b>${cutoff.toFixed(3)}</b></div><div><span>Attribution</span><b>Point-wise squared residual</b></div><div><span>Clinical measurement</span>ST/QRS measurements are not inferred from residual error.</div><span class="chip">Not diagnostic</span></div><div class="row"><i></i><div><span>Patient / window</span><b>${escapeHtml(patient?.id ?? '—')} · ${ECG_SAMPLES} samples</b></div><div><span>Inference time</span><b>${escapeHtml(analysisTimestamp ?? 'Not analyzed')}</b></div><div><span>Sampling / units</span><b>${escapeHtml(sourceMetadata)}</b></div><div><span>Model version</span><b>${escapeHtml(modelVersion)}</b></div><span class="chip">Provenance</span></div>`;
}

function updateLog() {
  setText('cnt', `(${events.length})`);
  $('t-log').innerHTML = events.length ? events.slice().reverse().map((event) => `<div class="ev"><span><b>Elevated reconstruction residual</b> · ${escapeHtml(event.patientId)} · Lead ${escapeHtml(event.lead)}<br><span>Model MSE ${event.mse.toExponential(3)} · clinician review required</span></span><span class="num">${escapeHtml(event.time)}</span></div>`).join('') : '<div class="p"><p>No model-flagged windows in this session.</p></div>';
  setText('rsub', `${events.length} model-flagged window${events.length === 1 ? '' : 's'} this session`);
  $('rep').innerHTML = `<div class="p"><h2>Research session summary</h2><p>Model flags are reconstruction-residual results and require clinician review. They do not identify a disease or replace ECG interpretation.</p><p>${events.length ? `${events.length} flagged window${events.length === 1 ? ' was' : 's were'} recorded.` : 'No flagged windows have been recorded.'}</p></div>`;
}

function beep() {
  if (!audioEnabled) return;
  try {
    const Audio = window.AudioContext || window.webkitAudioContext, audio = new Audio(), oscillator = audio.createOscillator(), gain = audio.createGain();
    oscillator.frequency.value = 880; gain.gain.value = .04; oscillator.connect(gain); gain.connect(audio.destination); oscillator.start(); oscillator.stop(audio.currentTime + .18); oscillator.addEventListener('ended', () => audio.close());
  } catch (error) { console.warn('Audio alert is unavailable.', error); }
}

function updateStatus() {
  const acknowledged = acknowledgedRevision === revision;
  $('status').classList.toggle('bad', anomaly);
  setText('sv', mse === null ? 'Awaiting analysis' : anomaly ? 'Model flag raised' : 'No model flag');
  setText('sp', mse === null ? 'Select Analyze to run the fixed ECG window through the reconstruction model.' : anomaly ? 'Elevated reconstruction residual. Clinician review is required.' : 'No model flag for this window; this does not rule out clinical abnormality.');
  $('alert').classList.toggle('on', anomaly && !acknowledged);
  setText('mse', mse === null ? '—' : mse.toFixed(4)); setText('hr', patient?.hr ?? '—'); updateFindings();
}

function renderPatientOptions() {
  $('ptlist').innerHTML = patients.map((item, index) => `<button class="pt" data-i="${index}"><h3>${escapeHtml(item.name)}</h3><p><b>${escapeHtml(item.id)}</b><br>${escapeHtml(item.info || 'Research record')}</p><span class="chip">${escapeHtml(item.diagnosis || 'Research record')}</span></button>`).join('');
  $('pt').innerHTML = patients.map((item, index) => `<option value="${index}">${escapeHtml(item.name)} (${escapeHtml(item.id)})</option>`).join('');
}

function fallbackSignal() {
  if (!fallbackSignalPromise) fallbackSignalPromise = fetch('./sample_ecg.json').then(async (response) => {
    if (!response.ok) throw new Error(`Sample ECG load failed (${response.status}).`);
    return validateWindow(await response.json(), 'Fallback ECG signal');
  });
  return fallbackSignalPromise;
}
function signalFor(item) {
  if (item.signal) return Promise.resolve(validateWindow(item.signal, `ECG signal for ${item.id}`));
  if (!patientSignalPromises.has(item.id)) {
    patientSignalPromises.set(item.id, fetch(`./data/patients/${encodeURIComponent(item.id)}.json`).then(async (response) => {
      if (!response.ok) throw new Error(`ECG signal load failed (${response.status}).`);
      const payload = await response.json();
      if (payload?.id !== item.id) throw new Error('ECG signal response belongs to a different patient.');
      return validateWindow(payload.signal, `ECG signal for ${item.id}`);
    }));
  }
  return patientSignalPromises.get(item.id).catch(async (error) => {
    patientSignalPromises.delete(item.id);
    if (patients.length === 3 && !item.signal) return fallbackSignal();
    throw error;
  });
}

async function setPatient(index) {
  const nextPatient = patients[index]; if (!nextPatient) return;
  const requestRevision = ++revision;
  patient = nextPatient; signal = reconstruction = residual = null; mse = null; analysisTimestamp = null; modelMetadata = null; anomaly = false; acknowledgedRevision = null;
  analyzing = false; $('stream').disabled = false; $('stream').querySelector('span').textContent = 'Analyze Next Window';
  $('pt').value = String(index); setText('sub', `Loading ECG · ${patient.name}`); updateStatus(); renderSignal();
  try {
    const nextSignal = await signalFor(nextPatient);
    if (requestRevision !== revision || patient !== nextPatient) return;
    signal = nextSignal; renderSignal(); setText('sub', `Ready to analyze · ${patient.name}`);
  } catch (error) {
    if (requestRevision === revision) { console.error(error); setText('sub', `Unable to load ECG signal: ${error.message}`); }
  }
}

function selectSegment(id, callback) {
  $(id).onclick = (event) => { const button = event.target.closest('button'); if (!button) return; $(id).querySelectorAll('button').forEach((item) => item.setAttribute('aria-pressed', String(item === button))); callback(button); };
}

$('stream').onclick = async function () {
  if (analyzing || !patient) return;
  const targetPatient = patient, requestRevision = revision;
  analyzing = true; this.disabled = true; setText('sub', 'Running STAE inference…'); this.querySelector('span').textContent = 'Analyzing…';
  try {
    const targetSignal = await signalFor(targetPatient);
    const result = parseResult(await ApiService.analyzeECGSignal(targetSignal));
    if (requestRevision !== revision || patient !== targetPatient) return;
    signal = targetSignal; reconstruction = result.reconstruction; mse = result.mse; modelMetadata = result.metadata; analysisTimestamp = new Date().toLocaleString(); anomaly = result.anomaly; acknowledgedRevision = null;
    renderSignal(); updateStatus(); setText('sub', `Analysis complete · ${patient.name}`);
    if (anomaly) { events.push({ patientId: patient.id, lead, mse, time: new Date().toLocaleTimeString() }); updateLog(); beep(); }
  } catch (error) {
    if (requestRevision === revision) { console.error('Inference failed:', error); setText('sub', `Inference failed: ${error.message}`); }
  } finally {
    if (requestRevision === revision) { analyzing = false; this.disabled = false; this.querySelector('span').textContent = 'Analyze Next Window'; }
  }
};

$('ptlist').onclick = (event) => { const button = event.target.closest('.pt'); if (!button) return; void setPatient(Number(button.dataset.i)); document.querySelectorAll('.wrap > section').forEach((section) => section.classList.add('hide')); $('v-monitor').classList.remove('hide'); };
$('pt').onchange = (event) => void setPatient(Number(event.target.value));
$('ack').onclick = () => { acknowledgedRevision = revision; $('alert').classList.remove('on'); };
selectSegment('lead', (button) => { lead = button.textContent.trim(); setText('tag', `Lead ${lead} · visual display only`); renderSignal(); updateFindings(); });
selectSegment('sens', (button) => { cutoff = Number(button.dataset.t); setText('thv', cutoff.toFixed(3)); drawEcg(); updateHeatmap(); updateFindings(); });
selectSegment('snd', (button) => { audioEnabled = button.textContent.trim() === 'On'; });
selectSegment('tabs', (button) => ['find', 'log', 'about'].forEach((name) => $('t-' + name).classList.toggle('hide', name !== button.dataset.t)));
$('theme').onclick = () => { const root = document.documentElement; root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light'; drawEcg(); };
document.querySelectorAll('nav button[data-v]').forEach((button) => { button.onclick = () => { document.querySelectorAll('nav button[data-v]').forEach((item) => item.removeAttribute('aria-current')); button.setAttribute('aria-current', 'page'); document.querySelectorAll('.wrap > section').forEach((section) => section.classList.add('hide')); $('v-' + button.dataset.v).classList.remove('hide'); }; });
new ResizeObserver(() => drawEcg()).observe(canvas);

async function init() {
  try {
    const response = await fetch('./patients_index.json'); if (!response.ok) throw new Error(`Patient index load failed (${response.status}).`);
    const data = await response.json(); if (!Array.isArray(data) || !data.length) throw new Error('Patient data is empty.'); patients = data;
  } catch (error) { console.warn('Using fallback patient metadata.', error); }
  renderPatientOptions(); updateLog(); updateFindings(); await setPatient(0);
}
void init();
