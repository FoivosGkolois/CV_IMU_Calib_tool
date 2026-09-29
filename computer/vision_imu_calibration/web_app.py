from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .runtime import VisionRuntime


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision–IMU Calibration</title>
<style>
:root {
  color-scheme: dark;
  font-family: Inter, "Segoe UI", sans-serif;
  --bg:#0b0f16; --surface:#111925; --surface-2:#0d1420; --line:#26354b;
  --text:#eef4ff; --muted:#91a4bf; --blue:#39a0ff; --green:#51e26d; --red:#ff6464;
}
* { box-sizing:border-box; }
html, body { width:100%; height:100%; min-width:320px; overflow:hidden; }
body { margin:0; background:var(--bg); color:var(--text); }
#dashboardFit { width:100%; transform-origin:top left; }
main { display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr) minmax(300px,.8fr); gap:12px; padding:12px; width:100%; margin:0; align-items:start; }
.camera-column { min-width:0; width:100%; display:flex; justify-content:stretch; align-items:flex-start; }
.live-column { min-width:0; display:grid; grid-template-rows:auto auto; align-content:start; gap:10px; }
.video-shell { background:#05070b; border:1px solid var(--line); border-radius:14px; overflow:hidden; width:100%; min-width:0; aspect-ratio:16 / 9; align-self:start; display:flex; align-items:center; justify-content:center; }
.video-shell img { width:100%; height:100%; object-fit:contain; display:block; }
.live-summary { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; align-items:start; }
.summary-card, .card, .chart-card { background:var(--surface); border:1px solid var(--line); border-radius:12px; }
.summary-card { padding:10px 12px; min-width:0; max-width:100%; overflow:visible; }
.summary-card .value { font-size:clamp(16px,1.4vw,21px); }
.control-panel { min-width:0; display:grid; gap:10px; }
.card { padding:12px; min-width:0; max-width:100%; overflow:visible; }
.card.priority { border-color:#365374; box-shadow:0 0 0 1px rgba(57,160,255,.08); }
.card summary { cursor:pointer; list-style:none; display:flex; align-items:center; justify-content:space-between; gap:8px; }
.card summary::-webkit-details-marker { display:none; }
.card summary::after { content:"▾"; color:var(--muted); font-size:12px; }
.card:not([open]) summary::after { transform:rotate(-90deg); }
.label { color:var(--muted); font-size:10px; letter-spacing:.1em; text-transform:uppercase; }
.value { margin-top:5px; font-size:20px; font-weight:700; line-height:1.15; }
.ok { color:var(--green); } .bad { color:var(--red); } .rec { color:#ff7b7b; }
.small { font-size:12px; color:#d8e3f4; overflow-wrap:anywhere; }
.note { color:var(--muted); font-size:10.5px; line-height:1.4; margin-top:7px; overflow-wrap:anywhere; }
.controls { display:grid; gap:12px; margin-top:10px; }
.control-row { display:grid; gap:5px; }
.control-head { display:flex; justify-content:space-between; gap:10px; font-size:12px; color:#c9d5e7; }
.control-help { color:#8fa3bd; font-size:10px; line-height:1.35; }
.control-help b { color:#c9d5e7; font-weight:650; }
input[type=range] { width:100%; accent-color:var(--blue); }
select, button { width:100%; min-width:0; height:36px; border:1px solid #354764; border-radius:9px; background:var(--surface-2); color:var(--text); padding:0 10px; font:inherit; font-size:12px; }
button { cursor:pointer; font-weight:700; }
button.primary { background:#1767b2; border-color:#2789de; }
button.record { background:#8d2028; border-color:#d3414d; }
button.stop { background:#263348; }
button.secondary { background:#172234; }
button:disabled { opacity:.45; cursor:not-allowed; }
.button-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:9px; }
.inline-grid { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:7px; align-items:center; margin-top:8px; }
.inline-grid button { width:auto; min-width:78px; }
.path-link { display:block; width:100%; height:auto; min-height:34px; margin-top:8px; padding:7px 9px; text-align:left; text-decoration:none; white-space:normal; line-height:1.3; font-weight:600; font-size:12px; color:#a9d6ff; background:#0b1726; border:1px dashed #354764; border-radius:9px; overflow-wrap:anywhere; }
.path-link:hover { border-color:var(--blue); background:#10243b; }
.path-link[aria-disabled="true"] { opacity:.45; pointer-events:none; }
.metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:8px; }
.metric { background:var(--surface-2); border:1px solid #2a3952; border-radius:9px; padding:8px; min-width:0; }
.metric .name { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.07em; }
.metric .number { margin-top:4px; font-size:16px; font-weight:700; overflow-wrap:anywhere; }
.chart-card { padding:12px; height:clamp(230px,32vh,360px); min-width:0; max-width:100%; overflow:hidden; }
.recommendation-card { padding:11px 12px; min-width:0; }
.recommendation-values { margin-top:6px; color:#d8e3f4; font-size:11px; line-height:1.5; overflow-wrap:anywhere; }
.chart-shell { height:calc(100% - 48px); min-height:140px; margin-top:8px; }
.chart-shell canvas { width:100%; height:100%; display:block; background:#0a101a; border-radius:9px; }
.calibration-card { border-color:#6f5b25; box-shadow:0 0 0 1px rgba(255,194,77,.10); }
.calibration-actions { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:9px; }
.calibration-actions button { min-height:38px; height:auto; padding:7px 9px; }
.progress-track { height:8px; margin-top:9px; border-radius:999px; overflow:hidden; background:#0a101a; border:1px solid #2a3952; }
.progress-fill { height:100%; width:0%; background:#39a0ff; transition:width .25s linear; }
.modal-backdrop { position:fixed; inset:0; z-index:1000; display:flex; align-items:center; justify-content:center; padding:18px; background:rgba(3,7,12,.86); }
.modal-backdrop[hidden] { display:none; }
.modal-card { width:min(560px,100%); background:#111925; border:1px solid #3a4e6c; border-radius:16px; padding:18px; box-shadow:0 24px 80px rgba(0,0,0,.5); }
.modal-card h2 { margin:0 0 10px; font-size:20px; }
.modal-card ol { margin:10px 0 14px 22px; padding:0; color:#d8e3f4; line-height:1.55; font-size:13px; }
.countdown { min-height:52px; display:flex; align-items:center; justify-content:center; font-size:42px; font-weight:800; color:#ffcc66; }
.modal-buttons { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
footer { max-width:none; margin:0; padding:0 12px 12px; color:#73849c; font-size:10px; }
@media (max-width:1280px) {
  main { grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); }
  .control-panel { grid-column:1 / -1; grid-template-columns:repeat(2,minmax(0,1fr)); }
  .control-panel > .wide { grid-column:1 / -1; }
}
@media (max-width:860px) {
  main { grid-template-columns:1fr; padding:10px; gap:10px; }
  .camera-column { justify-content:flex-start; }
  .live-column { grid-template-rows:auto auto; }
  .control-panel { grid-template-columns:1fr; }
  .live-summary { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .chart-card { width:100%; height:250px; }
  .chart-shell { height:calc(100% - 48px); }
  .button-grid { grid-template-columns:1fr 1fr; }
}
@media (max-width:440px) {
  .live-summary { grid-template-columns:1fr; }
  .button-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<div id="dashboardFit">
<main>
<section class="camera-column">
  <div id="videoShell" class="video-shell"><img id="videoStream" src="/video.mjpg" alt="Live camera stream"></div>
</section>
<section class="live-column">
  <div class="live-summary">
    <div class="summary-card"><div class="label">Tag reference</div><div id="tracking" class="value">Starting…</div><div id="reason" class="small">Initializing camera and tag detector</div></div>
    <div class="summary-card"><div class="label">Reference angle</div><div id="angle" class="value">--.-°</div><div id="referenceNote" class="small">Camera calibration required</div></div>
    <div class="summary-card"><div class="label">IMU angle</div><div id="imuAngle" class="value">--.-°</div><div id="imuEstimatorState" class="small">NO SAMPLE</div></div>
    <div class="summary-card"><div class="label">Tags</div><div id="tagStatus" class="value">0 / 2</div><div id="fps" class="small">0.0 FPS</div></div>
  </div>
  <div class="chart-card">
    <div class="label">Live posterior knee-angle presentation · last 15 s</div>
    <div class="note">Blue: marker reference · Green: IMU estimate. Live display only; calibration scoring runs offline after both stages.</div>
    <div class="chart-shell"><canvas id="angleChart"></canvas></div>
  </div>
  <div class="recommendation-card card">
    <div class="label">Recommended IMU filter settings</div>
    <div id="imuRecommendationState" class="note">Complete both stages of one calibration run to generate data-driven settings.</div>
    <div id="imuRecommendationValues" class="recommendation-values"></div>
    <div id="imuValidationSummary" class="note"></div>
  </div>
</section>
<aside class="control-panel">
  <div class="card priority wide calibration-card">
    <div class="label">Guided IMU filter calibration</div>
    <div id="calibrationState" class="value small">READY</div>
    <div id="calibrationMessage" class="note">Start calibration with the static stage, then continue to the dynamic stage.</div>
    <div class="progress-track"><div id="calibrationProgress" class="progress-fill"></div></div>
    <div class="calibration-actions">
      <button id="startStaticTest" class="primary" onclick="showCalibrationInstructions('static')">Start calibration</button>
      <button id="startDynamicTest" class="primary" onclick="showCalibrationInstructions('dynamic')">Continue to dynamic calibration</button>
      <button id="stopCalibrationTest" class="stop" onclick="stopCalibrationTest()">Stop current test</button>
      <button id="cancelCalibrationTest" class="secondary" onclick="cancelCalibrationTest()">Cancel test</button>
    </div>
    <a id="calibrationPath" class="path-link" href="/exports/calibration/" target="_blank" rel="noopener">Calibration files will be exported to calibration_suggestions.</a>
  </div>

  <div class="card priority wide">
    <div class="label">Recording</div>
    <div id="recording" class="value small">IDLE</div>
    <div class="button-grid">
      <button id="startRecording" class="record" onclick="recordStart()">Start recording</button>
      <button id="stopRecording" class="stop" onclick="recordStop()">Stop</button>
    </div>
    <a id="recordingPath" class="path-link" href="/exports/recordings/" target="_blank" rel="noopener">Video recordings are finalized as ZIP files inside video_recordings.</a>
  </div>

  <div class="card wide">
    <div class="label">Raw IMU source</div>
    <div id="imuState" class="value small">DISCONNECTED</div>
    <div id="imuSerialControls">
      <div class="inline-grid"><select id="imuPort"><option value="">Select COM port</option></select><button class="secondary" onclick="refreshImuPorts()">Refresh</button></div>
      <div class="button-grid"><button id="imuConnect" class="primary" onclick="connectImu()">Connect</button><button id="imuDisconnect" class="stop" onclick="disconnectImu()">Disconnect</button></div>
    </div>
    <div id="imuMessage" class="note">Continuous 100 Hz acquisition; recording decides whether samples enter the final ZIP.</div>
    <div id="imuRate" class="note">Rate: 0.0 Hz · Port: —</div>
    <div id="imuSensors" class="note">Thigh: NOT READY · Shank: NOT READY</div>
    <div id="imuHealth" class="note">Samples: 0 · Gaps: 0 · Parse errors: 0 · Queue drops: 0</div>
  </div>


  <div class="card wide">
    <div class="label">Reference camera calibration</div>
    <div id="cameraCalibrationState" class="value small">IDLE</div>
    <div id="cameraCalibrationMessage" class="note">Use the printed 9×6 checkerboard before enabling the 3-D reference.</div>
    <div class="progress-track"><div id="cameraCalibrationProgress" class="progress-fill"></div></div>
    <div class="button-grid">
      <button id="startCameraCalibration" class="primary" onclick="cameraCalibrationStart()">Start checkerboard calibration</button>
      <button id="cancelCameraCalibration" class="secondary" onclick="cameraCalibrationCancel()">Cancel</button>
    </div>
    <div id="cameraCalibrationQuality" class="note">15 diverse views · 25 mm squares · persistent after quality acceptance</div>
  </div>

  <div class="card wide">
    <div class="label">One-time tag reference validation</div>
    <div id="referenceValidationState" class="value small">NOT PERFORMED</div>
    <div id="referenceValidationMessage" class="note">After camera calibration, mount Tag 0 and Tag 1 on a rigid hinged goniometer/fixture.</div>
    <div class="progress-track"><div id="referenceValidationProgress" class="progress-fill"></div></div>
    <div class="button-grid">
      <button id="startReferenceValidation" class="primary" onclick="referenceValidationStart()">Start known-angle validation</button>
      <button id="captureReferenceValidation" class="primary" onclick="referenceValidationCapture()">Capture current angle</button>
      <button id="cancelReferenceValidation" class="secondary" onclick="referenceValidationCancel()">Cancel</button>
    </div>
    <div id="referenceValidationQuality" class="note">Targets: 180°, 150°, 120°, 90° · 5 captures each · one-time persistent result</div>
  </div>

  <div class="card">
    <div class="label">Camera profile</div>
    <select id="cameraProfile" style="margin-top:8px"><option value="fast">Fast · 640×480</option><option value="quality">Quality · 1280×720</option></select>
    <button id="applyCameraProfile" class="primary" style="margin-top:8px" onclick="setCameraProfile()">Apply profile</button>
    <div id="cameraProfileMessage" class="note"><b>Fast:</b> lower load. <b>Quality:</b> more detail at difficult angles.</div>
  </div>

  <details class="card" open>
    <summary><span class="label">IMU estimator details</span></summary>
    <div class="metric-grid">
      <div class="metric"><div class="name">Visual − IMU</div><div id="angleDifference" class="number">--.-°</div></div>
      <div class="metric"><div class="name">Knee speed</div><div id="imuSpeed" class="number">--.-°/s</div></div>
      <div class="metric"><div class="name">Thigh / Shank</div><div id="segmentAngles" class="number">--.- / --.-°</div></div>
      <div class="metric"><div class="name">State</div><div class="number">Live</div></div>
    </div>
    <button id="reloadImuFilter" class="secondary" style="margin-top:8px" onclick="reloadFilterConfig()">Reload IMU filter config</button>
    <div id="imuFilterInfo" class="note">Waiting for synchronized raw IMU samples.</div>
  </details>

  <details class="card wide" open>
    <summary><span class="label">Tag reference</span></summary>
    <div id="thighTagState" class="note">Thigh · Tag 0: NOT DETECTED</div>
    <div id="shankTagState" class="note">Shank · Tag 1: NOT DETECTED</div>
    <div id="referenceQuality" class="note">3-D pose quality: waiting for calibrated camera setup.</div>
    <div class="note">The calibrated 3-D relative-angle geometry uses the persistent checkerboard camera setup and fixed quality gates.</div>
  </details>

  <details class="card wide">
    <summary><span class="label">Runtime diagnostics</span></summary>
    <div id="runtime" class="value small">STARTING</div>
    <div id="backend" class="note">Backend: AprilTag 36h11</div>
    <div id="source" class="note">Source: —</div>
    <div id="frameCount" class="note">Frames: 0</div>
  </details>
</aside>
</main>
<footer>Final protocol: validated two-tag geometric reference · physics-based static calibration · 12-cycle multi-speed tuning · GP Bayesian Optimization · held-out validation.</footer>
</div>
<div id="calibrationModal" class="modal-backdrop" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="calibrationModalTitle">
    <h2 id="calibrationModalTitle">Calibration test</h2>
    <div id="calibrationModalIntro" class="note"></div>
    <ol id="calibrationInstructions"></ol>
    <div id="calibrationCountdown" class="countdown">Ready</div>
    <div class="modal-buttons">
      <button id="beginCalibrationCountdown" class="primary" onclick="beginCalibrationCountdown()">Start 5-second countdown</button>
      <button id="closeCalibrationModal" class="secondary" onclick="closeCalibrationModal()">Back</button>
    </div>
  </div>
</div>
<script>
const reason=document.getElementById('reason');
const angle=document.getElementById('angle');
const referenceNote=document.getElementById('referenceNote');
const tagStatus=document.getElementById('tagStatus');
const thighTagState=document.getElementById('thighTagState');
const shankTagState=document.getElementById('shankTagState');
const referenceQuality=document.getElementById('referenceQuality');
const fps=document.getElementById('fps');
const source=document.getElementById('source');
const backend=document.getElementById('backend');
const runtime=document.getElementById('runtime');
const frameCount=document.getElementById('frameCount');
const recording=document.getElementById('recording');
const startRecording=document.getElementById('startRecording');
const stopRecording=document.getElementById('stopRecording');
const recordingPath=document.getElementById('recordingPath');
const calibrationState=document.getElementById('calibrationState');
const imuValidationSummary=document.getElementById('imuValidationSummary');
const calibrationMessage=document.getElementById('calibrationMessage');
const calibrationProgress=document.getElementById('calibrationProgress');
const startStaticTest=document.getElementById('startStaticTest');
const startDynamicTest=document.getElementById('startDynamicTest');
const stopCalibrationTestButton=document.getElementById('stopCalibrationTest');
const cancelCalibrationTestButton=document.getElementById('cancelCalibrationTest');
const calibrationPath=document.getElementById('calibrationPath');
const calibrationModal=document.getElementById('calibrationModal');
const calibrationModalTitle=document.getElementById('calibrationModalTitle');
const calibrationModalIntro=document.getElementById('calibrationModalIntro');
const calibrationInstructions=document.getElementById('calibrationInstructions');
const calibrationCountdown=document.getElementById('calibrationCountdown');
const beginCalibrationCountdownButton=document.getElementById('beginCalibrationCountdown');
const closeCalibrationModalButton=document.getElementById('closeCalibrationModal');
const imuState=document.getElementById('imuState');
const imuRate=document.getElementById('imuRate');
const imuSensors=document.getElementById('imuSensors');
const imuHealth=document.getElementById('imuHealth');
const imuSerialControls=document.getElementById('imuSerialControls');
const imuPort=document.getElementById('imuPort');
const imuConnect=document.getElementById('imuConnect');
const imuDisconnect=document.getElementById('imuDisconnect');
const imuMessage=document.getElementById('imuMessage');
const cameraProfile=document.getElementById('cameraProfile');
const applyCameraProfile=document.getElementById('applyCameraProfile');
const cameraProfileMessage=document.getElementById('cameraProfileMessage');
const imuEstimatorState=document.getElementById('imuEstimatorState');
const imuAngle=document.getElementById('imuAngle');
const angleDifference=document.getElementById('angleDifference');
const imuSpeed=document.getElementById('imuSpeed');
const segmentAngles=document.getElementById('segmentAngles');
const imuFilterInfo=document.getElementById('imuFilterInfo');
const imuRecommendationState=document.getElementById('imuRecommendationState');
const imuRecommendationValues=document.getElementById('imuRecommendationValues');
const reloadImuFilter=document.getElementById('reloadImuFilter');
const angleChart=document.getElementById('angleChart');
const videoShell=document.getElementById('videoShell');
const videoStream=document.getElementById('videoStream');
const dashboardFit=document.getElementById('dashboardFit');
let dashboardFitPending=false;
function fitDashboardToViewport(){
  if(!dashboardFit) return;
  dashboardFit.style.zoom='1';
  const naturalWidth=Math.max(1,dashboardFit.scrollWidth);
  const naturalHeight=Math.max(1,dashboardFit.scrollHeight);
  const scale=Math.min(1, window.innerWidth/naturalWidth, window.innerHeight/naturalHeight);
  dashboardFit.style.zoom=String(Math.max(0.1,scale));
  dashboardFitPending=false;
}
function scheduleDashboardFit(){
  if(dashboardFitPending) return;
  dashboardFitPending=true;
  window.requestAnimationFrame(fitDashboardToViewport);
}
function resizeVideoShell(){
  const naturalWidth=Number(videoStream.naturalWidth||0);
  const naturalHeight=Number(videoStream.naturalHeight||0);
  if(!naturalWidth||!naturalHeight) return;
  videoShell.style.aspectRatio=String(naturalWidth/naturalHeight);
  drawAngleChart(window.latestAngleRows||[]);
}
videoStream.addEventListener('load',resizeVideoShell);
window.addEventListener('resize',()=>{ resizeVideoShell(); drawAngleChart(window.latestAngleRows||[]); scheduleDashboardFit(); },{passive:true});
if('ResizeObserver' in window){
  const observer=new ResizeObserver(()=>drawAngleChart(window.latestAngleRows||[]));
  observer.observe(angleChart.parentElement);
}
async function refreshImuPorts(){
  const selected=imuPort.value;
  try {
    const r=await fetch('/api/imu/ports',{cache:'no-store'}); const body=await r.json();
    if(!r.ok) throw new Error(body.error||'port scan failed');
    imuPort.innerHTML='<option value="">Select COM port</option>';
    for(const port of body.ports||[]){
      const option=document.createElement('option');
      option.value=port.device;
      option.textContent=port.description?`${port.device} · ${port.description}`:port.device;
      imuPort.appendChild(option);
    }
    if([...imuPort.options].some(o=>o.value===selected)) imuPort.value=selected;
    window.imuPortsLoaded=true;
    imuMessage.textContent=(body.ports||[]).length?`${body.ports.length} serial port(s) found.`:'No serial ports found.';
  } catch(e) { window.imuPortsLoaded=true; imuMessage.textContent='Port scan failed: '+e; }
}
async function connectImu(){
  if(!imuPort.value){ imuMessage.textContent='Select a COM port first.'; return; }
  const r=await fetch('/api/imu/connect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port:imuPort.value,baud_rate:921600})});
  const body=await r.json();
  imuMessage.textContent=r.ok?`Connected to ${imuPort.value}; continuous IMU acquisition is active.`:'Connect failed: '+(body.error||'unknown error');
}
async function disconnectImu(){
  const r=await fetch('/api/imu/disconnect',{method:'POST'}); const body=await r.json();
  imuMessage.textContent=r.ok?'IMU serial disconnected.':'Disconnect failed: '+(body.error||'unknown error');
}
async function setCameraProfile(){
  applyCameraProfile.disabled=true;
  cameraProfileMessage.textContent='Restarting camera with '+cameraProfile.value+' profile…';
  try {
    const r=await fetch('/api/camera/profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:cameraProfile.value})});
    const body=await r.json();
    if(!r.ok) throw new Error(body.error||'profile change failed');
    cameraProfileMessage.textContent=`Applied ${body.profile}: ${body.width}×${body.height} @ ${Number(body.fps).toFixed(0)} FPS.`;
  } catch(e) { cameraProfileMessage.textContent='Camera profile failed: '+e; }
  finally { applyCameraProfile.disabled=false; setTimeout(resizeVideoShell,350); }
}
async function cameraCalibrationStart(){
  cameraCalibrationMessage.textContent='Starting checkerboard capture…';
  const r=await fetch('/api/reference/camera-calibration/start',{method:'POST'}); const body=await r.json();
  cameraCalibrationMessage.textContent=r.ok?(body.message||'Checkerboard calibration started.'):('Camera calibration failed to start: '+(body.error||'unknown error'));
}
async function cameraCalibrationCancel(){
  const r=await fetch('/api/reference/camera-calibration/cancel',{method:'POST'}); const body=await r.json();
  cameraCalibrationMessage.textContent=r.ok?(body.message||'Checkerboard calibration cancelled.'):('Cancel failed: '+(body.error||'unknown error'));
}
async function referenceValidationStart(){
  const r=await fetch('/api/reference/known-angle-validation/start',{method:'POST'}); const body=await r.json();
  referenceValidationMessage.textContent=r.ok?(body.message||'Known-angle validation started.'):('Start failed: '+(body.error||'unknown error'));
}
async function referenceValidationCapture(){
  referenceValidationMessage.textContent='Capturing the last second of stable tag data…';
  const r=await fetch('/api/reference/known-angle-validation/capture',{method:'POST'}); const body=await r.json();
  referenceValidationMessage.textContent=r.ok?(body.message||'Capture accepted.'):('Capture failed: '+(body.error||'unknown error'));
}
async function referenceValidationCancel(){
  const r=await fetch('/api/reference/known-angle-validation/cancel',{method:'POST'}); const body=await r.json();
  referenceValidationMessage.textContent=r.ok?(body.message||'Known-angle validation cancelled.'):('Cancel failed: '+(body.error||'unknown error'));
}

const CALIBRATION_GUIDANCE={
  static:{
    title:'Static calibration test',
    intro:'The tool records for 12 seconds and then stops automatically.',
    steps:[
      'Wear both IMUs firmly in the same mounting that will be used for the dynamic test.',
      'Stand with the calibrated leg fully extended: the physical reference condition is 180°.',
      'Remain completely still for the full 12 seconds. AprilTags are optional in this static stage and are used only as a sanity check.'
    ]
  },
  dynamic:{
    title:'Dynamic calibration test',
    intro:'The tool records for up to 60 seconds. The final protocol uses exactly 12 accepted complete cycles.',
    steps:[
      'Keep exactly the same IMU, tag, and camera mounting used for the static test.',
      'Keep both AprilTags visible and wait for Reference VALID.',
      'Perform 4 deliberately slow complete flexion–extension cycles, then 4 comfortable/normal cycles, then 4 brisk cycles. The tool uses 9 for tuning and keeps one from each speed block unseen for the final check.'
    ]
  }
};
let pendingCalibrationType='';
let calibrationCountdownTimer=null;
function showCalibrationInstructions(testType){
  const guide=CALIBRATION_GUIDANCE[testType]; if(!guide) return;
  pendingCalibrationType=testType;
  calibrationModalTitle.textContent=guide.title;
  calibrationModalIntro.textContent=guide.intro;
  calibrationInstructions.innerHTML='';
  for(const step of guide.steps){ const item=document.createElement('li'); item.textContent=step; calibrationInstructions.appendChild(item); }
  calibrationCountdown.textContent='Ready';
  beginCalibrationCountdownButton.disabled=false;
  closeCalibrationModalButton.disabled=false;
  calibrationModal.hidden=false;
}
function closeCalibrationModal(){
  if(calibrationCountdownTimer!==null) return;
  calibrationModal.hidden=true; pendingCalibrationType='';
}
async function beginCalibrationCountdown(){
  if(!pendingCalibrationType||calibrationCountdownTimer!==null) return;
  beginCalibrationCountdownButton.disabled=true; closeCalibrationModalButton.disabled=true;
  let remaining=5; calibrationCountdown.textContent=String(remaining);
  calibrationCountdownTimer=window.setInterval(async()=>{
    remaining-=1;
    if(remaining>0){ calibrationCountdown.textContent=String(remaining); return; }
    window.clearInterval(calibrationCountdownTimer); calibrationCountdownTimer=null;
    calibrationCountdown.textContent='START';
    const testType=pendingCalibrationType;
    try{
      const r=await fetch('/api/calibration/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({test_type:testType})});
      const body=await r.json();
      if(!r.ok) throw new Error(body.error||'calibration start failed');
      calibrationModal.hidden=true; pendingCalibrationType='';
      calibrationMessage.textContent=body.message||`${testType} test started.`;
    }catch(e){
      calibrationCountdown.textContent='Could not start';
      calibrationMessage.textContent='Calibration start failed: '+e;
      beginCalibrationCountdownButton.disabled=false; closeCalibrationModalButton.disabled=false;
    }
  },1000);
}
async function stopCalibrationTest(){
  calibrationMessage.textContent='Stopping and finalizing the current calibration test…';
  const r=await fetch('/api/calibration/stop',{method:'POST'}); const body=await r.json();
  calibrationMessage.textContent=r.ok?(body.message||'Calibration test stopped.'):('Stop failed: '+(body.error||'unknown error'));
}
async function cancelCalibrationTest(){
  const r=await fetch('/api/calibration/cancel',{method:'POST'}); const body=await r.json();
  calibrationMessage.textContent=r.ok?(body.message||'Calibration test cancelled.'):('Cancel failed: '+(body.error||'unknown error'));
}

async function recordStart(){
  recordingPath.textContent='Starting video recording in video_recordings…';
  const r=await fetch('/api/recording/start',{method:'POST'});
  const body=await r.json();
  recordingPath.textContent=r.ok?'Recording started. The export link will update automatically.':('Start failed: '+(body.error||'unknown error'));
}
async function recordStop(){
  recordingPath.textContent='Finalizing synchronized data and ZIP…';
  const r=await fetch('/api/recording/stop',{method:'POST'});
  const body=await r.json();
  recordingPath.textContent=r.ok?'Recording finalized. The clickable export link will update automatically.':('Stop failed: '+(body.error||'unknown error'));
}
async function reloadFilterConfig(){
  reloadImuFilter.disabled=true;
  const r=await fetch('/api/imu-filter/reload',{method:'POST'}); const body=await r.json();
  imuFilterInfo.textContent=r.ok?'IMU filter parameters reloaded and estimator state reset.':'Reload failed: '+(body.error||'unknown error');
  reloadImuFilter.disabled=false;
}
function drawAngleChart(rows){
  const rect=angleChart.getBoundingClientRect(); const dpr=window.devicePixelRatio||1;
  const width=Math.max(320,Math.floor(rect.width)); const height=Math.max(160,Math.floor(rect.height));
  angleChart.width=Math.floor(width*dpr); angleChart.height=Math.floor(height*dpr);
  const ctx=angleChart.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
  ctx.fillStyle='#0a101a'; ctx.fillRect(0,0,width,height);
  if(!rows||rows.length<2){ ctx.fillStyle='#91a4bf'; ctx.font='13px Segoe UI'; ctx.fillText('Waiting for camera and IMU angle samples…',16,28); return; }
  const values=[]; for(const row of rows){ if(row.visual_knee_angle_deg!==null&&Number.isFinite(Number(row.visual_knee_angle_deg))) values.push(Number(row.visual_knee_angle_deg)); if(row.imu_knee_angle_deg!==null&&Number.isFinite(Number(row.imu_knee_angle_deg))) values.push(Number(row.imu_knee_angle_deg)); }
  if(!values.length) return;
  let min=Math.min(...values), max=Math.max(...values); if(max-min<20){ const mid=(max+min)/2; min=mid-10; max=mid+10; }
  const left=42,right=12,top=12,bottom=28; const plotW=width-left-right, plotH=height-top-bottom;
  ctx.strokeStyle='#27364d'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){ const y=top+plotH*i/4; ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(width-right,y); ctx.stroke(); const val=max-(max-min)*i/4; ctx.fillStyle='#8194ae'; ctx.font='10px Segoe UI'; ctx.fillText(val.toFixed(0)+'°',4,y+3); }
  const t0=Number(rows[0].host_monotonic_ns), t1=Number(rows[rows.length-1].host_monotonic_ns); const dt=Math.max(1,t1-t0);
  function line(key,validKey,color){ ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); let drawing=false; for(const row of rows){ const value=Number(row[key]); const valid=Boolean(row[validKey]); if(!valid||row[key]===null||!Number.isFinite(value)){ drawing=false; continue; } const x=left+(Number(row.host_monotonic_ns)-t0)/dt*plotW; const y=top+(max-value)/(max-min)*plotH; if(!drawing){ctx.moveTo(x,y);drawing=true;}else ctx.lineTo(x,y);} ctx.stroke(); }
  line('visual_knee_angle_deg','visual_valid','#39a0ff'); line('imu_knee_angle_deg','imu_valid','#51e26d');
  ctx.fillStyle='#91a4bf'; ctx.font='10px Segoe UI'; ctx.fillText('−15 s',left,height-8); ctx.fillText('now',width-right-22,height-8);
}
async function refreshHistory(){ try{ const r=await fetch('/api/acquisition/history?seconds=15&max_points=300',{cache:'no-store'}); const body=await r.json(); if(r.ok){ window.latestAngleRows=body.rows||[]; drawAngleChart(window.latestAngleRows); } }catch(e){} }
function updateImuRecommendations(cal){
  const p=(cal&&cal.recommended_parameters)||{};
  if(cal&&cal.valid&&Object.keys(p).length){
    imuRecommendationState.textContent='Generated from the completed static and dynamic tests.';
    imuRecommendationState.className='note ok';
    imuRecommendationValues.textContent=
      `α stable ${Number(p.alpha_stable).toFixed(3)} · α dynamic ${Number(p.alpha_dynamic).toFixed(3)} · `+
      `deadband ${Number(p.gyro_deadband_deg_s).toFixed(2)}°/s · straight offset ${Number(p.straight_leg_offset_deg).toFixed(2)}° · `+
      `gyro bias thigh/shank ${Number(p.thigh_gyro_bias_deg_s).toFixed(3)} / ${Number(p.shank_gyro_bias_deg_s).toFixed(3)}°/s · `+
      `max accel correction ${Number(p.max_accel_correction_deg).toFixed(1)}° · relative sign ${Number(p.relative_angle_sign).toFixed(0)}`;
    const summary=cal.result_summary||{};
    const ci=summary.held_out_rmse_ci95_deg;
    imuValidationSummary.textContent=Object.keys(summary).length ?
      `${summary.optimizer||'Optimizer'} · held-out RMSE ${Number(summary.held_out_rmse_deg).toFixed(2)}° · `+
      `MAE ${Number(summary.held_out_mae_deg).toFixed(2)}° · lag ${Number(summary.held_out_lag_ms).toFixed(0)} ms · `+
      `RMSE change ${Number(summary.held_out_rmse_improvement_percent).toFixed(1)}%`+
      (Array.isArray(ci)?` · 95% CI ${Number(ci[0]).toFixed(2)}–${Number(ci[1]).toFixed(2)}°`:'') : '';
    return;
  }
  const staticSuggestion=(cal&&cal.static_suggestion)||{};
  if(cal&&cal.state==='AWAITING_DYNAMIC'&&Object.keys(staticSuggestion).length){
    imuRecommendationState.textContent='Static-stage suggestion saved. Continue to the dynamic stage for the final settings.';
    imuRecommendationState.className='note ok';
    imuRecommendationValues.textContent=
      `static deadband ${Number(staticSuggestion.gyro_deadband_deg_s).toFixed(2)}°/s · `+
      `straight offset ${Number(staticSuggestion.straight_leg_offset_deg).toFixed(2)}° · `+
      `gyro bias thigh/shank ${Number(staticSuggestion.thigh_gyro_bias_deg_s).toFixed(3)} / ${Number(staticSuggestion.shank_gyro_bias_deg_s).toFixed(3)}°/s`;
    return;
  }
  imuRecommendationState.className='note';
  imuRecommendationValues.textContent='';
  imuValidationSummary.textContent='';
  if(cal&&String(cal.state||'').startsWith('ANALYZING')) imuRecommendationState.textContent='Analyzing calibration data and updating the shared suggestions YAML…';
  else if(cal&&(cal.state==='INVALID'||cal.state==='FAILED')) imuRecommendationState.textContent='No valid recommendation was generated. Review the calibration result and repeat the failed test.';
  else imuRecommendationState.textContent='Complete both stages of one calibration run to generate data-driven settings.';
}
async function refresh(){
  try {
    const r=await fetch('/api/status',{cache:'no-store'}); const s=await r.json();
    const t=document.getElementById('tracking');
    t.textContent=s.tracking_valid?'VALID':'INVALID'; t.className='value '+(s.tracking_valid?'ok':'bad');
    reason.textContent=s.tracking_reason||'—';
    angle.textContent=s.visual_knee_angle_deg==null?'--.-°':Number(s.visual_knee_angle_deg).toFixed(1)+'°';
    const referenceConfig=s.reference||{};
    referenceNote.textContent=s.tracking_valid?'Calibrated 3-D tag reference':(referenceConfig.setup_required?'Camera calibration required':'Reference quality gate failed');
    const detected=Number(Boolean(s.thigh_tag_detected))+Number(Boolean(s.shank_tag_detected));
    tagStatus.textContent=detected+' / 2';
    tagStatus.className='value '+(detected===2?'ok':'bad');
    thighTagState.textContent='Thigh · Tag 0: '+(s.thigh_tag_detected?'DETECTED':'NOT DETECTED');
    thighTagState.className='note '+(s.thigh_tag_detected?'ok':'bad');
    shankTagState.textContent='Shank · Tag 1: '+(s.shank_tag_detected?'DETECTED':'NOT DETECTED');
    shankTagState.className='note '+(s.shank_tag_detected?'ok':'bad');
    const fmtMetric=(value,suffix)=>value==null?'—':Number(value).toFixed(2)+suffix;
    referenceQuality.textContent='3-D pose · thigh reproj '+fmtMetric(s.thigh_reprojection_rmse_px,' px')+' / view '+fmtMetric(s.thigh_view_angle_deg,'°')+' · shank reproj '+fmtMetric(s.shank_reprojection_rmse_px,' px')+' / view '+fmtMetric(s.shank_view_angle_deg,'°');
    fps.textContent=Number(s.processing_fps||0).toFixed(1)+' FPS';
    source.textContent='Source: '+(s.source||'—');
    backend.textContent='Backend: '+(s.reference_backend||'—');
    runtime.textContent=s.error?`${s.state}: ${s.error}`:s.state;
    frameCount.textContent='Frames: '+Number(s.frame_index||0);
    const imu=s.imu||{};
    imuState.textContent=imu.configured?(imu.state||'UNKNOWN'):'DISCONNECTED';
    imuState.className='value small '+(imu.connected?'ok':(imu.configured?'bad':''));
    imuRate.textContent='Rate: '+Number(imu.sample_rate_hz||0).toFixed(1)+' Hz · Source: '+(imu.port||'—');
    imuSensors.textContent='Thigh: '+(imu.thigh_ready?'READY':'NOT READY')+' · Shank: '+(imu.shank_ready?'READY':'NOT READY');
    imuHealth.textContent='Samples: '+Number(imu.sample_count||0)+' · Gaps: '+Number(imu.sequence_gaps||0)+' · Parse errors: '+Number(imu.parse_errors||0)+' · Queue drops: '+Number(imu.queue_drops||0);
    if(imu.last_error) imuHealth.textContent+=' · Error: '+imu.last_error;
    const est=s.imu_estimator||{};
    imuEstimatorState.textContent=est.valid?'VALID':String(est.filter_reason||est.reason||'NO SAMPLE').toUpperCase();
    imuEstimatorState.className='small '+(est.valid?'ok':'bad');
    imuAngle.textContent=est.knee_angle_deg==null?'--.-°':Number(est.knee_angle_deg).toFixed(1)+'°';
    imuSpeed.textContent=est.knee_speed_deg_s==null?'--.-°/s':Number(est.knee_speed_deg_s).toFixed(1)+'°/s';
    segmentAngles.textContent=(est.thigh_angle_deg==null?'--.-':Number(est.thigh_angle_deg).toFixed(1))+' / '+(est.shank_angle_deg==null?'--.-':Number(est.shank_angle_deg).toFixed(1))+'°';
    const visual=Number(s.visual_knee_angle_deg), imuValue=Number(est.knee_angle_deg);
    angleDifference.textContent=(s.tracking_valid&&est.valid&&Number.isFinite(visual)&&Number.isFinite(imuValue))?(visual-imuValue).toFixed(1)+'°':'--.-°';
    const params=est.parameters||{};
    imuFilterInfo.textContent='Config: '+(est.config_path||'defaults')+' · axis '+(params.rotation_axis_name||'—')+' · α '+Number(params.alpha_stable||0).toFixed(3)+' / '+Number(params.alpha_dynamic||0).toFixed(3)+' · offset '+Number(params.straight_leg_offset_deg||0).toFixed(1)+'°';
    const calibrationLocked=Boolean((s.calibration||{}).active)||String((s.calibration||{}).state||'').startsWith('ANALYZING');
    reloadImuFilter.disabled=Boolean((s.recording||{}).active)||calibrationLocked;
    const bridgeManaged=imu.transport==='uno-q-bridge';
    imuSerialControls.hidden=bridgeManaged;
    imuConnect.disabled=bridgeManaged||Boolean(imu.configured)||Boolean((s.recording||{}).active)||calibrationLocked;
    imuDisconnect.disabled=bridgeManaged||!Boolean(imu.configured)||Boolean((s.recording||{}).active)||calibrationLocked;
    imuPort.disabled=bridgeManaged;
    if(bridgeManaged) imuMessage.textContent='UNO Q Bridge is active; IMU data stays inside the board.';
    else if(!window.imuPortsLoaded) refreshImuPorts();
    if(imu.port && [...imuPort.options].some(o=>o.value===imu.port)) imuPort.value=imu.port;
    const camera=s.camera||{};
    if(!window.cameraProfileLoaded && camera.profile){ cameraProfile.value=camera.profile; window.cameraProfileLoaded=true; }
    const camCal=s.camera_calibration||{};
    cameraCalibrationState.textContent=String(camCal.state||'IDLE').replaceAll('_',' ');
    cameraCalibrationState.className='value small '+(camCal.state==='COMPLETE'?'ok':(camCal.state==='FAILED'?'bad':''));
    cameraCalibrationMessage.textContent=camCal.message||'Use the printed checkerboard before enabling the 3-D reference.';
    const acceptedViews=Number(camCal.accepted_views||0), requiredViews=Math.max(1,Number(camCal.required_views||15));
    cameraCalibrationProgress.style.width=(100*Math.min(1,acceptedViews/requiredViews)).toFixed(1)+'%';
    const detectedText=camCal.checkerboard_detected?'checkerboard detected':'checkerboard not detected';
    const rmsText=camCal.rms_reprojection_error_px==null?'RMS —':('RMS '+Number(camCal.rms_reprojection_error_px).toFixed(2)+' px');
    const persistentState=String(camCal.persistent_state||'NOT LOADED').replaceAll('_',' ');
    const savedText=camCal.saved_utc?(' · saved '+camCal.saved_utc):'';
    cameraCalibrationQuality.textContent=`${acceptedViews}/${requiredViews} views · ${detectedText} · ${rmsText} · persistent ${persistentState}${savedText}`;
    const cameraCalibrationBusy=Boolean(camCal.active);
    startCameraCalibration.disabled=cameraCalibrationBusy||Boolean((s.recording||{}).active)||calibrationLocked||!['CAMERA_PREVIEW','RUNNING','REFERENCE_FAILED'].includes(s.state);
    cancelCameraCalibration.disabled=!cameraCalibrationBusy;
    const refVal=s.reference_validation||{};
    const refValState=String(refVal.state||'IDLE');
    referenceValidationState.textContent=refValState.replaceAll('_',' ');
    referenceValidationState.className='value small '+(refVal.persisted_passed?'ok':(refValState.includes('FAILED')?'bad':''));
    referenceValidationMessage.textContent=refVal.message||'Run once after camera calibration using the rigid known-angle fixture.';
    referenceValidationProgress.style.width=(100*Number(refVal.progress||0)).toFixed(1)+'%';
    const currentTarget=refVal.current_target_deg==null?'—':Number(refVal.current_target_deg).toFixed(0)+'°';
    const refMetrics=(refVal.result||{}).metrics||{};
    const metricText=refMetrics.rmse_deg==null?'RMSE —':`RMSE ${Number(refMetrics.rmse_deg).toFixed(2)}° · repeatability ${Number(refMetrics.max_repeatability_sd_deg).toFixed(2)}°`;
    referenceValidationQuality.textContent=`Target ${currentTarget} · ${Number(refVal.completed_captures||0)}/${Number(refVal.required_captures||20)} captures · ${metricText}`;
    startReferenceValidation.disabled=Boolean(refVal.active)||cameraCalibrationBusy||Boolean((s.recording||{}).active)||calibrationLocked||camCal.persistent_state!=='LOADED';
    captureReferenceValidation.disabled=!Boolean(refVal.active)||!Boolean(s.tracking_valid);
    cancelReferenceValidation.disabled=!Boolean(refVal.active);
    applyCameraProfile.disabled=Boolean((s.recording||{}).active)||calibrationLocked||cameraCalibrationBusy||Boolean(refVal.active)||camera.reconfigurable===false||!['CAMERA_PREVIEW','RUNNING','REFERENCE_FAILED'].includes(s.state);
    const cal=s.calibration||{};
    calibrationState.textContent=String(cal.state||'IDLE').replaceAll('_',' ');
    calibrationState.className='value small '+(cal.valid?'ok':(cal.state==='FAILED'||cal.state==='INVALID'?'bad':''));
    calibrationMessage.textContent=cal.error?`${cal.message||''} Error: ${cal.error}`:(cal.message||'Start calibration with the static stage.');
    if(!cal.workflow_ready) calibrationMessage.textContent='Scientific IMU calibration remains locked during the V2 migration. Reference recording and inspection remain available.';
    calibrationProgress.style.width=(100*Number(cal.progress||0)).toFixed(1)+'%';
    updateImuRecommendations(cal);
    const calibrationBusy=Boolean(cal.active)||String(cal.state||'').startsWith('ANALYZING');
    const imuReady=Boolean(cal.workflow_ready)&&Boolean(imu.connected)&&Boolean(imu.thigh_ready&&imu.shank_ready);
    const setupReady=camCal.persistent_state==='LOADED'&&Boolean(refVal.persisted_passed);
    const dynamicReady=imuReady&&setupReady&&Boolean(s.tracking_valid);
    startStaticTest.disabled=calibrationBusy||Boolean((s.recording||{}).active)||!imuReady||!setupReady||cal.state==='AWAITING_DYNAMIC';
    startStaticTest.textContent=(cal.state==='COMPLETE'||cal.state==='FAILED')?'Start new calibration':'Start calibration';
    startDynamicTest.disabled=calibrationBusy||Boolean((s.recording||{}).active)||!dynamicReady||!Boolean(cal.static_suggestion&&Object.keys(cal.static_suggestion).length)||!['AWAITING_DYNAMIC','INVALID'].includes(cal.state);
    stopCalibrationTestButton.disabled=!Boolean(cal.active);
    cancelCalibrationTestButton.disabled=!Boolean(cal.active);
    const calibrationHref=cal.recommendation_url||cal.browse_url||'/exports/calibration/';
    calibrationPath.href=calibrationHref;
    calibrationPath.setAttribute('aria-disabled',calibrationHref?'false':'true');
    if(cal.recommendation_display_path) calibrationPath.textContent='Open calibration suggestions: '+cal.recommendation_display_path;
    else calibrationPath.textContent='Open calibration exports: '+(cal.display_path||'calibration_suggestions');
    const rec=s.recording||{};
    recording.textContent=rec.active?`RECORDING · ${Number(rec.elapsed_s||0).toFixed(1)} s · ${rec.frame_count||0} camera / ${rec.imu_sample_count||0} IMU samples`:'IDLE';
    recording.className='value small '+(rec.active?'rec':'');
    startRecording.disabled=Boolean(rec.active)||calibrationBusy||cameraCalibrationBusy; stopRecording.disabled=!rec.active||Boolean(cal.active);
    const recordingHref=(!rec.active&&rec.download_url)?rec.download_url:(rec.browse_url||'/exports/recordings/');
    recordingPath.href=recordingHref;
    recordingPath.setAttribute('aria-disabled',recordingHref?'false':'true');
    if(rec.display_path) recordingPath.textContent=(rec.active?'ZIP destination: ':'Open saved ZIP: ')+rec.display_path;
    else recordingPath.textContent='Open video recordings: video_recordings';
    if(rec.error) recordingPath.textContent='Recorder error: '+rec.error;
  } catch(e) { runtime.textContent='UI connection error: '+e; }
  scheduleDashboardFit();
}
setInterval(refresh,350); setInterval(refreshHistory,700); refresh(); refreshHistory(); resizeVideoShell(); scheduleDashboardFit();
</script>
</body>
</html>
'''


def create_app(runtime: VisionRuntime) -> Any:
    try:
        from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
    except ImportError as exc:
        raise RuntimeError("Flask is not installed. Run 08_SETUP_VISION_IMU_PC.bat") from exc

    app = Flask(__name__)
    recording_root = Path(
        getattr(runtime, "recording_root", Path.cwd() / "video_recordings")
    ).expanduser().resolve()
    calibration_root = Path(
        getattr(runtime, "calibration_root", Path.cwd() / "calibration_suggestions")
    ).expanduser().resolve()

    @app.get("/")
    def index() -> str:
        return _PAGE

    @app.get("/api/status")
    def status() -> Any:
        return jsonify(runtime.status())

    @app.get("/exports/recordings/")
    def recordings_index() -> Any:
        return _export_directory_page(
            recording_root,
            recording_root,
            route_prefix="/exports/recordings/",
            title="Video recordings",
        )

    @app.get("/exports/recordings/<path:relative_path>")
    def recordings_export(relative_path: str) -> Any:
        target = _safe_export_target(recording_root, relative_path)
        if target is None or not target.exists():
            return "Export not found", 404
        if target.is_dir():
            return _export_directory_page(
                recording_root,
                target,
                route_prefix="/exports/recordings/",
                title="Video recordings",
            )
        return send_from_directory(
            str(recording_root),
            target.relative_to(recording_root).as_posix(),
            as_attachment=False,
        )

    @app.get("/exports/calibration/")
    def calibration_index() -> Any:
        return _export_directory_page(
            calibration_root,
            calibration_root,
            route_prefix="/exports/calibration/",
            title="Calibration suggestions",
        )

    @app.get("/exports/calibration/<path:relative_path>")
    def calibration_export(relative_path: str) -> Any:
        target = _safe_export_target(calibration_root, relative_path)
        if target is None or not target.exists():
            return "Export not found", 404
        if target.is_dir():
            return _export_directory_page(
                calibration_root,
                target,
                route_prefix="/exports/calibration/",
                title="Calibration suggestions",
            )
        return send_from_directory(
            str(calibration_root),
            target.relative_to(calibration_root).as_posix(),
            as_attachment=False,
        )

    @app.get("/api/tuning")
    def tuning() -> Any:
        return jsonify(runtime.tuning())

    @app.post("/api/tuning")
    def update_tuning() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(runtime.update_tuning(payload))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.get("/api/imu/ports")
    def imu_ports() -> Any:
        try:
            return jsonify({"ports": runtime.list_imu_ports()})
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "ports": []}), 503

    @app.post("/api/imu/connect")
    def imu_connect() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(
                runtime.connect_imu(
                    str(payload.get("port", "")),
                    baud_rate=int(payload.get("baud_rate", 921600)),
                )
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/imu/disconnect")
    def imu_disconnect() -> Any:
        try:
            return jsonify(runtime.disconnect_imu())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/imu-filter/reload")
    def imu_filter_reload() -> Any:
        try:
            return jsonify(runtime.reload_imu_filter())
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.get("/api/acquisition/history")
    def acquisition_history() -> Any:
        try:
            seconds = float(request.args.get("seconds", 15.0))
            max_points = int(request.args.get("max_points", 300))
            return jsonify({"rows": runtime.acquisition_history(seconds=seconds, max_points=max_points)})
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc), "rows": []}), 400

    @app.post("/api/camera/profile")
    def camera_profile() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(runtime.update_camera_profile(str(payload.get("profile", ""))))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/reference/camera-calibration/start")
    def camera_calibration_start() -> Any:
        try:
            return jsonify(runtime.start_camera_calibration())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/reference/camera-calibration/cancel")
    def camera_calibration_cancel() -> Any:
        try:
            return jsonify(runtime.cancel_camera_calibration())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/reference/known-angle-validation/start")
    def reference_validation_start() -> Any:
        try:
            return jsonify(runtime.start_reference_validation())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/reference/known-angle-validation/capture")
    def reference_validation_capture() -> Any:
        try:
            return jsonify(runtime.capture_reference_validation())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/reference/known-angle-validation/cancel")
    def reference_validation_cancel() -> Any:
        try:
            return jsonify(runtime.cancel_reference_validation())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/recording/start")
    def start_recording() -> Any:
        try:
            return jsonify(runtime.start_recording())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/recording/stop")
    def stop_recording() -> Any:
        try:
            return jsonify(runtime.stop_recording())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/recordings/open")
    def open_recordings() -> Any:
        try:
            return jsonify(runtime.open_recordings_location())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.post("/api/calibration/start")
    def calibration_start() -> Any:
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(runtime.start_calibration_test(str(payload.get("test_type", ""))))
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/calibration/stop")
    def calibration_stop() -> Any:
        try:
            return jsonify(runtime.stop_calibration_test())
        except (RuntimeError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/calibration/cancel")
    def calibration_cancel() -> Any:
        try:
            return jsonify(runtime.cancel_calibration_test())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/calibration/open")
    def calibration_open() -> Any:
        try:
            return jsonify(runtime.open_calibration_location())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.get("/video.mjpg")
    def video() -> Any:
        response = Response(
            stream_with_context(_mjpeg_stream(runtime)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


def _safe_export_target(root: Path, relative_path: str) -> Path | None:
    candidate = (root / str(relative_path or ".")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _export_directory_page(
    root: Path,
    directory: Path,
    *,
    route_prefix: str,
    title: str,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    relative_directory = directory.relative_to(root)
    rows: list[str] = []
    if directory != root:
        parent_relative = relative_directory.parent.as_posix()
        parent_url = route_prefix if parent_relative == "." else (
            route_prefix + quote(parent_relative) + "/"
        )
        rows.append(f'<li><a href="{escape(parent_url)}">../</a></li>')
    for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        relative = child.relative_to(root).as_posix()
        href = route_prefix + quote(relative)
        label = child.name + ("/" if child.is_dir() else "")
        if child.is_dir():
            href += "/"
            size_text = "folder"
        else:
            size_text = _format_file_size(child.stat().st_size)
        rows.append(
            '<li><a href="{href}">{label}</a><span>{size}</span></li>'.format(
                href=escape(href),
                label=escape(label),
                size=escape(size_text),
            )
        )
    if not rows:
        rows.append("<li><span>No exports yet.</span></li>")
    location = root.name
    if relative_directory.as_posix() != ".":
        location += "/" + relative_directory.as_posix()
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{margin:0;padding:24px;background:#0b0f16;color:#eef4ff;font-family:Segoe UI,sans-serif}}
main{{max-width:920px;margin:auto}} h1{{font-size:22px}} .path{{color:#91a4bf;margin-bottom:18px;overflow-wrap:anywhere}}
ul{{list-style:none;padding:0;border:1px solid #26354b;border-radius:12px;overflow:hidden}}
li{{display:flex;justify-content:space-between;gap:16px;padding:11px 14px;background:#111925;border-bottom:1px solid #26354b}}
li:last-child{{border-bottom:0}} a{{color:#a9d6ff;text-decoration:none;overflow-wrap:anywhere}} a:hover{{text-decoration:underline}}
span{{color:#91a4bf;font-size:12px;white-space:nowrap}}
</style></head><body><main><h1>{title}</h1><div class="path">{location}</div><ul>{rows}</ul></main></body></html>""".format(
        title=escape(title),
        location=escape(location),
        rows="".join(rows),
    )


def _format_file_size(size_bytes: int) -> str:
    size = float(max(0, int(size_bytes)))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def _mjpeg_stream(runtime: VisionRuntime) -> Iterable[bytes]:
    sequence = -1
    while True:
        sequence, jpeg = runtime.wait_for_jpeg(sequence, timeout_s=1.0)
        if jpeg is None:
            state = runtime.status()["state"]
            if state in {"FAILED", "STOPPED"}:
                break
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
