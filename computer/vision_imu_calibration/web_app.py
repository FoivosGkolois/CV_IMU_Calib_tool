from __future__ import annotations

from typing import Any, Iterable

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
html { min-width:320px; }
body { margin:0; background:var(--bg); color:var(--text); overflow-x:hidden; }
main { display:grid; grid-template-columns:minmax(360px,max-content) minmax(360px,1fr) minmax(300px,340px); gap:12px; padding:12px; width:100%; max-width:none; margin:0; align-items:start; }
.camera-column { min-width:0; display:flex; justify-content:flex-start; align-items:flex-start; }
.live-column { min-width:0; display:grid; grid-template-rows:auto minmax(280px,1fr); gap:10px; }
.video-shell { background:#05070b; border:1px solid var(--line); border-radius:14px; overflow:hidden; width:min(100%,960px); aspect-ratio:16 / 9; justify-self:start; align-self:start; display:flex; align-items:center; justify-content:center; transition:width .12s ease,height .12s ease; }
.video-shell img { width:100%; height:100%; object-fit:contain; display:block; }
.live-summary { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
.summary-card, .card, .chart-card { background:var(--surface); border:1px solid var(--line); border-radius:12px; }
.summary-card { padding:10px 12px; min-width:0; }
.summary-card .value { font-size:clamp(16px,1.4vw,21px); }
.control-panel { min-width:0; display:grid; gap:10px; max-height:calc(100vh - 112px); overflow-y:auto; overflow-x:hidden; padding-right:3px; scrollbar-gutter:stable; position:sticky; top:12px; }
.card { padding:12px; min-width:0; }
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
.path-link { height:auto; min-height:34px; margin-top:8px; padding:7px 9px; text-align:left; white-space:normal; line-height:1.3; font-weight:600; color:#a9d6ff; background:#0b1726; border-style:dashed; overflow-wrap:anywhere; }
.path-link:hover:not(:disabled) { border-color:var(--blue); background:#10243b; }
.metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; margin-top:8px; }
.metric { background:var(--surface-2); border:1px solid #2a3952; border-radius:9px; padding:8px; min-width:0; }
.metric .name { color:var(--muted); font-size:9px; text-transform:uppercase; letter-spacing:.07em; }
.metric .number { margin-top:4px; font-size:16px; font-weight:700; overflow-wrap:anywhere; }
.chart-card { padding:12px; }
.chart-shell { height:clamp(280px,calc(100vh - 245px),650px); margin-top:8px; }
.chart-shell canvas { width:100%; height:100%; display:block; background:#0a101a; border-radius:9px; }
footer { max-width:none; margin:0; padding:0 12px 12px; color:#73849c; font-size:10px; }
@media (max-width:1280px) {
  main { grid-template-columns:minmax(340px,max-content) minmax(340px,1fr); }
  .control-panel { grid-column:1 / -1; position:static; max-height:none; overflow:visible; grid-template-columns:repeat(2,minmax(0,1fr)); }
  .control-panel > .wide { grid-column:1 / -1; }
}
@media (max-width:860px) {
  main { grid-template-columns:1fr; padding:10px; gap:10px; }
  .camera-column { justify-content:flex-start; }
  .live-column { grid-template-rows:auto auto; }
  .control-panel { grid-template-columns:1fr; }
  .live-summary { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .chart-shell { height:260px; }
  .button-grid { grid-template-columns:1fr 1fr; }
}
@media (max-width:440px) {
  .live-summary { grid-template-columns:1fr; }
  .button-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<main>
<section class="camera-column">
  <div id="videoShell" class="video-shell"><img id="videoStream" src="/video.mjpg" alt="Live camera stream"></div>
</section>
<section class="live-column">
  <div class="live-summary">
    <div class="summary-card"><div class="label">Tracking</div><div id="tracking" class="value">Starting…</div><div id="reason" class="small">Initializing camera and pose model</div></div>
    <div class="summary-card"><div class="label">Visual angle</div><div id="angle" class="value">--.-°</div><div id="rawAngle" class="small">Raw: --.-°</div></div>
    <div class="summary-card"><div class="label">IMU angle</div><div id="imuAngle" class="value">--.-°</div><div id="imuEstimatorState" class="small">NO SAMPLE</div></div>
    <div class="summary-card"><div class="label">Pose quality</div><div id="confidence" class="value">0.00</div><div id="fps" class="small">0.0 FPS</div></div>
  </div>
  <div class="chart-card">
    <div class="label">Live posterior knee-angle presentation · last 15 s</div>
    <div class="note">Blue: visual reference · Green: IMU estimate. Presentation only; no scoring or recommendation.</div>
    <div class="chart-shell"><canvas id="angleChart"></canvas></div>
  </div>
</section>
<aside class="control-panel">
  <div class="card priority wide">
    <div class="label">Recording</div>
    <div id="recording" class="value small">IDLE</div>
    <div class="button-grid">
      <button id="startRecording" class="record" onclick="recordStart()">Start recording</button>
      <button id="stopRecording" class="stop" onclick="recordStop()">Stop</button>
    </div>
    <button id="recordingPath" class="path-link" onclick="openRecordingsLocation()" disabled>Trials are finalized as ZIP files inside the recordings folder.</button>
  </div>

  <div class="card wide">
    <div class="label">UNO R4 raw IMU stream</div>
    <div id="imuState" class="value small">DISCONNECTED</div>
    <div class="inline-grid"><select id="imuPort"><option value="">Select COM port</option></select><button class="secondary" onclick="refreshImuPorts()">Refresh</button></div>
    <div class="button-grid"><button id="imuConnect" class="primary" onclick="connectImu()">Connect</button><button id="imuDisconnect" class="stop" onclick="disconnectImu()">Disconnect</button></div>
    <div id="imuMessage" class="note">Continuous 100 Hz acquisition; recording decides whether samples enter the final ZIP.</div>
    <div id="imuRate" class="note">Rate: 0.0 Hz · Port: —</div>
    <div id="imuSensors" class="note">Thigh: NOT READY · Shank: NOT READY</div>
    <div id="imuHealth" class="note">Samples: 0 · Gaps: 0 · Parse errors: 0 · Queue drops: 0</div>
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

  <details class="card wide">
    <summary><span class="label">Pose-reference tuning</span></summary>
    <div class="controls">
      <div class="control-row">
        <div class="control-head"><span>Tracked leg</span></div>
        <select id="legSide"><option value="auto">Auto</option><option value="left">Left</option><option value="right">Right</option></select>
        <div class="control-help"><b>Auto:</b> highest-confidence leg. <b>Left/Right:</b> locks the corrected user-facing side.</div>
      </div>
      <div class="control-row">
        <div class="control-head"><span>Frame rotation</span></div>
        <select id="frameRotation"><option value="counterclockwise_90" selected>90° counter-clockwise</option><option value="none">0°</option><option value="clockwise_90">90° clockwise</option><option value="rotate_180">180°</option></select>
        <div class="control-help">Default for the current portrait camera mounting.</div>
      </div>
      <div class="control-row"><div class="control-head"><span>Acceptance confidence</span><span id="confidenceThresholdValue">0.30</span></div><input id="confidenceThreshold" type="range" min="0.05" max="0.99" step="0.01"><div class="control-help">Higher values reject uncertain frames more aggressively.</div></div>
      <div class="control-row"><div class="control-head"><span>Landmark smoothing</span><span id="landmarkSmoothingValue">0.55</span></div><input id="landmarkSmoothing" type="range" min="0" max="0.95" step="0.01"><div class="control-help">More smoothing reduces jitter but adds lag.</div></div>
      <div class="control-row"><div class="control-head"><span>Angle smoothing</span><span id="angleSmoothingValue">0.35</span></div><input id="angleSmoothing" type="range" min="0" max="0.95" step="0.01"><div class="control-help">More smoothing stabilizes the angle but slows response.</div></div>
      <div class="control-row"><div class="control-head"><span>Minimum visible segment</span><span id="segmentLengthValue">0.025</span></div><input id="segmentLength" type="range" min="0.005" max="0.20" step="0.005"><div class="control-help">Rejects implausibly short thigh/shank projections.</div></div>
      <div class="control-row"><div class="control-head"><span>Maximum angle jump</span><span id="angleJumpValue">45°</span></div><input id="angleJump" type="range" min="5" max="180" step="1"><div class="control-help">Lower values reject spikes more aggressively.</div></div>
    </div>
    <button class="primary" style="margin-top:10px" onclick="applyTuning()">Apply tuning</button>
    <div id="tuningMessage" class="note">Application-level filtering only; model weights are unchanged.</div>
  </details>

  <details class="card wide">
    <summary><span class="label">Runtime diagnostics</span></summary>
    <div id="runtime" class="value small">STARTING</div>
    <div id="backend" class="note">Backend: MoveNet Thunder</div>
    <div id="source" class="note">Source: —</div>
    <div id="frameCount" class="note">Frames: 0</div>
  </details>
</aside>
</main>
<footer>Acquisition/presentation stage only: live vision, raw dual-IMU, current filter output, synchronized evidence. Parameter scoring and recommendation are intentionally outside this stage.</footer>
<script>
const ids = {
  confidence_threshold:'confidenceThreshold', landmark_smoothing:'landmarkSmoothing',
  angle_smoothing:'angleSmoothing', min_segment_length:'segmentLength',
  max_angle_jump_deg:'angleJump', leg_side:'legSide', frame_rotation:'frameRotation'
};
const confidenceThreshold=document.getElementById('confidenceThreshold');
const landmarkSmoothing=document.getElementById('landmarkSmoothing');
const angleSmoothing=document.getElementById('angleSmoothing');
const segmentLength=document.getElementById('segmentLength');
const angleJump=document.getElementById('angleJump');
const legSide=document.getElementById('legSide');
const frameRotation=document.getElementById('frameRotation');
const confidenceThresholdValue=document.getElementById('confidenceThresholdValue');
const landmarkSmoothingValue=document.getElementById('landmarkSmoothingValue');
const angleSmoothingValue=document.getElementById('angleSmoothingValue');
const segmentLengthValue=document.getElementById('segmentLengthValue');
const angleJumpValue=document.getElementById('angleJumpValue');
const tuningMessage=document.getElementById('tuningMessage');
const reason=document.getElementById('reason');
const angle=document.getElementById('angle');
const rawAngle=document.getElementById('rawAngle');
const confidence=document.getElementById('confidence');
const fps=document.getElementById('fps');
const source=document.getElementById('source');
const backend=document.getElementById('backend');
const runtime=document.getElementById('runtime');
const frameCount=document.getElementById('frameCount');
const recording=document.getElementById('recording');
const startRecording=document.getElementById('startRecording');
const stopRecording=document.getElementById('stopRecording');
const recordingPath=document.getElementById('recordingPath');
const imuState=document.getElementById('imuState');
const imuRate=document.getElementById('imuRate');
const imuSensors=document.getElementById('imuSensors');
const imuHealth=document.getElementById('imuHealth');
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
const reloadImuFilter=document.getElementById('reloadImuFilter');
const angleChart=document.getElementById('angleChart');
const videoShell=document.getElementById('videoShell');
const videoStream=document.getElementById('videoStream');
function resizeVideoShell(){
  const naturalWidth=Number(videoStream.naturalWidth||0);
  const naturalHeight=Number(videoStream.naturalHeight||0);
  if(!naturalWidth||!naturalHeight) return;
  const availableWidth=Math.max(1,videoShell.parentElement.clientWidth);
  const singleColumn=window.matchMedia('(max-width:860px)').matches;
  const reservedHeight=singleColumn?24:24;
  const availableHeight=Math.max(260,Math.min(900,window.innerHeight-reservedHeight));
  const aspect=naturalWidth/naturalHeight;
  const width=Math.min(availableWidth,availableHeight*aspect);
  const height=width/aspect;
  videoShell.style.width=Math.max(1,Math.round(width))+'px';
  videoShell.style.height=Math.max(1,Math.round(height))+'px';
  videoShell.style.aspectRatio=String(aspect);
}
videoStream.addEventListener('load',resizeVideoShell);
window.addEventListener('resize',resizeVideoShell,{passive:true});
function updateSliderLabels(){
  confidenceThresholdValue.textContent=Number(confidenceThreshold.value).toFixed(2);
  landmarkSmoothingValue.textContent=Number(landmarkSmoothing.value).toFixed(2);
  angleSmoothingValue.textContent=Number(angleSmoothing.value).toFixed(2);
  segmentLengthValue.textContent=Number(segmentLength.value).toFixed(3);
  angleJumpValue.textContent=Number(angleJump.value).toFixed(0)+'°';
}
for (const id of ['confidenceThreshold','landmarkSmoothing','angleSmoothing','segmentLength','angleJump']) {
  document.getElementById(id).addEventListener('input',updateSliderLabels);
}
function loadTuning(t){
  if(!t) return;
  for(const [key,id] of Object.entries(ids)) if(t[key]!==undefined) document.getElementById(id).value=t[key];
  updateSliderLabels();
}
async function applyTuning(){
  const payload={
    leg_side:legSide.value,
    frame_rotation:frameRotation.value,
    confidence_threshold:Number(confidenceThreshold.value),
    landmark_smoothing:Number(landmarkSmoothing.value),
    angle_smoothing:Number(angleSmoothing.value),
    min_segment_length:Number(segmentLength.value),
    max_angle_jump_deg:Number(angleJump.value)
  };
  const r=await fetch('/api/tuning',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const body=await r.json();
  tuningMessage.textContent=r.ok?'Tuning applied live.':('Tuning rejected: '+(body.error||'unknown error'));
  if(r.ok) loadTuning(body);
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
    imuMessage.textContent=(body.ports||[]).length?`${body.ports.length} serial port(s) found.`:'No serial ports found.';
  } catch(e) { imuMessage.textContent='Port scan failed: '+e; }
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
async function recordStart(){
  recordingPath.textContent='Starting trial…';
  const r=await fetch('/api/recording/start',{method:'POST'});
  const body=await r.json();
  recordingPath.textContent=r.ok?('ZIP destination: '+(body.archive_path||body.path||'recordings')):('Start failed: '+(body.error||'unknown error'));
}
async function recordStop(){
  recordingPath.textContent='Finalizing synchronized data and ZIP…';
  const r=await fetch('/api/recording/stop',{method:'POST'});
  const body=await r.json();
  recordingPath.textContent=r.ok?('Saved ZIP: '+(body.archive_path||body.path||'recordings')):('Stop failed: '+(body.error||'unknown error'));
}
async function openRecordingsLocation(){
  const r=await fetch('/api/recordings/open',{method:'POST'});
  const body=await r.json();
  if(!r.ok) recordingPath.textContent='Could not open recordings folder: '+(body.error||'unknown error');
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
async function refreshHistory(){ try{ const r=await fetch('/api/acquisition/history?seconds=15&max_points=300',{cache:'no-store'}); const body=await r.json(); if(r.ok) drawAngleChart(body.rows||[]); }catch(e){} }
async function refresh(){
  try {
    const r=await fetch('/api/status',{cache:'no-store'}); const s=await r.json();
    const t=document.getElementById('tracking');
    t.textContent=s.tracking_valid?'VALID':'INVALID'; t.className='value '+(s.tracking_valid?'ok':'bad');
    reason.textContent=s.tracking_reason||'—';
    angle.textContent=s.visual_knee_angle_deg==null?'--.-°':Number(s.visual_knee_angle_deg).toFixed(1)+'°';
    rawAngle.textContent=s.raw_knee_angle_deg==null?'Raw: --.-°':'Raw: '+Number(s.raw_knee_angle_deg).toFixed(1)+'°';
    confidence.textContent=Number(s.pose_confidence).toFixed(2);
    fps.textContent=Number(s.inference_fps).toFixed(1)+' FPS';
    source.textContent='Source: '+(s.source||'—');
    backend.textContent='Backend: '+(s.pose_backend||'—');
    runtime.textContent=s.error?`${s.state}: ${s.error}`:s.state;
    frameCount.textContent='Frames: '+Number(s.frame_index||0);
    const imu=s.imu||{};
    imuState.textContent=imu.configured?(imu.state||'UNKNOWN'):'DISCONNECTED';
    imuState.className='value small '+(imu.connected?'ok':(imu.configured?'bad':''));
    imuRate.textContent='Rate: '+Number(imu.sample_rate_hz||0).toFixed(1)+' Hz · Port: '+(imu.port||'—');
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
    reloadImuFilter.disabled=Boolean((s.recording||{}).active);
    imuConnect.disabled=Boolean(imu.configured)||Boolean((s.recording||{}).active);
    imuDisconnect.disabled=!Boolean(imu.configured)||Boolean((s.recording||{}).active);
    if(imu.port && [...imuPort.options].some(o=>o.value===imu.port)) imuPort.value=imu.port;
    const camera=s.camera||{};
    if(!window.cameraProfileLoaded && camera.profile){ cameraProfile.value=camera.profile; window.cameraProfileLoaded=true; }
    applyCameraProfile.disabled=Boolean((s.recording||{}).active)||camera.reconfigurable===false||!['CAMERA_PREVIEW','RUNNING','AI_FAILED'].includes(s.state);
    const rec=s.recording||{};
    recording.textContent=rec.active?`RECORDING · ${Number(rec.elapsed_s||0).toFixed(1)} s · ${rec.frame_count||0} camera / ${rec.imu_sample_count||0} IMU samples`:'IDLE';
    recording.className='value small '+(rec.active?'rec':'');
    startRecording.disabled=Boolean(rec.active); stopRecording.disabled=!rec.active;
    recordingPath.disabled=!Boolean(rec.output_root);
    if(rec.path) recordingPath.textContent=(rec.active?'ZIP destination: ':'Saved ZIP: ')+rec.path;
    else if(rec.output_root) recordingPath.textContent='Open recordings folder: '+rec.output_root;
    if(rec.error) recordingPath.textContent='Recorder error: '+rec.error;
    if(!window.tuningLoaded && s.tuning){ loadTuning(s.tuning); window.tuningLoaded=true; }
  } catch(e) { runtime.textContent='UI connection error: '+e; }
}
setInterval(refresh,350); setInterval(refreshHistory,700); setInterval(resizeVideoShell,1000); refresh(); refreshHistory(); refreshImuPorts(); resizeVideoShell();
</script>
</body>
</html>
'''


def create_app(runtime: VisionRuntime) -> Any:
    try:
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError as exc:
        raise RuntimeError("Flask is not installed. Run 08_SETUP_VISION_IMU_PC.bat") from exc

    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return _PAGE

    @app.get("/api/status")
    def status() -> Any:
        return jsonify(runtime.status())

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
