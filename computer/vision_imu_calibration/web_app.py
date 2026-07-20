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
:root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }
* { box-sizing:border-box; }
body { margin:0; background:#0b0f16; color:#eef4ff; }
header { padding:18px 28px; border-bottom:1px solid #253147; background:#101722; }
h1 { margin:0; font-size:22px; letter-spacing:.04em; }
.sub { margin-top:6px; color:#91a4bf; font-size:13px; }
main { display:grid; grid-template-columns:minmax(520px,920px) 380px; justify-content:center; gap:18px; padding:18px; max-width:1400px; margin:0 auto; }
.left-stack { display:flex; flex-direction:column; gap:14px; min-width:0; }
.video-shell { background:#05070b; border:1px solid #253147; border-radius:16px; overflow:hidden; height:min(70vh,720px); min-height:380px; display:flex; align-items:center; justify-content:center; }
.video-shell img { max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block; }
.panel { display:flex; flex-direction:column; gap:12px; max-height:min(70vh,720px); overflow-y:auto; padding-right:4px; }
.card { background:#121a27; border:1px solid #253147; border-radius:14px; padding:14px; }
.label { color:#91a4bf; font-size:11px; letter-spacing:.1em; text-transform:uppercase; }
.value { margin-top:6px; font-size:22px; font-weight:700; }
.ok { color:#51e26d; } .bad { color:#ff6464; } .rec { color:#ff6363; }
.small { font-size:13px; color:#d8e3f4; word-break:break-word; }
.controls { display:grid; gap:15px; margin-top:12px; }
.control-row { display:grid; gap:6px; }
.control-head { display:flex; justify-content:space-between; gap:12px; font-size:13px; color:#c9d5e7; }
.control-help { color:#8fa3bd; font-size:11px; line-height:1.4; }
.control-help b { color:#c9d5e7; font-weight:650; }
input[type=range] { width:100%; accent-color:#39a0ff; }
select, button { width:100%; border:1px solid #354764; border-radius:10px; background:#0d1420; color:#eef4ff; padding:9px 11px; font:inherit; }
button { cursor:pointer; font-weight:700; }
button.primary { background:#1767b2; border-color:#2789de; }
button.record { background:#8d2028; border-color:#d3414d; }
button.stop { background:#263348; }
button:disabled { opacity:.45; cursor:not-allowed; }
.button-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px; }
.inline-grid { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; margin-top:10px; }
.inline-grid button { width:auto; min-width:96px; }
.note { color:#91a4bf; font-size:11px; line-height:1.45; margin-top:9px; }
.chart-card { background:#121a27; border:1px solid #253147; border-radius:14px; padding:14px; }
.chart-shell { height:220px; margin-top:10px; }
.chart-shell canvas { width:100%; height:100%; display:block; background:#0a101a; border-radius:10px; }
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
.metric { background:#0d1420; border:1px solid #2a3952; border-radius:10px; padding:10px; }
.metric .name { color:#91a4bf; font-size:10px; text-transform:uppercase; letter-spacing:.08em; }
.metric .number { margin-top:5px; font-size:18px; font-weight:700; }
footer { max-width:1400px; margin:0 auto; padding:0 18px 18px; color:#73849c; font-size:11px; }
@media(max-width:1050px){ main{grid-template-columns:1fr;max-width:900px}.video-shell{height:min(62vh,620px);min-height:300px}.panel{max-height:none;overflow:visible} }
</style>
</head>
<body>
<header>
<h1>VISION–IMU ACQUISITION & PRESENTATION</h1>
<div class="sub">Live visual reference, raw dual-IMU acquisition, STM32-equivalent filter output, synchronized recording, and evidence presentation</div>
</header>
<main>
<section class="left-stack">
  <div class="video-shell"><img id="videoStream" src="/video.mjpg" alt="Live annotated camera stream"></div>
  <div class="chart-card">
    <div class="label">Live posterior knee-angle presentation · last 15 s</div>
    <div class="note">Blue: visual posterior-knee-angle reference · Green: current IMU posterior-knee-angle estimate. This is presentation only; no scoring or parameter recommendation is performed.</div>
    <div class="chart-shell"><canvas id="angleChart"></canvas></div>
  </div>
</section>
<aside class="panel">
<div class="card"><div class="label">Tracking</div><div id="tracking" class="value">Starting…</div><div id="reason" class="small">Initializing camera and pose model</div></div>
<div class="card"><div class="label">Visual posterior knee angle</div><div id="angle" class="value">--.-°</div><div id="rawAngle" class="small">Raw: --.-°</div></div>
<div class="card"><div class="label">Pose confidence</div><div id="confidence" class="value">0.00</div></div>
<div class="card">
  <div class="label">Current IMU posterior-knee-angle estimator</div>
  <div id="imuEstimatorState" class="value small">NO SAMPLE</div>
  <div class="metric-grid">
    <div class="metric"><div class="name">IMU angle</div><div id="imuAngle" class="number">--.-°</div></div>
    <div class="metric"><div class="name">Visual − IMU</div><div id="angleDifference" class="number">--.-°</div></div>
    <div class="metric"><div class="name">Knee speed</div><div id="imuSpeed" class="number">--.-°/s</div></div>
    <div class="metric"><div class="name">Thigh / Shank</div><div id="segmentAngles" class="number">--.- / --.-°</div></div>
  </div>
  <button id="reloadImuFilter" class="primary" style="margin-top:10px" onclick="reloadFilterConfig()">Reload IMU filter config</button>
  <div id="imuFilterInfo" class="note">Waiting for synchronized raw IMU samples.</div>
</div>
<div class="card"><div class="label">Inference rate</div><div id="fps" class="value">0.0 FPS</div></div>
<div class="card">
  <div class="label">UNO R4 raw IMU stream</div>
  <div id="imuState" class="value small">DISCONNECTED</div>
  <div class="inline-grid"><select id="imuPort"><option value="">Select COM port</option></select><button onclick="refreshImuPorts()">Refresh</button></div>
  <div class="button-grid"><button id="imuConnect" class="primary" onclick="connectImu()">Connect</button><button id="imuDisconnect" class="stop" onclick="disconnectImu()">Disconnect</button></div>
  <div id="imuMessage" class="note">Connect starts the continuous 100 Hz calibration stream. Recording only decides whether samples are written to a trial folder.</div>
  <div id="imuRate" class="note">Rate: 0.0 Hz · Port: —</div>
  <div id="imuSensors" class="note">Thigh: NOT READY · Shank: NOT READY</div>
  <div id="imuHealth" class="note">Samples: 0 · Gaps: 0 · Parse errors: 0 · Queue drops: 0</div>
</div>
<div class="card">
  <div class="label">Camera quality profile</div>
  <select id="cameraProfile" style="margin-top:10px"><option value="fast">Fast · 640×480</option><option value="quality">Quality · 1280×720</option></select>
  <button id="applyCameraProfile" class="primary" style="margin-top:10px" onclick="setCameraProfile()">Apply camera profile</button>
  <div id="cameraProfileMessage" class="note"><b>Fast:</b> quicker startup and lower load. <b>Quality:</b> more image detail and potentially better tracking at difficult knee angles, with slower startup.</div>
</div>
<div class="card">
  <div class="label">Annotated video recorder</div>
  <div id="recording" class="value small">IDLE</div>
  <div class="button-grid">
    <button id="startRecording" class="record" onclick="recordStart()">Start recording</button>
    <button id="stopRecording" class="stop" onclick="recordStop()">Stop</button>
  </div>
  <div id="recordingPath" class="note">Each trial stores video, pose timeline, raw IMU, processed IMU estimator timeline, synchronized dataset, acquisition summary, and overview plot.</div>
</div>
<div class="card">
  <div class="label">Live pose-reference tuning</div>
  <div class="controls">
    <div class="control-row">
      <div class="control-head"><span>Tracked leg</span></div>
      <select id="legSide"><option value="auto">Auto</option><option value="left">Left</option><option value="right">Right</option></select>
      <div class="control-help"><b>Auto:</b> διαλέγει το πιο βέβαιο πόδι αλλά μπορεί να αλλάξει πλευρά. <b>Left/Right:</b> κλειδώνει το αναγνωρισμένο κοντινό στην κάμερα πόδι με τη διορθωμένη convention αριστερού/δεξιού.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Frame rotation</span></div>
      <select id="frameRotation"><option value="none">0°</option><option value="clockwise_90">90° clockwise</option><option value="counterclockwise_90">90° counter-clockwise</option><option value="rotate_180">180°</option></select>
      <div class="control-help">Για κάμερα σε portrait θέση. Διάλεξε την περιστροφή που εμφανίζει το σώμα όρθιο.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Acceptance confidence</span><span id="confidenceThresholdValue">0.30</span></div>
      <input id="confidenceThreshold" type="range" min="0.05" max="0.99" step="0.01">
      <div class="control-help"><b>Αριστερά:</b> περισσότερα valid frames αλλά περισσότερες λανθασμένες θέσεις. <b>Δεξιά:</b> αυστηρότερη αποδοχή και περισσότερα INVALID gaps.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Landmark smoothing</span><span id="landmarkSmoothingValue">0.55</span></div>
      <input id="landmarkSmoothing" type="range" min="0" max="0.95" step="0.01">
      <div class="control-help"><b>Αριστερά:</b> οι τελείες και μπάρες αντιδρούν γρήγορα αλλά τρεμοπαίζουν. <b>Δεξιά:</b> σταθεροποιούνται περισσότερο αλλά αποκτούν καθυστέρηση.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Angle smoothing</span><span id="angleSmoothingValue">0.35</span></div>
      <input id="angleSmoothing" type="range" min="0" max="0.95" step="0.01">
      <div class="control-help"><b>Αριστερά:</b> γρήγορη αλλά πιο θορυβώδης γωνία. <b>Δεξιά:</b> ομαλότερη ένδειξη με μεγαλύτερο lag.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Minimum visible segment</span><span id="segmentLengthValue">0.025</span></div>
      <input id="segmentLength" type="range" min="0.005" max="0.20" step="0.005">
      <div class="control-help"><b>Αριστερά:</b> δέχεται μικρότερα ή μερικώς κομμένα τμήματα. <b>Δεξιά:</b> απορρίπτει πολύ μικρό μηρό/κνήμη και μειώνει γεωμετρικά λάθη.</div>
    </div>
    <div class="control-row">
      <div class="control-head"><span>Maximum accepted angle jump</span><span id="angleJumpValue">45°</span></div>
      <input id="angleJump" type="range" min="5" max="180" step="1">
      <div class="control-help"><b>Αριστερά:</b> απορρίπτει επιθετικά απότομα outliers αλλά μπορεί να κόψει πραγματική γρήγορη κίνηση. <b>Δεξιά:</b> επιτρέπει γρήγορες μεταβολές αλλά περνά περισσότερα spikes.</div>
    </div>
  </div>
  <button class="primary" style="margin-top:14px" onclick="applyTuning()">Apply tuning</button>
  <div id="tuningMessage" class="note">Οι ρυθμίσεις εφαρμόζονται live γύρω από το pretrained model· δεν αλλάζουν τα weights του AI.</div>
</div>
<div class="card"><div class="label">Runtime</div><div id="runtime" class="value small">STARTING</div><div id="backend" class="note">Backend: MoveNet Thunder</div><div id="source" class="note">Source: —</div><div id="frameCount" class="note">Frames: 0</div></div>
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
  finally { applyCameraProfile.disabled=false; }
}
async function recordStart(){
  const r=await fetch('/api/recording/start',{method:'POST'});
  const body=await r.json();
  if(!r.ok) recordingPath.textContent='Start failed: '+(body.error||'unknown error');
}
async function recordStop(){
  const r=await fetch('/api/recording/stop',{method:'POST'});
  const body=await r.json();
  if(!r.ok) recordingPath.textContent='Stop failed: '+(body.error||'unknown error');
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
    imuEstimatorState.className='value small '+(est.valid?'ok':'bad');
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
    if(rec.path) recordingPath.textContent='Saved to: '+rec.path;
    if(rec.error) recordingPath.textContent='Recorder error: '+rec.error;
    if(!window.tuningLoaded && s.tuning){ loadTuning(s.tuning); window.tuningLoaded=true; }
  } catch(e) { runtime.textContent='UI connection error: '+e; }
}
setInterval(refresh,350); setInterval(refreshHistory,700); refresh(); refreshHistory(); refreshImuPorts();
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
