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
.note { color:#91a4bf; font-size:11px; line-height:1.45; margin-top:9px; }
footer { max-width:1400px; margin:0 auto; padding:0 18px 18px; color:#73849c; font-size:11px; }
@media(max-width:1050px){ main{grid-template-columns:1fr;max-width:900px}.video-shell{height:min(62vh,620px);min-height:300px}.panel{max-height:none;overflow:visible} }
</style>
</head>
<body>
<header>
<h1>VISION–IMU CALIBRATION · PC FEASIBILITY PROTOTYPE</h1>
<div class="sub">MoveNet Thunder side-view pose reference, stabilized landmarks, portrait support, and evidence recording</div>
</header>
<main>
<section class="video-shell"><img id="videoStream" src="/video.mjpg" alt="Live annotated camera stream"></section>
<aside class="panel">
<div class="card"><div class="label">Tracking</div><div id="tracking" class="value">Starting…</div><div id="reason" class="small">Initializing camera and pose model</div></div>
<div class="card"><div class="label">Visual knee flexion</div><div id="angle" class="value">--.-°</div><div id="rawAngle" class="small">Raw: --.-°</div></div>
<div class="card"><div class="label">Pose confidence</div><div id="confidence" class="value">0.00</div></div>
<div class="card"><div class="label">Inference rate</div><div id="fps" class="value">0.0 FPS</div></div>
<div class="card"><div class="label">UNO R4 raw IMU stream</div><div id="imuState" class="value small">DISABLED</div><div id="imuRate" class="note">Rate: 0.0 Hz</div><div id="imuSensors" class="note">Thigh: — · Shank: —</div><div id="imuHealth" class="note">Samples: 0 · Gaps: 0 · Parse errors: 0</div></div>
<div class="card">
  <div class="label">Annotated video recorder</div>
  <div id="recording" class="value small">IDLE</div>
  <div class="button-grid">
    <button id="startRecording" class="record" onclick="recordStart()">Start recording</button>
    <button id="stopRecording" class="stop" onclick="recordStop()">Stop</button>
  </div>
  <div id="recordingPath" class="note">Video timing follows the monotonic CSV timeline.</div>
</div>
<div class="card">
  <div class="label">Live pose-reference tuning</div>
  <div class="controls">
    <div class="control-row">
      <div class="control-head"><span>Tracked leg</span></div>
      <select id="legSide"><option value="auto">Auto</option><option value="left">Left</option><option value="right">Right</option></select>
      <div class="control-help"><b>Auto:</b> διαλέγει το πιο βέβαιο πόδι αλλά μπορεί να αλλάξει πλευρά. <b>Left/Right:</b> κλειδώνει το κοντινό στην κάμερα πόδι.</div>
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
<footer>Standalone calibration path: USB webcam + UNO R4 raw dual-IMU stream. Production telemetry, motor, and safety remain unchanged.</footer>
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
    imuState.textContent=imu.configured?(imu.state||'UNKNOWN'):'DISABLED';
    imuState.className='value small '+(imu.connected?'ok':(imu.configured?'bad':''));
    imuRate.textContent='Rate: '+Number(imu.sample_rate_hz||0).toFixed(1)+' Hz · Port: '+(imu.port||'—');
    imuSensors.textContent='Thigh: '+(imu.thigh_ready?'READY':'NOT READY')+' · Shank: '+(imu.shank_ready?'READY':'NOT READY');
    imuHealth.textContent='Samples: '+Number(imu.sample_count||0)+' · Gaps: '+Number(imu.sequence_gaps||0)+' · Parse errors: '+Number(imu.parse_errors||0)+' · Queue drops: '+Number(imu.queue_drops||0);
    if(imu.last_error) imuHealth.textContent+=' · Error: '+imu.last_error;
    const rec=s.recording||{};
    recording.textContent=rec.active?`RECORDING · ${Number(rec.elapsed_s||0).toFixed(1)} s · ${rec.frame_count||0} camera / ${rec.imu_sample_count||0} IMU samples`:'IDLE';
    recording.className='value small '+(rec.active?'rec':'');
    startRecording.disabled=Boolean(rec.active); stopRecording.disabled=!rec.active;
    if(rec.path) recordingPath.textContent='Saved to: '+rec.path;
    if(rec.error) recordingPath.textContent='Recorder error: '+rec.error;
    if(!window.tuningLoaded && s.tuning){ loadTuning(s.tuning); window.tuningLoaded=true; }
  } catch(e) { runtime.textContent='UI connection error: '+e; }
}
setInterval(refresh,350); refresh();
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
