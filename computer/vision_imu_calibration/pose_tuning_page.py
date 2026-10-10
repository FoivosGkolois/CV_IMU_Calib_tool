"""Standalone guided browser view using the SAME runtime and camera pipeline as the main app.

No model retraining, background optimizer, automatic safety-threshold selection or
hidden application of settings. Measurements are descriptive and operator-reviewed.
"""

TUNING_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UNO Q — Pose Tuning Laboratory</title>
<style>
:root{font-family:Inter,"Segoe UI",sans-serif;color-scheme:dark;background:#0b0f16;color:#eef4ff}
*{box-sizing:border-box}body{margin:0;padding:16px}.layout{display:grid;grid-template-columns:minmax(300px,53fr) minmax(330px,47fr);gap:14px;max-width:1600px;margin:auto}.panel{border:1px solid #2d405a;background:#111925;border-radius:12px;padding:16px}.video{width:100%;max-height:79vh;object-fit:contain;background:#080d13;border-radius:8px}h1{margin:0 0 12px;font-size:19px}h2{font-size:15px;margin:12px 0}p,small{color:#b0bfd3;line-height:1.4}label{display:block;font-size:13px;margin:12px 0 6px}button,select,input{background:#17263a;color:#fff;border:1px solid #345171;border-radius:8px;padding:9px;font:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:not-allowed}button.primary{background:#1867b7}.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1}input[type=number]{width:100%}.note{padding:10px;border:1px solid #31445e;border-radius:8px;margin:10px 0;font-size:13px}.metric{font-variant-numeric:tabular-nums}.ok{color:#51e26d}.bad{color:#ff7474}table{width:100%;border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #30415c;padding:6px;text-align:left}details{margin:10px 0}summary{cursor:pointer;color:#a9d6ff}a{color:#a9d6ff}@media(max-width:930px){.layout{grid-template-columns:1fr}.video{max-height:60vh}}
</style></head><body><div class="layout"><div class="panel"><h1>UNO Q · Pose tuning laboratory</h1>
<p>Independent guided tool · same MoveNet/camera runtime · no changes to the motor or IMU filter.</p>
<img class="video" src="/video.mjpg" alt="Live pose overlay"><div id="live" class="note">Waiting for live camera…</div>
<small>Keep camera, subject, leg selection and lighting consistent across compared trials. Confidence scores do not establish geometric accuracy.</small>
</div><div class="panel"><h2 id="step">Step 1 / 5</h2><p id="help"></p>
<label for="candidate">Candidate setting</label><select id="candidate"></select>
<label for="custom">Or enter a custom value (optional)</label><input id="custom" type="number" step="any" placeholder="Use selected candidate">
<div class="row" style="margin-top:10px"><button class="primary" id="apply">Apply candidate temporarily</button><button id="measure" disabled>Record 3 s trial</button></div>
<div class="note" id="message">Choose a candidate, apply it and record a controlled trial. Measure multiple candidates under the same conditions.</div>
<h2>Trial evidence</h2><table><thead><tr><th>Setting</th><th>Value</th><th>Valid</th><th>Raw SD</th><th>Filtered SD</th><th>FPS</th></tr></thead><tbody id="results"></tbody></table>
<p><small>Valid = accepted camera frames; SD = variation in degrees during the trial. A moving leg naturally increases SD. Compare like with like; do not maximize a single number.</small></p>
<div class="row"><button id="back">Previous parameter</button><button id="next">Next parameter</button></div>
<h2>Finalize</h2>
<p>Review each parameter and choose a value. No setting is committed automatically.</p>
<div id="selections" class="note"></div>
<div class="row"><button id="export">Download test evidence JSON</button><button class="primary" id="save">Save selected values as default</button></div>
<div id="saved" class="note">No permanent changes saved.</div>
<details><summary>Usage / measurement limitations</summary><p>Acceptance confidence: test missed landmarks against false positives across poses. Smoothing: compare stationary noise against response delay on matched slow bends. Minimum segment: reject implausibly short segments without discarding real flexion. Max angle jump: avoid spikes without suppressing rapid legitimate motion. Changes reset calibration setup, so finish this tuning before IMU calibration. All trial comparisons are descriptive, not an automated optimizer or an accuracy certification.</p></details>
<a href="/">Open main Vision–IMU dashboard</a>
</div></div><script>
const parameters=[
  {key:'confidence_threshold',name:'Acceptance confidence',hint:'Hold still, then move slowly. Higher thresholds reject weak detections but can increase missed frames.',choices:[0.20,0.30,0.40],min:0.05,max:0.99},
  {key:'landmark_smoothing',name:'Landmark smoothing',hint:'Compare stillness and slow bend/return. More smoothing may lag real movement.',choices:[0.25,0.55,0.75],min:0,max:0.95},
  {key:'angle_smoothing',name:'Angle smoothing',hint:'Compare stationary jitter and delayed angle response on matched slow bends.',choices:[0.15,0.35,0.55],min:0,max:0.95},
  {key:'min_segment_length',name:'Minimum visible segment',hint:'Stand near image edges and at different knee angles; avoid discarding short but legitimate limb projections.',choices:[0.015,0.025,0.04],min:0.005,max:0.2},
  {key:'max_angle_jump_deg',name:'Maximum angle jump',hint:'Perform comparable bends; low limits may freeze valid fast changes, high limits may allow spikes.',choices:[25,45,80],min:5,max:180},
];
const byId=id=>document.getElementById(id);
let step=0,activeValue=null,selected={},evidence=[],baseline=null,busy=false;
const format=v=>Number(v).toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
function render(){
 const p=parameters[step]; byId('step').textContent=`Step ${step+1} / 5 · ${p.name}`;byId('help').textContent=p.hint;
 byId('candidate').replaceChildren();for(const v of p.choices){const opt=document.createElement('option');opt.value=v;opt.textContent=format(v);byId('candidate').appendChild(opt);}
 byId('candidate').value=String(p.choices.includes(selected[p.key])?selected[p.key]:p.choices[1]);byId('custom').value=(selected[p.key]!==undefined&&!p.choices.includes(selected[p.key]))?String(selected[p.key]):'';
 activeValue=null;byId('measure').disabled=true;
 byId('back').disabled=step===0;byId('next').disabled=step===parameters.length-1;
 byId('results').replaceChildren();
 for(const e of evidence.filter(e=>e.parameter===p.key)){
   const tr=document.createElement('tr');for(const k of ['parameter','value','valid_fraction','raw_sd','filtered_sd','fps']){
    const td=document.createElement('td');td.textContent=k==='parameter'?p.name:k==='valid_fraction'?`${(e[k]*100).toFixed(0)}%`:Number(e[k]).toFixed(k==='value'?3:2);tr.appendChild(td);
   }byId('results').appendChild(tr);
 }
 byId('selections').textContent=parameters.map(item=>item.name+': '+(selected[item.key]===undefined?'not chosen':format(selected[item.key]))).join(' · ');
}
function message(s){byId('message').textContent=s;}
async function request(url,options){const r=await fetch(url,options);const o=await r.json();if(!r.ok)throw Error(o.error||`HTTP ${r.status}`);return o;}
function candidate(){const p=parameters[step];const custom=byId('custom').value.trim();const v=custom?Number(custom):Number(byId('candidate').value);if(!Number.isFinite(v)||v<p.min||v>p.max)throw Error(`${p.name}: value must be ${p.min}–${p.max}`);return v;}
byId('apply').onclick=async()=>{if(busy)return;busy=true;try{
 const p=parameters[step],v=candidate();await request('/api/tuning',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[p.key]:v})});
 selected[p.key]=v;activeValue=v;byId('measure').disabled=false;message(`${p.name} = ${format(v)} applied for this session. Keep the setup steady, then record a 3-second trial.`);renderSelections();
}catch(e){message('Could not apply: '+e.message);}finally{busy=false;}};
function renderSelections(){byId('selections').textContent=parameters.map(p=>p.name+': '+(selected[p.key]===undefined?'not chosen':format(selected[p.key]))).join(' · ');}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const std=arr=>{if(arr.length<2)return NaN;const avg=arr.reduce((a,b)=>a+b,0)/arr.length;return Math.sqrt(arr.reduce((a,b)=>a+(b-avg)**2,0)/(arr.length-1));};
byId('measure').onclick=async()=>{if(busy||activeValue===null)return;busy=true;byId('measure').disabled=true;
 try{
   message('Preparing…');await delay(650);
   const start=await request('/api/pose-tuning/history?seconds=2');
   const initial=Math.max(0,...start.rows.map(x=>x.frame_index));
   message('Recording 3 seconds. Keep the same posture/movement protocol for every candidate.');await delay(3200);
   const after=await request('/api/pose-tuning/history?seconds=5');
   const rows=after.rows.filter(r=>r.frame_index>initial);if(rows.length<4)throw Error('Too few frames collected; retry.');
   const valid=rows.filter(r=>r.tracking_valid===true);
   const raw=valid.map(r=>r.raw_angle_deg).filter(Number.isFinite);
   const filtered=valid.map(r=>r.filtered_angle_deg).filter(Number.isFinite);
   if(raw.length<3||filtered.length<3)throw Error('Fewer than three valid angles; retry or lower overly restrictive thresholds.');
   const span=(rows.at(-1).host_monotonic_ns-rows[0].host_monotonic_ns)/1e9;
   const result={parameter:parameters[step].key,value:activeValue,protocol:'operator-controlled 3 s trial',frames:rows.length,valid_frames:valid.length,
     valid_fraction:valid.length/rows.length,raw_sd:std(raw),filtered_sd:std(filtered),fps:span>0?(rows.length-1)/span:0,
     timestamp:new Date().toISOString(),camera_profile:after.tuning,
     note:'Candidate comparisons require the same physical setup and movement; jitter is not angular accuracy.'};
   evidence.push(result);message(`Recorded ${rows.length} frames, ${valid.length} valid. Repeat with another candidate or move to the next parameter.`);render();
 }catch(e){message('Trial failed: '+e.message);}
 finally{busy=false;byId('measure').disabled=activeValue===null;}
};
byId('next').onclick=()=>{if(busy)return;step=Math.min(parameters.length-1,step+1);render();};
byId('back').onclick=()=>{if(busy)return;step=Math.max(0,step-1);render();};
byId('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema_version:1,tool:'UNO Q Pose Tuning Laboratory',baseline,selected,evidence,disclaimer:'No angular accuracy proven; controlled operator trials.'},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='pose_tuning_evidence_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
byId('save').onclick=async()=>{if(busy)return;if(parameters.some(p=>selected[p.key]===undefined)){byId('saved').textContent='Choose all five parameters before saving.';return;}
 if(!confirm('Save all five chosen values as the main app default? This changes the preset and invalidates any older calibration setup.'))return;
 busy=true;try{const values=Object.fromEntries(parameters.map(p=>[p.key,selected[p.key]]));
   const r=await request('/api/tuning/preset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
   byId('saved').textContent='Preset saved: '+(r.path||'pose_tuning_preset_v1.json')+' · Export the JSON evidence separately.';
 }catch(e){byId('saved').textContent='Save failed: '+e.message;}finally{busy=false;}};
setInterval(async()=>{try{const s=await request('/api/status');const a=s.landmark_confidences||{};byId('live').textContent=`Tracking: ${s.tracking_valid?'VALID':'INVALID'} · angle: ${s.visual_knee_angle_deg==null?'—':s.visual_knee_angle_deg.toFixed(1)+'°'} · Hip ${a.hip==null?'—':a.hip.toFixed(2)} · Knee ${a.knee==null?'—':a.knee.toFixed(2)} · Ankle ${a.ankle==null?'—':a.ankle.toFixed(2)} · ${Number(s.inference_fps||0).toFixed(1)} FPS`; }catch(e){byId('live').textContent='Runtime not connected: '+e.message;}},600);
request('/api/tuning').then(r=>{baseline=r;render();}).catch(e=>message('Could not read runtime tuning: '+e.message));
</script></body></html>'''
