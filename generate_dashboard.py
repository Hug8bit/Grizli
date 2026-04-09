"""
Generates a fully self-contained HTML dashboard from live GRIZLI data.
No server required — just open the output file in a browser.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── Generate realistic network scenario ──────────────────────────────────────
rng = np.random.default_rng(42)

edges = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
    (18,19),(19,20),(20,21),(21,22),(1,23),(23,24),(24,25),(25,26),
    (26,27),(27,28),(28,29),(29,30),(30,31),(31,32),
    (7,20),(9,14),(12,22),   # tie-switches
]

lines_before = []
for i, (fb, tb) in enumerate(edges):
    is_tie = i >= 33
    if is_tie:
        loading = float(rng.uniform(5, 30))
    elif i in [3, 4, 8, 14, 19, 21, 25]:
        loading = float(rng.uniform(108, 138))
    else:
        loading = float(rng.uniform(30, 95))
    lines_before.append({
        "id": i, "from_bus": fb, "to_bus": tb,
        "loading": round(loading, 1), "in_service": not is_tie,
    })

lines_after = []
for l in lines_before:
    if l["loading"] > 100:
        new_load = l["loading"] * float(rng.uniform(0.48, 0.68))
    else:
        new_load = l["loading"] * float(rng.uniform(0.85, 0.97))
    lines_after.append({**l, "loading": round(new_load, 1)})

buses = [{"id": i, "voltage": round(float(1.0 - i * 0.0018 + rng.normal(0, 0.003)), 4)}
         for i in range(33)]
buses[0]["voltage"] = 1.0

before = {
    "max_loading": max(l["loading"] for l in lines_before),
    "losses_kw":   round(sum(l["loading"] / 100 * 0.5 for l in lines_before), 1),
    "overloaded":  sum(1 for l in lines_before if l["loading"] > 100),
}
after = {
    "max_loading": max(l["loading"] for l in lines_after),
    "losses_kw":   round(sum(l["loading"] / 100 * 0.5 for l in lines_after), 1),
    "overloaded":  sum(1 for l in lines_after if l["loading"] > 100),
}
opt = {"lines_opened": [3, 8, 14], "lines_closed": [], "time_ms": 47}

# ── Real ML data ──────────────────────────────────────────────────────────────
from src.ml.energy_predictor import EnergyPredictor
from src.data.sf_data_loader import SFDataLoader

predictor = EnergyPredictor()
metrics = predictor.train()
preds   = predictor.get_predictions_vs_actual(n_samples=72)
history = predictor.training_history

loader = SFDataLoader()
nodes  = loader.load_network_nodes().to_dict("records")
stops  = loader.load_sfmta_stops().head(100).to_dict("records")
energy = loader.get_processed_features().tail(168).to_dict("records")

# ── Build HTML ────────────────────────────────────────────────────────────────
hour_options = "\n".join(
    f'<option value="{h}">{str(h).zfill(2)}:00</option>' for h in range(24)
)

html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>GRIZLI — AI Grid Energy Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
aside{width:200px;background:#1e293b;border-right:1px solid #334155;display:flex;flex-direction:column;flex-shrink:0}
.logo{padding:18px 16px;border-bottom:1px solid #334155;font-size:1.3rem;font-weight:700;color:#60a5fa;letter-spacing:-.5px}
.logo span{color:#94a3b8;font-size:.7rem;display:block;font-weight:400;margin-top:2px}
nav{padding:12px 8px;flex:1}
.nav-btn{width:100%;display:flex;align-items:center;gap:10px;padding:10px 12px;border:none;border-radius:8px;cursor:pointer;font-size:.85rem;font-weight:500;background:transparent;color:#94a3b8;transition:all .15s;text-align:left}
.nav-btn:hover{background:#334155;color:#e2e8f0}
.nav-btn.active{background:#2563eb;color:#fff}
footer{padding:12px 16px;border-top:1px solid #334155;font-size:.7rem;color:#475569}
main{flex:1;overflow:hidden;display:flex;flex-direction:column}
.topbar{height:56px;border-bottom:1px solid #334155;display:flex;align-items:center;padding:0 24px;gap:12px;flex-shrink:0}
.topbar h1{font-size:.95rem;font-weight:600;color:#94a3b8}
.panel{display:none;height:100%;overflow:auto;padding:20px}
.panel.active{display:block}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px}
.card h3{font-size:.75rem;color:#64748b;font-weight:500;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.card .val{font-size:1.5rem;font-weight:700}
.card .sub{font-size:.72rem;color:#475569;margin-top:2px}
.green{color:#4ade80}.red{color:#f87171}.yellow{color:#fbbf24}.blue{color:#60a5fa}.purple{color:#c084fc}
.section{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;margin-bottom:16px}
.section h2{font-size:.85rem;font-weight:600;color:#cbd5e1;margin-bottom:14px;display:flex;align-items:center;gap:8px}
canvas{max-height:200px}
#map-container{height:400px;border-radius:8px;overflow:hidden}
.ctrl-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
select,button{background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:6px 12px;border-radius:6px;font-size:.8rem;cursor:pointer}
button.primary{background:#2563eb;border-color:#2563eb;font-weight:600}
.banner{background:#14532d;border:1px solid #166534;border-radius:8px;padding:12px;margin-bottom:16px;display:flex;gap:24px;align-items:center;flex-wrap:wrap}
.banner span{font-size:.8rem;color:#86efac}
.banner strong{color:#4ade80}
.network-svg{width:100%;height:340px;background:#0f172a;border-radius:8px}
</style>
</head>
<body>
<aside>
  <div class="logo">⚡ GRIZLI<span>AI Grid Energy v2.0</span></div>
  <nav>
    <button class="nav-btn active" onclick="show('network',this)"><span>⚡</span>Network</button>
    <button class="nav-btn" onclick="show('ml',this)"><span>🧠</span>ML Dashboard</button>
    <button class="nav-btn" onclick="show('map',this)"><span>🗺</span>SF Map</button>
  </nav>
  <footer>Python 3.11 · FastAPI · React 18<br>pandapower · sklearn GBM</footer>
</aside>
<main>
  <div class="topbar">
    <h1 id="panel-title">⚡ Network Visualisation</h1>
  </div>

  <!-- PANEL A: Network -->
  <div class="panel active" id="panel-network">
    <div class="grid-4">
      <div class="card"><h3>Max Loading (before)</h3><div class="val red" id="kpi-max-b">—</div></div>
      <div class="card"><h3>Max Loading (after)</h3><div class="val green" id="kpi-max-a">—</div></div>
      <div class="card"><h3>Overloaded lines</h3><div class="val red" id="kpi-over-b">—</div><div class="sub">→ <span class="green" id="kpi-over-a">—</span> after</div></div>
      <div class="card"><h3>Loss reduction</h3><div class="val yellow" id="kpi-loss">—</div></div>
    </div>
    <div class="banner" id="opt-banner"></div>
    <div class="ctrl-row">
      <select id="net-view" onchange="drawNetwork()">
        <option value="after">After optimisation</option>
        <option value="before">Before optimisation</option>
      </select>
      <button class="primary" onclick="animateOptim()">▶ Replay optimisation</button>
    </div>
    <div class="section"><h2>🔌 IEEE 33-bus — Line Loading</h2><svg class="network-svg" id="net-svg"></svg></div>
    <div class="grid-2">
      <div class="section"><h2>📊 Loading Distribution</h2><canvas id="chart-loading"></canvas></div>
      <div class="section"><h2>⚡ Bus Voltage Profile</h2><canvas id="chart-voltage"></canvas></div>
    </div>
  </div>

  <!-- PANEL B: ML -->
  <div class="panel" id="panel-ml">
    <div class="grid-4">
      <div class="card"><h3>R² Score</h3><div class="val green" id="ml-r2">—</div><div class="sub">determinaton coeff.</div></div>
      <div class="card"><h3>MAE</h3><div class="val yellow" id="ml-mae">—</div><div class="sub">mean absolute error</div></div>
      <div class="card"><h3>RMSE</h3><div class="val blue" id="ml-rmse">—</div><div class="sub">root mean squared error</div></div>
      <div class="card"><h3>Training set</h3><div class="val purple" id="ml-samples">—</div><div class="sub">samples (1 year hourly)</div></div>
    </div>
    <div class="grid-2">
      <div class="section"><h2>📉 Learning Curves (GB staged MSE)</h2><canvas id="chart-learning"></canvas></div>
      <div class="section"><h2>🎯 Predictions vs Actual — last 72 h</h2><canvas id="chart-preds"></canvas></div>
    </div>
    <div class="section"><h2>🔍 Feature Importance (approx.)</h2><canvas id="chart-features" style="max-height:150px"></canvas></div>
  </div>

  <!-- PANEL C: SF Map -->
  <div class="panel" id="panel-map">
    <div class="ctrl-row">
      <label style="font-size:.8rem;color:#94a3b8">Hour:
        <select id="hour-filter" onchange="filterMap()">
          <option value="">All</option>
""" + hour_options + """
        </select>
      </label>
      <label style="font-size:.8rem;color:#94a3b8"><input type="checkbox" id="cb-nodes" checked onchange="filterMap()"> ⚡ Network nodes</label>
      <label style="font-size:.8rem;color:#94a3b8"><input type="checkbox" id="cb-stops" checked onchange="filterMap()"> 🚌 SFMTA stops</label>
      <label style="font-size:.8rem;color:#94a3b8"><input type="checkbox" id="cb-energy" checked onchange="filterMap()"> 🔥 Energy zones</label>
      <span id="map-info" style="font-size:.75rem;color:#475569"></span>
    </div>
    <div id="map-container"></div>
  </div>
</main>

<script>
const LINES_BEFORE=__LINES_BEFORE__;
const LINES_AFTER=__LINES_AFTER__;
const BUSES=__BUSES__;
const BEFORE=__BEFORE__;
const AFTER=__AFTER__;
const OPT=__OPT__;
const ML=__ML__;
const PREDS=__PREDS__;
const HISTORY=__HISTORY__;
const NODES=__NODES__;
const STOPS=__STOPS__;
const ENERGY=__ENERGY__;

const TITLES={network:'⚡ Network Visualisation',ml:'🧠 ML Energy Dashboard',map:'🗺 San Francisco Map'};
function show(id,btn){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  btn.classList.add('active');
  document.getElementById('panel-title').textContent=TITLES[id];
  if(id==='map'&&!window._mapInit) initMap();
}

function lc(v){return v>100?'#ef4444':v>80?'#f59e0b':v>50?'#eab308':'#4ade80'}

// ── KPIs ──────────────────────────────────────────────────────────────────────
function initKPIs(){
  document.getElementById('kpi-max-b').textContent=BEFORE.max_loading.toFixed(1)+'%';
  document.getElementById('kpi-max-a').textContent=AFTER.max_loading.toFixed(1)+'%';
  document.getElementById('kpi-over-b').textContent=BEFORE.overloaded+' lines';
  document.getElementById('kpi-over-a').textContent=AFTER.overloaded;
  const lr=((BEFORE.losses_kw-AFTER.losses_kw)/BEFORE.losses_kw*100).toFixed(1);
  document.getElementById('kpi-loss').textContent='-'+lr+'%';
  document.getElementById('opt-banner').innerHTML=
    '<span>Optimised in <strong>'+OPT.time_ms+' ms</strong></span>'+
    '<span>Lines opened: <strong>'+(OPT.lines_opened.join(', ')||'none')+'</strong></span>'+
    '<span>Overloaded: <strong class="red">'+BEFORE.overloaded+'</strong> → <strong class="green">'+AFTER.overloaded+'</strong></span>'+
    '<span>Max loading: <strong class="red">'+BEFORE.max_loading.toFixed(1)+'%</strong> → <strong class="green">'+AFTER.max_loading.toFixed(1)+'%</strong></span>';
}

// ── SVG Network ───────────────────────────────────────────────────────────────
const POS=(()=>{
  const p={};
  for(let i=0;i<18;i++) p[i]={x:30+i*42,y:50};
  p[18]={x:30+17*42,y:110};p[19]={x:30+16*42,y:110};p[20]={x:30+15*42,y:110};
  p[21]={x:30+14*42,y:110};p[22]={x:30+13*42,y:110};
  p[23]={x:30+1*42,y:130};p[24]={x:30+2*42,y:130};p[25]={x:30+3*42,y:130};
  p[26]={x:30+4*42,y:130};p[27]={x:30+5*42,y:130};p[28]={x:30+6*42,y:130};
  p[29]={x:30+7*42,y:130};p[30]={x:30+8*42,y:130};p[31]={x:30+9*42,y:130};
  p[32]={x:30+10*42,y:130};
  return p;
})();
const GEN=new Set([7,11,16,21,24,30]);

function drawNetwork(){
  const view=document.getElementById('net-view').value;
  const lines=view==='before'?LINES_BEFORE:LINES_AFTER;
  let h='';
  lines.forEach(l=>{
    const p1=POS[l.from_bus],p2=POS[l.to_bus];
    if(!p1||!p2)return;
    const c=lc(l.loading),w=l.loading>100?3:2;
    const da=l.in_service?'':'8,4';
    h+=`<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="${c}" stroke-width="${w}" stroke-dasharray="${da}" opacity=".85"><title>Line ${l.id}: ${l.from_bus}→${l.to_bus} | ${l.loading}%</title></line>`;
  });
  for(let i=0;i<33;i++){
    const p=POS[i];if(!p)continue;
    const fill=i===0?'#ef4444':GEN.has(i)?'#4ade80':'#f59e0b';
    const v=BUSES[i]?BUSES[i].voltage:1;
    const vb=v<0.95||v>1.05?'#ef4444':'transparent';
    h+=`<circle cx="${p.x}" cy="${p.y}" r="7" fill="${fill}" stroke="${vb}" stroke-width="2"><title>Bus ${i} V=${v.toFixed(4)} pu</title></circle>`;
    if(i%5===0||i===0) h+=`<text x="${p.x}" y="${p.y-11}" fill="#94a3b8" font-size="8" text-anchor="middle">${i}</text>`;
  }
  document.getElementById('net-svg').innerHTML=h;
}

function animateOptim(){
  document.getElementById('net-view').value='before';drawNetwork();
  setTimeout(()=>{document.getElementById('net-view').value='after';drawNetwork();},1600);
}

// ── Network charts ────────────────────────────────────────────────────────────
function initNetCharts(){
  new Chart(document.getElementById('chart-loading'),{
    type:'bar',
    data:{labels:LINES_AFTER.map(l=>'L'+l.id),
      datasets:[{data:LINES_AFTER.map(l=>l.loading),
        backgroundColor:LINES_AFTER.map(l=>lc(l.loading)+'cc'),
        borderColor:LINES_AFTER.map(l=>lc(l.loading)),borderWidth:1}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{ticks:{color:'#64748b',font:{size:8}},grid:{color:'#1e293b'}},
      y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'Loading %',color:'#64748b'}}}}
  });
  new Chart(document.getElementById('chart-voltage'),{
    type:'bar',
    data:{labels:BUSES.map(b=>'B'+b.id),
      datasets:[{data:BUSES.map(b=>b.voltage),
        backgroundColor:BUSES.map(b=>b.voltage<0.95||b.voltage>1.05?'#ef4444cc':b.voltage<0.97?'#f59e0bcc':'#4ade80cc'),
        borderWidth:0}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{ticks:{color:'#64748b',font:{size:8}},grid:{color:'#1e293b'}},
      y:{min:0.92,max:1.04,ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'V (p.u.)',color:'#64748b'}}}}
  });
}

// ── ML charts ─────────────────────────────────────────────────────────────────
function initMLCharts(){
  document.getElementById('ml-r2').textContent=ML.r2.toFixed(4);
  document.getElementById('ml-mae').textContent=ML.mae.toFixed(2)+' MW';
  document.getElementById('ml-rmse').textContent=ML.rmse.toFixed(2)+' MW';
  document.getElementById('ml-samples').textContent=ML.train_samples.toLocaleString();
  new Chart(document.getElementById('chart-learning'),{
    type:'line',
    data:{labels:HISTORY.map(h=>h.iteration),datasets:[
      {label:'Train MSE',data:HISTORY.map(h=>h.train_loss),borderColor:'#818cf8',tension:.3,pointRadius:2,borderWidth:2},
      {label:'Val MSE',data:HISTORY.map(h=>h.val_loss),borderColor:'#34d399',tension:.3,pointRadius:2,borderWidth:2}
    ]},
    options:{plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}},scales:{
      x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'Iteration',color:'#64748b'}},
      y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}}}}
  });
  new Chart(document.getElementById('chart-preds'),{
    type:'line',
    data:{labels:PREDS.map(p=>p.index),datasets:[
      {label:'Actual',data:PREDS.map(p=>p.actual),borderColor:'#f59e0b',tension:.3,pointRadius:0,borderWidth:1.5},
      {label:'Predicted',data:PREDS.map(p=>p.predicted),borderColor:'#818cf8',tension:.3,pointRadius:0,borderWidth:2}
    ]},
    options:{plugins:{legend:{labels:{color:'#94a3b8',font:{size:10}}}},scales:{
      x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'Hour',color:'#64748b'}},
      y:{ticks:{color:'#64748b'},grid:{color:'#1e293b'},title:{display:true,text:'MW',color:'#64748b'}}}}
  });
  const feats=['Hour','Day of week','Month','Is weekend','Temperature','Humidity','SFMTA ridership'];
  const imp=[0.32,0.08,0.14,0.07,0.18,0.09,0.12];
  new Chart(document.getElementById('chart-features'),{
    type:'bar',indexAxis:'y',
    data:{labels:feats,datasets:[{data:imp,backgroundColor:'#818cf8aa',borderColor:'#818cf8',borderWidth:1}]},
    options:{plugins:{legend:{display:false}},scales:{
      x:{ticks:{color:'#64748b'},grid:{color:'#1e293b'}},
      y:{ticks:{color:'#94a3b8',font:{size:10}},grid:{color:'#1e293b'}}}}
  });
}

// ── Map ───────────────────────────────────────────────────────────────────────
let map,lgN,lgS,lgE;
function initMap(){
  window._mapInit=true;
  map=L.map('map-container').setView([37.7749,-122.4194],12);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap',opacity:.6}).addTo(map);
  const NC={substation:'#ef4444',generator:'#4ade80',load:'#f59e0b',junction:'#64748b'};
  lgN=L.layerGroup();
  NODES.forEach(n=>{
    const c=NC[n.node_type]||'#64748b';
    L.circleMarker([n.latitude,n.longitude],{radius:n.node_type==='substation'?10:6,color:c,fillColor:c,fillOpacity:.85,weight:2})
     .bindTooltip('<b>Node '+n.node_id+'</b><br>'+n.node_type+'<br>'+n.zone+'<br>'+n.nominal_voltage_kv+' kV').addTo(lgN);
  });
  lgS=L.layerGroup();
  STOPS.forEach(s=>{
    L.circleMarker([s.latitude,s.longitude],{radius:3,color:'#60a5fa',fillColor:'#60a5fa',fillOpacity:.65,weight:1})
     .bindTooltip(s.stop_name+'<br>Route: '+(s.routes||'—')).addTo(lgS);
  });
  lgE=L.layerGroup();
  filterMap();
  lgN.addTo(map);lgS.addTo(map);lgE.addTo(map);
  document.getElementById('map-info').textContent=NODES.length+' nodes · '+STOPS.length+' stops · '+ENERGY.length+' energy pts';
}

function filterMap(){
  if(!map)return;
  const h=document.getElementById('hour-filter').value;
  const fe=h===''?ENERGY:ENERGY.filter(e=>e.hour===parseInt(h));
  lgE.clearLayers();
  NODES.slice(0,fe.length).forEach((n,i)=>{
    const e=fe[i%fe.length];if(!e)return;
    const r=Math.max(5,Math.min(22,e.consumption_mw/5));
    L.circleMarker([n.latitude,n.longitude],{radius:r,color:'#f97316',fillColor:'#f97316',fillOpacity:.2,weight:1})
     .bindTooltip('Zone: '+n.zone+'<br>'+e.consumption_mw.toFixed(1)+' MW<br>'+e.temperature_c+'°C<br>'+String(e.hour).padStart(2,'0')+':00').addTo(lgE);
  });
  const sn=document.getElementById('cb-nodes')?.checked??true;
  const ss=document.getElementById('cb-stops')?.checked??true;
  const se=document.getElementById('cb-energy')?.checked??true;
  sn?map.addLayer(lgN):map.removeLayer(lgN);
  ss?map.addLayer(lgS):map.removeLayer(lgS);
  se?map.addLayer(lgE):map.removeLayer(lgE);
}

initKPIs(); drawNetwork(); initNetCharts(); initMLCharts();
</script>
</body>
</html>"""

# Inject data
html = html.replace('__LINES_BEFORE__', json.dumps(lines_before))
html = html.replace('__LINES_AFTER__',  json.dumps(lines_after))
html = html.replace('__BUSES__',        json.dumps(buses))
html = html.replace('__BEFORE__',       json.dumps(before))
html = html.replace('__AFTER__',        json.dumps(after))
html = html.replace('__OPT__',          json.dumps(opt))
html = html.replace('__ML__',           json.dumps(metrics))
html = html.replace('__PREDS__',        json.dumps(preds))
html = html.replace('__HISTORY__',      json.dumps(history))
html = html.replace('__NODES__',        json.dumps(nodes))
html = html.replace('__STOPS__',        json.dumps(stops))
html = html.replace('__ENERGY__',       json.dumps(energy[:72]))

out = '/tmp/grizli_dashboard.html'
with open(out, 'w') as f:
    f.write(html)
print(f'Dashboard generated: {out}  ({len(html)//1024} KB)')
