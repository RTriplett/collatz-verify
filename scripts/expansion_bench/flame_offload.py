#!/usr/bin/env python3
"""Build a self-contained CPU/GPU flame graph from measured timing trees.

CPU cycle samples, CPU core-seconds, GPU operation seconds and diagnostic GPU
lane-clock cycles retain their own units. No conversion between those metrics
is inferred. Run with --input DATA.json --out index.html.
"""

import argparse
import json
import math
from pathlib import Path


def validate(data):
    if data.get("schema_version") not in (1, 2):
        raise ValueError("expected schema_version 1 or 2")
    variants = data.get("variants", [])
    if len(variants) < 2:
        raise ValueError("at least two variants are required")
    identifiers = set()

    def number(value, location):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{location} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{location} must be finite and nonnegative")

    def node(item, location, metric):
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError(f"{location}: missing frame name")
        field = "samples" if metric in ("cpu_samples", "gpu_phase_samples") else "cycles" if metric == "gpu_cycles" else "seconds"
        value = item.get(field, item.get("sample_count") if field == "samples" else None)
        number(value, location + "." + field)
        children = item.get("children", [])
        total = 0
        for index, child in enumerate(children):
            total += node(child, f"{location}.children[{index}]", metric)
        if total > value + max(1e-8, value * 1e-6):
            raise ValueError(f"{location}: child weights exceed parent")
        return value

    for variant in variants:
        identifier = variant.get("variant")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise ValueError("variant identifiers must be distinct strings")
        identifiers.add(identifier)
        for field in ("wall_seconds", "process_cpu_seconds", "deferred_cpu_seconds"):
            if field in variant:
                number(variant[field], identifier + "." + field)
        for device in ("gpu", "cpu", "gpu_phases"):
            if device == "gpu_phases" and device not in variant:
                continue
            root = variant[device]
            metric = root.get("metric", "sum_device_operation_seconds" if device == "gpu" else "cpu_core_seconds")
            if metric not in ("sum_device_operation_seconds", "cpu_core_seconds", "cpu_samples", "gpu_cycles", "gpu_phase_samples"):
                raise ValueError(f"{identifier}.{device}: unsupported metric {metric}")
            node(root, identifier + "." + device, metric)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPU tail offload · algorithm flame graph</title>
<style>
:root{color-scheme:light;--ink:#24272c;--muted:#61646a;--line:#e3ded7;--paper:#fff;--canvas:#f6f4f0;--accent:#a63d16}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1600px;margin:auto;padding:36px 36px 24px}h1{font-size:clamp(25px,3vw,36px);font-weight:700;line-height:1.2;margin:0 0 12px;letter-spacing:-.8px}p{margin:0 0 13px}.subtitle{max-width:1050px;color:var(--muted)}
.eyebrow{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:14px 24px;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:24px 0 10px}
label{font-size:13px;font-weight:650}select,input,button{font:inherit;color:inherit}select,input{border:1px solid #c9c2b8;border-radius:5px;background:white;padding:7px 9px}input{max-width:260px;width:100%}button{border:1px solid #d6cfc5;border-radius:5px;background:#fff;padding:5px 10px;cursor:pointer;font-size:12px}button:hover{background:#f3eee6}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #a95418;outline-offset:2px}button:disabled{cursor:default;opacity:.4}
.control{display:flex;align-items:center;gap:9px}.search{flex:1;min-width:260px}.search label{white-space:nowrap}.help{font-size:12px;color:var(--muted);margin:0 2px 22px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:22px}.column{min-width:0}.column-head{padding:0 4px 16px}.overline{font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:700;color:var(--muted)}h2{font-size:21px;line-height:1.3;margin:4px 0 7px}.variant{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);overflow-wrap:anywhere}.choice{margin:6px 0 0;max-width:100%;font-size:13px}.metrics{display:flex;gap:22px;flex-wrap:wrap;margin-top:17px}.metric strong{display:block;font-size:28px;letter-spacing:-.8px;line-height:1.2}.metric span{font-size:11px;color:var(--muted)}.delta{font-size:12px;color:var(--accent);margin-top:10px;min-height:18px}
.panel{background:var(--paper);border:1px solid var(--line);border-radius:9px;padding:15px 14px 12px;margin-bottom:18px;overflow:hidden}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}.panel h3{font-size:15px;margin:0 0 2px}.unit{font-size:11px;color:var(--muted)}.panel-total{font-size:18px;font-variant-numeric:tabular-nums;white-space:nowrap}.panel-tools{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:12px;min-height:28px}.breadcrumbs{display:flex;gap:5px;align-items:center;flex-wrap:wrap;min-width:0}.breadcrumbs button{max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;padding:3px 7px}.breadcrumbs span{font-size:11px;color:var(--muted)}.exports{display:flex;gap:6px;flex-shrink:0}.flame{width:100%;display:block;overflow:visible;margin-top:8px}.frame{cursor:pointer;outline:none}.frame:hover rect,.frame:focus rect{stroke:#3b2314;stroke-width:2}.frame text{pointer-events:none}.frame rect{shape-rendering:crispEdges}.stage-summary{font-size:11px;color:var(--muted);line-height:1.8;margin-top:5px}.stage-summary strong{font-weight:600;color:var(--ink)}.panel-note{font-size:11px;color:var(--muted);min-height:33px;margin-top:8px}.guide{display:flex;gap:18px;flex-wrap:wrap;padding:0 3px;font-size:12px;color:var(--muted)}.swatch{display:inline-block;width:12px;height:10px;background:#f4a453;border:1px solid #e58731;margin-right:5px}.swatch.unknown{background:#e5ddd1;border-color:#c6bdb0}
details{margin-top:22px;padding:17px 20px;background:#eeebe5;border-radius:8px;font-size:12px;color:var(--muted)}summary{font-weight:650;color:var(--ink);cursor:pointer}details ul{margin:12px 0;padding-left:19px}details li{margin:6px 0}a{color:var(--accent)}.provenance{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.footer{font-size:11px;color:var(--muted);margin-top:18px}
#tooltip{max-height:calc(100vh - 20px);overflow:hidden;position:fixed;pointer-events:none;z-index:10;max-width:440px;padding:12px 14px;border:1px solid #d0c4b4;border-radius:7px;background:rgba(255,252,247,.98);box-shadow:0 4px 18px #39261022;font-size:12px;line-height:1.5;display:none}#tooltip strong{display:block;font-size:13px;margin-bottom:5px}#tooltip .path{color:var(--muted);font-size:11px;margin-top:5px;overflow-wrap:anywhere}#tooltip .raw{font:10px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;margin-top:5px}
@media(max-width:800px){main{padding:22px 14px}.compare{grid-template-columns:1fr;gap:12px}.controls{gap:12px}.control{align-items:flex-start;flex-wrap:wrap}.metrics{gap:18px}.column-head{padding-top:10px}.search{min-width:200px}input{max-width:none}.panel{padding:13px 11px}.panel-total{font-size:16px}}
@media print{main{padding:0}.controls,.exports,.breadcrumbs button,#tooltip,.choice{display:none}.compare{grid-template-columns:1fr 1fr}.panel{break-inside:avoid}body{background:#fff}}
</style></head><body><main>
<div class="eyebrow">Collatz verification · CPU tail offload</div>
<h1>Where computation time was spent</h1>
<p class="subtitle" id="subtitle">A measured-stage flame graph: wider frames mean more measured work; frames above divide the stage below. Click a frame to zoom. GPU and CPU have separate time units because their work overlaps.</p>
<div class="controls">
 <div class="control"><label for="scale">Width scale</label><select id="scale"><option value="absolute">Common absolute scale</option><option value="relative">Normalize each tree</option></select></div>
 <div class="control" id="cpu-detail-control"><label for="cpu-detail">CPU detail</label><select id="cpu-detail"><option value="algorithms">Algorithms and substeps</option><option value="stacks">Full sampled stacks</option></select></div>
 <div class="control search"><label for="search">Find stage</label><input id="search" type="search" placeholder="e.g. tail, screening, transfer" autocomplete="off"><span id="matches" aria-live="polite"></span></div>
 <button id="reset">Reset all zoom</button>
</div>
<p class="help" id="scale-help"></p>
<div id="compare" class="compare"></div>
<div class="guide"><span><i class="swatch"></i>Recorded stage or sampled frame</span><span><i class="swatch unknown"></i>Unsplit or incomplete attribution</span><span>Bottom: total work · top: finer breakdown</span></div>
<details open><summary>How to read this comparison</summary><ul id="notes"></ul><div id="sources" class="provenance"></div><p style="margin-top:12px;margin-bottom:0"><a id="download-data" download="trace_trees.json">Download the source timing trees (JSON)</a></p></details>
<p class="footer"><a href="comparison.png">Static PNG</a> · <a href="comparison.svg">Static SVG</a> · No CPU/GPU durations are added together. Wall-clock duration is shown separately above each comparison. The horizontal position of a flame frame is a category order, not a timeline.</p>
</main><div id="tooltip" role="tooltip"></div><script id="profile-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('profile-data').textContent);
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const selected=[0,data.variants.length-1], zooms=new Map(), panels=[];
const sampled=data.variants.some(v=>v.cpu.metric==='cpu_samples');
const diagnostics=data.variants.some(v=>v.gpu_phases);
let query='', scale='absolute', cpuDetail='algorithms', serial=0;
const fmt=n=>n>=100?n.toFixed(1):n>=1?n.toFixed(3):n===0?'0':n>=.001?n.toFixed(5):n.toExponential(2);
const integer=n=>Math.round(n).toLocaleString('en-US');
const pct=(n,d)=>d>0?(100*n/d).toFixed(n/d<.001?3:2)+'%':'—';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const metrics={
 sum_device_operation_seconds:{field:'seconds',unit:'GPU operation-seconds',short:'GPU operation-s',header:'Summed device-operation seconds',format:fmt,title:'GPU operations',note:'Device operations may overlap across streams. These widths show summed operation durations, not exclusive GPU time.'},
 cpu_core_seconds:{field:'seconds',unit:'CPU core-seconds',short:'CPU core-s',header:'Core-seconds, summed across threads',format:fmt,title:'CPU work',note:'CPU widths measure thread work. Sleeping, waiting and idle time are not CPU core-seconds.'},
 cpu_samples:{field:'samples',unit:'CPU cycle samples',short:'CPU samples',header:'On-CPU cycle samples · no conversion to seconds',format:integer,title:'CPU algorithms and sampled stacks',note:'Each sample belongs to one algorithm category. Higher frames show recovered sampled stacks, including explicitly marked incomplete and compiler inline frames. Sample counts show observed activity; they are not elapsed seconds.'},
 gpu_cycles:{field:'cycles',unit:'sampled lane elapsed cycles',short:'lane clock cycles',header:'Diagnostic lane-clock cycles · includes waits and instrumentation',format:integer,title:'Inside the GPU verifier · diagnostic',note:'Sampled lane elapsed clocks include stalls, scheduling and instrumentation overhead. They are neither SM busy cycles nor a measured split of kernel seconds.'},
 gpu_phase_samples:{field:'samples',unit:'GPU phase samples',short:'GPU samples',header:'GPU diagnostic samples',format:integer,title:'Inside the GPU verifier · diagnostic',note:'This separate diagnostic reports sample attribution, without converting it to GPU operation seconds.'}
};
function el(tag,attrs={},text=null){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==null)n.textContent=text;return n}
function getWeight(n,metric){return n[metrics[metric].field]??n.sample_count??0}
function prep(n,metric,parent=null,path=[]){
 n._id=String(serial++);n._metric=metric;n._value=getWeight(n,metric);n._parent=parent;n._path=[...path,n.name];
 n.children=(n.children||[]).map(c=>prep(c,metric,n,n._path));
 const used=n.children.reduce((a,c)=>a+c._value,0),gap=n._value-used;
 if(n.children.length&&gap>Math.max(1e-8,n._value*1e-6)){
  const rest={name:metric==='cpu_samples'?'Self / unattributed samples':'Unsplit / unattributed within '+n.name,unattributed:true};
  rest[metrics[metric].field]=gap;n.children.push(prep(rest,metric,n,n._path));
 }
 return n;
}
for(const v of data.variants)for(const device of ['gpu','cpu','gpu_phases'])if(v[device])prep(v[device],v[device].metric||(device==='gpu'?'sum_device_operation_seconds':'cpu_core_seconds'));
function hash(s){let h=0;for(let i=0;i<s.length;i++)h=(h*31+s.charCodeAt(i))>>>0;return h}
function unknown(n){return n.unattributed||/unsplit|unattributed|not measured|unresolved|unclassified|other host work|unknown|broken backtrace|truncated/i.test(n.name)}
function color(n){if(unknown(n))return '#e5ddd1';const h=hash(n.name);return `hsl(${18+h%26} ${78+h%15}% ${65+h%16}%)`}
function descendants(n){return[n,...n.children.flatMap(descendants)]}
function operationSummary(root){const nodes=descendants(root),expansion=nodes.find(n=>rawNames(n).some(s=>s.includes('cv_expand_parents')))||nodes.find(n=>/parent expansion|endpoint expansion/i.test(n.name));return[['Endpoint expansion',expansion],['Host → GPU',nodes.find(n=>n.transfer_direction==='H2D')],['GPU → host',nodes.find(n=>n.transfer_direction==='D2H')]].filter(([,n])=>n)}
function depth(n){return n.children.length?1+Math.max(...n.children.map(depth)):1}
function rawNames(n){return[n.raw_function,...(n.raw_functions||[]),n.kernel_name,...(n.raw_kernel_names||[])].filter(Boolean)}
function showTip(event,node,root){
 const metric=metrics[root._metric];let extra='';const count=node.operation_count??node.count;
 if(count!==undefined&&count>0)extra+=`<br>${integer(count)} recorded operations`;
 if(node.sample_count!==undefined&&root._metric!=='cpu_samples')extra+=`<br>${integer(node.sample_count)} samples`;
 if(node.stream_id!==undefined)extra+='<br>CUDA stream: '+esc(node.stream_id);
 if(node.transfer_bytes)extra+='<br>'+integer(node.transfer_bytes)+' transferred bytes';
 if(node.streams)extra+='<br>Streams: '+esc(Array.isArray(node.streams)?node.streams.join(', '):node.streams);
 if(node.source)extra+='<div class="raw">Source: '+esc(node.source)+'</div>';
 if(node.inline)extra+='<br>Compiler inline frame (DWARF attribution)';
 if(node.unresolved)extra+='<br>Unresolved symbol';
 if(node.description)extra+='<div class="path">'+esc(node.description)+'</div>';
 if(node.attribution_method)extra+='<div class="path">Attribution: '+esc(node.attribution_method)+'</div>';
 if(!node.children.length)extra+='<br>No finer recorded breakdown in this frame.';
 const raw=rawNames(node);
 $('tooltip').innerHTML=`<strong>${esc(node.name)}</strong>${metric.format(node._value)} ${metric.unit}<br>${pct(node._value,root._value)} of this tree${node._parent?'<br>'+pct(node._value,node._parent._value)+' of parent stage':''}${extra}<div class="path">${(node._path.length>6?['…',...node._path.slice(-6)]:node._path).map(esc).join(' › ')}</div>${raw.length?'<div class="raw">'+raw.map(esc).join('<br>')+'</div>':''}`;
 $('tooltip').style.display='block';const bounds=event.target.getBoundingClientRect(),x=event.clientX??bounds.left,y=event.clientY??bounds.bottom;
 $('tooltip').style.left=Math.min(Math.max(8,x+14),window.innerWidth-$('tooltip').offsetWidth-10)+'px';
 $('tooltip').style.top=Math.min(Math.max(8,y+14),window.innerHeight-$('tooltip').offsetHeight-10)+'px';
}
function hideTip(){$('tooltip').style.display='none'}
function makeColumn(side){
 const variant=data.variants[selected[side]],column=document.createElement('section');column.className='column';let delta='';
 if(side===1){const before=data.variants[selected[0]].wall_seconds,after=variant.wall_seconds;if(before>0&&after!==undefined)delta=`${after>=before?'+':''}${((after/before-1)*100).toFixed(1)}% wall-clock duration in this measured pair`}
 const cpuTime=variant.process_cpu_seconds??(variant.cpu._metric==='cpu_core_seconds'?variant.cpu._value:null);
 const title=/^before$/i.test(variant.label||'')?'GPU-only tail verification':/^after$/i.test(variant.label||'')?'GPU + CPU tail verification':variant.label||variant.variant;
 column.innerHTML=`<div class="column-head"><div class="overline">${side?'After':'Before'}</div><h2>${esc(title)}</h2><div class="variant">${esc(variant.variant)}</div>${data.variants.length>2?'<select class="choice" aria-label="'+(side?'After':'Before')+' variant">'+data.variants.map((v,i)=>`<option value="${i}"${i===selected[side]?' selected':''}>${esc(v.label||v.variant)}</option>`).join('')+'</select>':''}<div class="metrics"><div class="metric"><strong>${variant.wall_seconds===undefined?'—':fmt(variant.wall_seconds)+' s'}</strong><span>${esc(variant.wall_label||'Profiled wall-clock duration')}</span></div><div class="metric"><strong>${cpuTime===null?'—':fmt(cpuTime)}</strong><span>Measured CPU core-seconds</span></div>${variant.cpu_profile_wall_seconds!==undefined?'<div class="metric"><strong>'+fmt(variant.cpu_profile_wall_seconds)+' s</strong><span>Separate CPU-sampled run</span></div>':''}</div><div class="delta">${esc(delta)}</div></div>`;
 const choice=column.querySelector('.choice');if(choice)choice.addEventListener('change',e=>{selected[side]=+e.target.value;zooms.clear();render()});
 for(const device of ['gpu','gpu_phases','cpu']){
  const root=variant[device];if(!root)continue;
  const metric=metrics[root._metric],key=side+'-'+device,panel=document.createElement('div');panel.className='panel';
  panel.innerHTML=`<div class="panel-head"><div><h3>${metric.title}</h3><div class="unit">${metric.header}</div></div><div class="panel-total">${metric.format(root._value)}${metric.field==='seconds'?' s':''}</div></div><div class="panel-tools"><div class="breadcrumbs"></div><div class="exports"><button class="reset-one" aria-label="Reset ${side?'after':'before'} ${device} zoom">Reset</button><button class="save" aria-label="Download ${side?'after':'before'} ${device} SVG">SVG ↓</button></div></div><div class="canvas"></div><div class="panel-note"></div><div class="stage-summary"></div>`;
  column.appendChild(panel);const info={side,device,key,root,panel,variant};panels.push(info);
  panel.querySelector('.reset-one').onclick=()=>{zooms.delete(key);drawAll()};panel.querySelector('.save').onclick=()=>saveSVG(info);
 }
 return column;
}
function draw(info){
 const {side,device,key,root,panel}=info,metric=metrics[root._metric],focus=zooms.get(key)||root,other=data.variants[selected[1-side]][device];
 const sameMetric=other&&other._metric===root._metric;
 const domain=focus!==root?focus._value:scale==='absolute'&&sameMetric?Math.max(root._value,other._value):root._value;
 const maxLevels=root._metric==='cpu_samples'&&focus===root&&cpuDetail==='algorithms'&&!query?3:Infinity;
 const width=Math.max(420,panel.querySelector('.canvas').clientWidth||600),levels=Math.min(depth(focus),maxLevels),row=29,height=Math.max(155,levels*row+38),base=height-33;
 const svg=el('svg',{viewBox:`0 0 ${width} ${height}`,class:'flame',role:'img','aria-label':`${side?'After':'Before'} ${metric.title}. ${metric.format(root._value)} ${metric.unit}.`});
 svg.appendChild(el('title',{},`${info.variant.label||info.variant.variant}: ${metric.title}`));const defs=el('defs');svg.appendChild(defs);let matched=0;
 function frame(n,start,level){
  const w=domain>0?n._value/domain*width:width,x=start/domain*width||0,y=base-level*row;if(w<.25)return;
  const g=el('g',{class:'frame',role:'button',tabindex:w>7?'0':'-1','aria-label':`${n.name}: ${metric.format(n._value)} ${metric.unit}, ${pct(n._value,root._value)} of total. Activate to zoom.`});
  const hit=!query||(n.name+' '+rawNames(n).join(' ')+' '+(n.description||'')).toLowerCase().includes(query);if(query&&hit)matched++;
  g.appendChild(el('rect',{x:x+.5,y,width:Math.max(0,w-1),height:27,rx:1,fill:color(n),stroke:query&&hit?'#6d2b12':'#fff','stroke-width':query&&hit?2:.7,opacity:hit?1:.25}));
  if(w>25){
   const clipId='clip-'+key+'-'+n._id,clip=el('clipPath',{id:clipId});clip.appendChild(el('rect',{x:x+5,y,width:Math.max(0,w-10),height:27}));defs.appendChild(clip);
   const displayName=n.name+(unknown(n)&&!n.children.length&&!/unsplit|unattributed|unknown|unresolved/i.test(n.name)?' (unsplit)':'');
   const label=w>145?`${displayName} · ${metric.format(n._value)}${metric.field==='seconds'?' s':''}`:displayName;
   g.appendChild(el('text',{x:x+6,y:y+18,'font-family':'system-ui, sans-serif','font-size':'11',fill:'#322717','clip-path':`url(#${clipId})`,opacity:hit?1:.4},label));
  }
  g.addEventListener('mousemove',e=>showTip(e,n,root));g.addEventListener('mouseleave',hideTip);g.addEventListener('focus',e=>showTip(e,n,root));g.addEventListener('blur',hideTip);
  function zoom(){hideTip();zooms.set(key,n);drawAll()}
  g.onclick=zoom;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();zoom()}if(e.key==='Escape'){zooms.delete(key);drawAll()}};
  svg.appendChild(g);let childStart=start;if(level+1<maxLevels)for(const child of n.children){frame(child,childStart,level+1);childStart+=child._value}
 }
 if(root._value>0)frame(focus,0,0);else{svg.appendChild(el('rect',{x:0,y:base,width,height:27,fill:'#eee9e1'}));svg.appendChild(el('text',{x:8,y:base+18,'font-family':'system-ui, sans-serif','font-size':12,fill:'#61646a'},'No recorded work in this tree'))}
 svg.appendChild(el('text',{x:0,y:height-2,'font-family':'system-ui, sans-serif','font-size':10,fill:'#61646a'},'0'));
 svg.appendChild(el('text',{x:width,y:height-2,'text-anchor':'end','font-family':'system-ui, sans-serif','font-size':10,fill:'#61646a'},`${metric.format(domain)} ${metric.short}`));panel.querySelector('.canvas').replaceChildren(svg);
 const crumbs=panel.querySelector('.breadcrumbs');crumbs.replaceChildren();const path=[];for(let n=focus;n;n=n._parent)path.unshift(n);
 for(let i=0;i<path.length;i++){
  if(i){const sep=document.createElement('span');sep.textContent='›';crumbs.appendChild(sep)}
  const button=document.createElement('button');button.textContent=i?path[i].name:'All work';button.title=path[i].name;button.disabled=path[i]===focus;
  button.onclick=()=>{if(path[i]===root)zooms.delete(key);else zooms.set(key,path[i]);drawAll()};crumbs.appendChild(button);
 }
 const summary=panel.querySelector('.stage-summary');
 const summaryNodes=focus===root&&(root._metric==='cpu_samples'||root._metric==='gpu_cycles')?[...root.children].sort((a,b)=>b._value-a._value).slice(0,root._metric==='cpu_samples'?3:4):[];
 summary.innerHTML=focus===root&&root._metric==='sum_device_operation_seconds'?operationSummary(root).map(([label,n])=>'<div><strong>'+label+'</strong> · '+fmt(n._value)+' GPU operation-s</div>').join(''):summaryNodes.map(n=>'<div><strong>'+esc(n.name)+'</strong> · '+pct(n._value,root._value)+(root._metric==='cpu_samples'?' ('+integer(n._value)+' samples)':'')+'</div>').join('');
 panel.querySelector('.reset-one').disabled=focus===root;
 panel.querySelector('.panel-note').textContent=focus!==root?`Zoomed to ${focus.name}: ${pct(focus._value,root._value)} of this tree. Reset to compare full widths.`:metric.note+(root._metric==='cpu_samples'&&maxLevels===3?' Click an algorithm to expand its sampled stack.':'')+(root.description?' '+root.description:'');return matched;
}
function drawAll(){
 hideTip();let matches=0;for(const panel of panels)matches+=draw(panel);$('matches').textContent=query?matches+' visible matches':'';
 $('scale-help').textContent=scale==='relative'?'Each tree fills its panel. Compare the shape and labeled values; equal widths across panels can represent different amounts of work.':'At the root view, before and after share an absolute scale for each metric. CPU samples, GPU lane clocks and operation seconds have separate scales. Zoomed frames fill their own panels.';
}
function render(){
 panels.length=0;$('compare').replaceChildren(makeColumn(0),makeColumn(1));drawAll();
 $('sources').textContent=data.variants.map(v=>(v.label||v.variant)+': '+[v.trace_file,...['gpu','cpu','gpu_phases'].flatMap(k=>v[k]?[v[k].trace_file,v[k].source_file].filter(Boolean):[])].filter(Boolean).join(', ')).join(' · ');
}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function saveSVG(info){const source=info.panel.querySelector('svg'),copy=source.cloneNode(true);copy.setAttribute('xmlns',NS);copy.setAttribute('width',source.viewBox.baseVal.width);copy.setAttribute('height',source.viewBox.baseVal.height);download(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml'}),info.variant.variant+'-'+info.device+'-flame.svg')}
$('subtitle').textContent=sampled?'Measured GPU stages and sampled CPU algorithms. Wider frames mean more recorded work; higher frames divide the stage below. Click a frame to zoom. Units stay separate for each measurement.':'A measured-stage flame graph: wider frames mean more measured work; frames above divide the stage below. Click a frame to zoom. GPU and CPU have separate time units because their work overlaps.';
const notes=[
 sampled?'CPU algorithm categories partition actual on-CPU samples. The frames above each category are sampled call stacks; samples are not converted to seconds. Parent and child widths are inclusive, so do not add a parent to its children.':'This is a measured-stage flame graph, not a sampled call-stack profile. Parent and child widths are inclusive: do not add a parent to its children.',
 'A leaf with no children has no finer recorded breakdown here. Unsplit, unresolved and unattributed regions remain explicit.',
 'GPU operation widths use the sum of individual measured device-operation durations. Concurrent CUDA streams can make that sum exceed wall-clock time. CPU and GPU values are never added together.',
 'The wall-clock figures describe their labeled captures. Profiling overhead and run-to-run variation mean they should not replace the repeated benchmark estimates.',
 ...(diagnostics?['The GPU phase chart is a separate instrumented diagnostic. Lane elapsed-clock cycles include stalls, scheduling and instrumentation; their proportions must not be presented as an exact division of CUDA kernel seconds.']:[]),
 ...(data.notes||[])];
for(const note of notes){const li=document.createElement('li');li.textContent=note;$('notes').appendChild(li)}
if(data.glossary){for(const [name,description]of Object.entries(data.glossary)){const li=document.createElement('li');li.textContent=name+': '+description;$('notes').appendChild(li)}}
$('cpu-detail-control').style.display=sampled?'':'none';$('cpu-detail').onchange=e=>{cpuDetail=e.target.value;drawAll()};
$('download-data').href=URL.createObjectURL(new Blob([$('profile-data').textContent],{type:'application/json'}));$('scale').value='absolute';$('scale').onchange=e=>{scale=e.target.value;drawAll()};$('search').oninput=e=>{query=e.target.value.trim().toLowerCase();drawAll()};$('reset').onclick=()=>{zooms.clear();drawAll()};
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawAll,100)});render();
</script></body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    validate(data)
    # Keep source strings from closing the JSON script element.
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HTML.replace("__DATA__", encoded))
    print(args.out)


if __name__ == "__main__":
    main()
