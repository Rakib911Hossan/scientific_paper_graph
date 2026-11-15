"""Render a paper's handover graph as two standalone HTML pages (no LLM calls).

Reads data/graph_handover/<id>/proposition_graph_final_with_evidence.json (written by
export_graph_handover.py) and writes:

    data/graph_handover/html/<id>_graph.html        every group plus the standalone claims
    data/graph_handover/html/<id>_graph_3plus.html  only groups holding a chain of 3+ claims
                                                    (skipped if the paper has none)

Click a claim to read it with its incoming links' evidence; click the yellow number on an
arrow to see both claims in full and the paper sentences behind the link.

Usage:
    python scripts/export_graph_html.py --arxiv-id 2503.14858

run_full_pipeline.py calls this after export_graph_handover.py.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDOVER = ROOT / "data/graph_handover"


def longest(c):
    ch = {n["id"]: [] for n in c["nodes"]}
    for e in c["edges"]:
        ch[e["from"]].append(e["to"])
    m = {}

    def f(n):
        if n not in m:
            m[n] = max([1 + f(x) for x in ch[n]], default=0)
        return m[n]

    return max(f(n) for n in ch)


tpl = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim graph __AID__</title><style>
body{margin:0;font:15px/1.45 system-ui,sans-serif;background:#f6f7f9;color:#1c2330}
header{padding:12px 16px;background:#fff;border-bottom:1px solid #dde2ec}
h1{font-size:18px;margin:0 0 4px}.help{color:#556;font-size:13px}
#tabs{display:flex;flex-wrap:wrap;gap:6px;padding:10px 16px}
#tabs button{border:1px solid #c5cde0;background:#fff;border-radius:16px;padding:4px 12px;cursor:pointer;font-size:13px}
#tabs button.on{background:#3b6fd4;color:#fff;border-color:#3b6fd4}
#main{display:flex;gap:12px;padding:0 16px 16px;flex-wrap:wrap}
#wrap{flex:1 1 560px;background:#fff;border:1px solid #dde2ec;border-radius:10px;overflow:auto;min-height:300px}
#side{flex:0 0 320px;max-width:100%;background:#fff;border:1px solid #dde2ec;border-radius:10px;padding:12px 14px;align-self:flex-start;position:sticky;top:8px}
#side h2{font-size:16px;margin:0 0 6px}.m{color:#445;font-size:14px}
#side ul{padding-left:18px;margin:4px 0;font-size:13px;color:#445}
.lnk{cursor:pointer;color:#2a5bc0;text-decoration:underline}
.warn{color:#b45f06;font-size:12px}
circle.n{fill:#dbe7fb;stroke:#3b6fd4;stroke-width:2;cursor:pointer}
circle.n:hover{fill:#b9d0f7}circle.sel{fill:#3b6fd4}
g.sel text.id{fill:#fff}
text.id{font-size:12px;font-weight:700;text-anchor:middle;pointer-events:none;fill:#1c3c80}
text.lb{stroke:#fff;stroke-width:3px;paint-order:stroke;font-size:11px;text-anchor:middle;fill:#334;pointer-events:none}
line.e{stroke:#8593b3;stroke-width:1.6}line.flag{stroke:#e08a1e;stroke-dasharray:5 4}
line.dim{opacity:.12}g.dim{opacity:.25}

.q{background:#fffbea;border-left:3px solid #e0b73a;padding:6px 9px;margin:6px 0;font-size:13px;border-radius:4px;color:#332}
.q b{font-size:11px;color:#8a6d0b;display:block}
.ev{margin-top:10px;padding-top:6px;border-top:1px solid #e3e7f0}
.ev h3{font-size:14px;margin:4px 0}
g.bd{cursor:pointer}g.bd circle{fill:#fff3c4;stroke:#e0b73a;stroke-width:1.5}g.bd:hover circle{fill:#ffe48a}
g.bd text{font-size:10px;font-weight:700;text-anchor:middle;fill:#7a5c00;pointer-events:none}
.tip{background:#fff3c4;border:1px solid #e0b73a;border-radius:6px;padding:4px 8px;font-size:12px;display:inline-block;margin-top:4px}
</style></head><body>
<header><h1>Claim graph — paper __AID__</h1>
<div class="help">Each circle is one claim. An arrow <b>A → B</b> means B builds on A (A is the basis). Click a circle to read it and see the paper's own evidence. Dashed orange = weaker link. <b>The yellow 📄 number on each arrow = how many paper sentences back that link — click it to read them.</b></div></header>
<div id="tabs"></div>
<div id="main"><div id="wrap"><svg id="svg"></svg></div><div id="side"><div class="m">Click a circle.</div></div></div>
<script>
const D=__DATA__;
const NS='http://www.w3.org/2000/svg';
const all={};D.components.forEach(c=>c.nodes.forEach(n=>all[n.id]=n));
const comps=D.components.slice().sort((a,b)=>b.nodes.length-a.nodes.length);
const tabs=document.getElementById('tabs');
comps.forEach((c,i)=>{const b=document.createElement('button');b.textContent='Group '+(i+1)+' ('+c.nodes.length+')';b.onclick=()=>show(i);tabs.appendChild(b)});

if(D.isolated_nodes.length){const bi=document.createElement('button');bi.textContent='Standalone ('+D.isolated_nodes.length+')';bi.onclick=()=>show(-1);tabs.appendChild(bi);}
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p&&p.appendChild(e);return e}
function wrap(s,n){const w=s.split(' '),L=[];let c='';w.forEach(x=>{if((c+' '+x).trim().length>n&&c){L.push(c);c=x}else c=(c+' '+x).trim()});if(c)L.push(c);return L.slice(0,3)}
let cur=null;
function show(i){
 [...tabs.children].forEach((b,j)=>b.classList.toggle('on',(i<0?j==comps.length:j==i)));
 const svg=document.getElementById('svg');svg.innerHTML='';const bx=document.getElementById('box');if(bx)bx.innerHTML='';
 document.getElementById('side').innerHTML='<div class="m">Click a circle.</div>';
 if(i<0){side2(D.isolated_nodes);return}
 const c=comps[i];cur=c;
 const par={},ch={};c.nodes.forEach(n=>{par[n.id]=[];ch[n.id]=[]});
 c.edges.forEach(e=>{par[e.to].push(e.from);ch[e.from].push(e.to)});
 const memo={};const L=n=>memo[n]??(memo[n]=par[n].length?1+Math.max(...par[n].map(L)):0);
 const layers=[];c.nodes.forEach(n=>{(layers[L(n.id)]??=[]).push(n.id)});
 const X={},Y={},W=150,H=130,R=22;
 layers.forEach((ids,li)=>{ if(li>0) ids.sort((a,b)=>bary(a)-bary(b)); ids.forEach((id,k)=>{X[id]=k;});
   function bary(n){const p=par[n].map(q=>X[q]).filter(v=>v!==undefined);return p.length?p.reduce((a,b)=>a+b)/p.length:0}
   });
 const maxw=Math.max(...layers.map(l=>l.length));
 layers.forEach((ids,li)=>ids.forEach((id,k)=>{X[id]=(maxw-ids.length)/2*W+k*W+W/2+10;Y[id]=li*H+40}));
 const w=maxw*W+20,h=layers.length*H+40;svg.setAttribute('width',w);svg.setAttribute('height',h);
 const defs=el('defs',{},svg);const m=el('marker',{id:'ar',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},defs);el('path',{d:'M0 0L10 5L0 10z',fill:'#8593b3'},m);
 const fl=new Set((c.fidelity.flagged_edges||[]).map(f=>f.from+'>'+f.to));
 const lines=[],nodes=[],badges=[];
 c.edges.forEach(e=>{const dx=X[e.to]-X[e.from],dy=Y[e.to]-Y[e.from],len=Math.hypot(dx,dy)||1;
  const l=el('line',{x1:X[e.from]+dx/len*R,y1:Y[e.from]+dy/len*R,x2:X[e.to]-dx/len*(R+4),y2:Y[e.to]-dy/len*(R+4),class:'e'+(fl.has(e.from+'>'+e.to)?' flag':''),'marker-end':'url(#ar)'},svg);l.ed=e;lines.push(l);
  const bg=el('g',{class:'bd'},svg);bg.ed=e;el('circle',{cx:(X[e.from]+X[e.to])/2,cy:(Y[e.from]+Y[e.to])/2,r:9},bg);
  const bt=el('text',{x:(X[e.from]+X[e.to])/2,y:(Y[e.from]+Y[e.to])/2+3.5},bg);bt.textContent=e.evidence.length;
  bg.onclick=()=>pickEdge(e,c,lines,nodes,badges,fl);badges.push(bg)});
 c.nodes.forEach(n=>{const g=el('g',{},svg);g.nid=n.id;const ci=el('circle',{cx:X[n.id],cy:Y[n.id],r:R,class:'n'},g);
  const t=el('text',{x:X[n.id],y:Y[n.id]+4,class:'id'},g);t.textContent=n.id;
  wrap(n.name,20).forEach((s,k)=>{const tt=el('text',{x:X[n.id],y:Y[n.id]+R+14+k*12,class:'lb'},g);tt.textContent=s});
  g.onclick=()=>pick(n.id,c,lines,nodes,par,ch,fl,badges);nodes.push(g)});
}
function pick(id,c,lines,nodes,par,ch,fl,badges){
 const near=new Set([id,...par[id],...ch[id]]);
 nodes.forEach(g=>{g.classList.toggle('sel',g.nid==id);g.classList.toggle('dim',!near.has(g.nid))});
 lines.forEach(l=>l.classList.toggle('dim',!(l.ed.from==id||l.ed.to==id)));badges.forEach(b=>b.style.opacity=(b.ed.from==id||b.ed.to==id)?1:.15);
 const n=all[id];let h='<h2>'+id+' · '+esc(n.name)+'</h2><div class="m">'+esc(n.description)+'</div>';
 const ins=c.edges.filter(e=>e.to==id);
 if(!ins.length)h+='<p class="m"><b>A basic fact</b> — it does not build on other claims.</p>';
 ins.forEach(e=>{h+='<p class="m"><b>Builds on</b> <span class="lnk" onclick="jump(\''+e.from+'\')">'+e.from+' · '+esc(all[e.from].name)+'</span>'+(fl.has(e.from+'>'+id)?' <span class="warn">⚠ weaker link</span>':'')+'</p><div class="ev"><h3>📄 Evidence from the paper</h3>'+quotes(e)+'</div>'});
 const outs=c.edges.filter(e=>e.from==id);
 if(outs.length)h+='<p class="m"><b>Used by:</b> '+outs.map(e=>'<span class="lnk" onclick="jump(\''+e.to+'\')">'+e.to+'</span>').join(', ')+'</p>';
 document.getElementById('side').innerHTML=h;
}
function quotes(e){return e.evidence.map(x=>'<div class="q"><b>Sentence '+x.id+'</b>'+esc(x.text)+'</div>').join('')}
function pickEdge(e,c,lines,nodes,badges,fl){
 nodes.forEach(g=>g.classList.toggle('dim',g.nid!=e.from&&g.nid!=e.to));
 lines.forEach(l=>l.classList.toggle('dim',l.ed!==e));badges.forEach(b=>b.style.opacity=b.ed===e?1:.15);
 document.getElementById('side').innerHTML='<h2>Why is this link here?</h2><p class="m"><span class="lnk" data-id="'+e.from+'" onclick="jump(this.dataset.id)"><b>'+e.from+'</b> · '+esc(all[e.from].name)+'</span><br><span class="m">'+esc(all[e.from].description)+'</span></p><p class="m" style="text-align:center">⬇ is the basis for ⬇</p><p class="m"><span class="lnk" data-id="'+e.to+'" onclick="jump(this.dataset.id)"><b>'+e.to+'</b> · '+esc(all[e.to].name)+'</span><br><span class="m">'+esc(all[e.to].description)+'</span></p>'+(fl.has(e.from+'>'+e.to)?'<p class="warn">⚠ weaker link — the quality check doubted this one</p>':'')+'<div class="ev"><h3>📄 Evidence from the paper</h3>'+quotes(e)+'</div>'}
function jump(id){[...document.querySelectorAll('#svg g')].find(g=>g.nid==id)?.dispatchEvent(new Event('click'))}
function side2(list){document.getElementById('svg').setAttribute('height',0);document.getElementById('svg').setAttribute('width',0);
 let b=document.getElementById('box');if(!b){b=document.createElement('div');b.id='box';b.style.padding='12px';document.getElementById('wrap').appendChild(b)}b.innerHTML=list.map(n=>'<p class="m"><b>'+n.id+' · '+esc(n.name)+'</b><br>'+esc(n.description)+'</p>').join('')}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
show(0);
</script></body></html>'''


def render_page(d, arxiv_id, big):
    """Page HTML for one paper's handover graph, or None if big and no 3+-claim chain.

    Also used by hf_space/app.py, so the hosted viewer matches these pages."""
    if big:
        d = dict(d, components=[c for c in d["components"] if longest(c) >= 2], isolated_nodes=[])
        if not d["components"]:
            return None
    data = json.dumps(d).replace("</", "<\\/")
    return tpl.replace("__AID__", arxiv_id).replace("__DATA__", data)


def write_page(arxiv_id, big):
    d = json.loads((HANDOVER / arxiv_id / "proposition_graph_final_with_evidence.json").read_text())
    page = render_page(d, arxiv_id, big)
    if page is None:
        print(f"{arxiv_id}: no group with a 2-hop chain, 3plus page skipped")
        return
    out = HANDOVER / "html" / f"{arxiv_id}_graph{'_3plus' if big else ''}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{arxiv_id}: wrote {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arxiv-id", required=True)
    args = parser.parse_args()
    write_page(args.arxiv_id, big=False)
    write_page(args.arxiv_id, big=True)


if __name__ == "__main__":
    main()
