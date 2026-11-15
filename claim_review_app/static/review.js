const QUESTIONS = [
 ["Does the evidence support A?", "Yes if the evidence states or directly implies A.", "Claim A unsupported"],
 ["Does the evidence support B?", "Yes if the evidence states or directly implies B.", "Claim B unsupported"],
 ["Does the evidence connect A and B?", "Yes if the evidence ties the two claims to each other. No if they only share a topic, or are just mentioned in the same sentence.", "Not related"],
 ["Does one of these claims depend on the other?", "Yes if one gives a reason, prerequisite, mechanism or basis for establishing the other. No if they are related but neither helps establish the other.", "No dependency"],
 ["Is it B that depends on A (not A on B)?", "A → B means A is needed to establish or explain B.", "Reversed"],
];
const FIELDS = ["reviewer","arxiv_id","from","to","q1","q2","q3","q4","q5","verdict","note","time"];
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/˜/g," ").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const S = {papers:[], paper:null, g:null, pos:0, reviews:{}, answers:{}, notes:{}, showCtx:false};
const key = (p,a,b) => `${p}|${a}|${b}`;
const name = () => $("reviewer").value.trim();
const pdfUrl = p => { const m = p.match(/^\d{4}\.\d{4,5}(v\d+)?/);
 return m ? "https://arxiv.org/pdf/"+m[0] : "https://www.google.com/search?q="+encodeURIComponent(p.replace(/_/g," ")+" paper"); };
const links = () => S.g ? S.g.links : [];
const rk = (i, p=S.paper) => { const l = (p===S.paper?links():S.cache[p]||[])[i]; return key(p,l.from,l.to); };

function verdict(a){
 for (let i=0;i<QUESTIONS.length;i++){ if(!a[i]) return null; if(a[i]==="No") return QUESTIONS[i][2]; }
 return "Correct";
}
async function api(u,o){ const r = await fetch(u,o); return r.json(); }

async function init(){
 S.papers = await api("/api/papers"); S.cache = {};
 const q = new URLSearchParams(location.search).get("paper");
 await openPaper(S.papers.some(p=>p.id===q) ? q : S.papers[0].id);
 $("reviewer").value = localStorage.getItem("reviewer")||"";
 if(name()) await restore();
 render();
}
async function openPaper(p){
 S.paper = p; S.g = await api("/api/graph/"+p); S.cache[p] = S.g.links;
 S.sent = Object.fromEntries(S.g.sentences.map((s,i)=>[s.id,i]));
 history.replaceState(null,"","?paper="+p);
 $("gf").src = "/graph_page/"+p;
 $("pdf").href = pdfUrl(p); $("pdf").textContent = "📄 Open "+p+" PDF ↗";
 S.pos = firstTodo(); 
}
function firstTodo(){ const i = links().findIndex(l=>!S.reviews[key(S.paper,l.from,l.to)]); return i<0?0:i; }
async function restore(){
 const rows = await api("/api/reviews?reviewer="+encodeURIComponent(name()));
 let n=0; rows.forEach(r=>{ if(!S.reviews[key(r.arxiv_id,r.from,r.to)]){ load(r); n++; } });
 if(n){ S.pos = firstTodo(); $("msg").textContent = `Loaded ${n} saved answers`; }
}
function load(r){
 const k = key(r.arxiv_id,r.from,r.to); S.reviews[k]=r;
 S.answers[k] = [1,2,3,4,5].map(i=>r["q"+i]||null); S.notes[k] = r.note||"";
}
function saveCurrent(){
 const l = links()[S.pos], k = key(S.paper,l.from,l.to), a = S.answers[k]||[];
 const v = verdict(a), note = S.notes[k]||"";
 const row = {reviewer:name(), arxiv_id:S.paper, from:l.from, to:l.to,
  q1:a[0]||"",q2:a[1]||"",q3:a[2]||"",q4:a[3]||"",q5:a[4]||"", verdict:v||"", note};
 const old = S.reviews[k];
 if(v===null){ if(old){ delete S.reviews[k]; if(name()) api("/api/reviews",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(row)}); } return; }
 if(!name()) return;
 if(!old || FIELDS.some(f=>f!=="time" && (old[f]||"")!==(row[f]||""))){
  S.reviews[k]=row; api("/api/reviews",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(row)});
 }
}
function renderPapers(){
 const el = $("papers"); el.innerHTML="";
 S.papers.forEach(p=>{
  const done = Object.keys(S.reviews).filter(k=>k.startsWith(p.id+"|")).length;
  const b = document.createElement("button");
  b.textContent = done===p.links ? "✅ "+p.id : `${p.id} · ${done}/${p.links}`;
  if(p.id===S.paper) b.className="primary";
  b.onclick = async()=>{ await openPaper(p.id); render(); };
  el.appendChild(b);
 });
 const i = S.papers.findIndex(p=>p.id===S.paper), n = document.createElement("button");
 n.textContent="Next paper →"; n.disabled = i===S.papers.length-1;
 n.onclick = async()=>{ await openPaper(S.papers[i+1].id); render(); }; el.appendChild(n);
}
function render(){
 renderPapers();
 const L = links(), done = L.filter(l=>S.reviews[key(S.paper,l.from,l.to)]).length;
 $("prog").firstElementChild.style.width = (L.length?100*done/L.length:100)+"%";
 $("progtxt").textContent = `Review progress: ${done} / ${L.length} done · ${L.length-done} remaining`;
 const main = $("main");
 if(!L.length){ $("msg").textContent="No links to review in this paper."; main.classList.add("hide"); return; }
 if(!name()){ $("msg").textContent="Enter your name above to start."; main.classList.add("hide"); return; }
 $("msg").textContent=""; main.classList.remove("hide");
 const l = L[S.pos], k = key(S.paper,l.from,l.to), nodes = S.g.nodes, A = nodes[l.from], B = nodes[l.to];
 // evidence
 let h = `<h3>📄 Evidence from the paper</h3><label><input type="checkbox" id="ctx" ${S.showCtx?"checked":""}> Show surrounding paper text</label>`;
 const evs = l.evidence.length ? l.evidence : [{id:"–",text:"(no evidence sentence cited)"}];
 evs.forEach(ev=>{
  h += `<div class="q"><b>${ev.id.startsWith("S")?"Sentence ":""}${esc(ev.id)}</b>${esc(ev.text)}</div>`;
  const i = S.sent[ev.id];
  if(S.showCtx && i!==undefined){
   h += '<div class="ctx">'+S.g.sentences.slice(Math.max(0,i-2),i+3).map(x=>x.id===ev.id?`<span class="me">▶ ${x.id}: …</span>`:`${x.id}: ${esc(x.text)}`).join("<br>")+"</div>";
  }
 });
 // nav + sel
 let nav = `<div class="row"><select id="sel" style="flex:5;font:inherit;max-width:100%">`+
  L.map((x,i)=>`<option value="${i}" ${i===S.pos?"selected":""}>${i+1}. ${x.from} → ${x.to} · ${esc(nodes[x.from].name)} → ${esc(nodes[x.to].name)}</option>`).join("")+
  `</select><button id="prev" ${S.pos===0?"disabled":""}>← Previous</button><button id="first" ${done===L.length?"disabled":""}>First unreviewed</button><button id="next" ${S.pos===L.length-1?"disabled":""}>Next →</button></div>`;
 $("info").innerHTML = nav + h;
 // questions
 const a = S.answers[k] = S.answers[k] || [null,null,null,null,null];
 let q = `<div class="ab"><a class="pdf" href="${pdfUrl(S.paper)}" target="_blank">📄 Paper PDF ↗</a>
<p><b>A = ${l.from}</b> · ${esc(A.name)}<br>${esc(A.description)}</p><p class="ar">⬇ is the basis for ⬇</p>
<p><b>B = ${l.to}</b> · ${esc(B.name)}<br>${esc(B.description)}</p></div><div class="cap">Answer using only the evidence on the left.</div>`;
 let stopped=false;
 QUESTIONS.forEach(([t,help],i)=>{
  const label=`Q${i+1}. ${t}`;
  if(stopped){ a[i]=null; q+=`<div class="skip">${label} — <i>not needed</i></div>`; return; }
  q += `<div style="margin-top:10px"><b>${label}</b><br><span class="hlp">${help}</span><br>`+
   ["Yes","No"].map(o=>`<label class="opt"><input type="radio" name="q${i}" value="${o}" ${a[i]===o?"checked":""}> ${o}</label>`).join("")+"</div>";
  stopped = a[i]==="No";
 });
 const v = verdict(a);
 q += v===null ? `<div class="v i"><b>Verdict:</b> answer the questions above</div>` :
  v==="Correct" ? `<div class="v ok"><b>Verdict: Correct</b> — all answers support this link</div>` :
  `<div class="v bad"><b>Verdict: ${v}</b> — the link is not supported as drawn</div>`;
 q += `<div>Why did you choose this? (optional)<br><input type="text" id="note" style="width:95%" value="${esc(S.notes[k]||"").replace(/"/g,"&quot;")}" placeholder="e.g. S118 defines d but does not establish B"></div>`;
 const last = S.pos===L.length-1, pi = S.papers.findIndex(p=>p.id===S.paper);
 q += `<div style="margin-top:10px"><button class="primary" id="sn" style="width:100%" ${last&&pi===S.papers.length-1?"disabled":""}>${last?"Save & Next paper →":"Save & Next →"}</button></div>
<div class="cap">Link ${S.pos+1} of ${L.length} · ${done} done in this paper. Answers are saved as soon as there is a verdict.</div>`;
 const tot = Object.keys(S.reviews).length;
 if(tot) q += `<button id="dl" style="width:100%">Download my answers (${tot} links, all papers)</button><div class="cap">Saved on this computer in claim_review_app/reviews/. The download is an optional backup.</div>`;
 $("qa").innerHTML = q;
 // wiring
 saveCurrent();
 $("sel").onchange = e=>{ S.pos=+e.target.value; render(); };
 $("prev").onclick = ()=>{ S.pos--; render(); };
 $("next").onclick = ()=>{ S.pos++; render(); };
 $("first").onclick = ()=>{ S.pos=firstTodo(); render(); };
 $("ctx").onchange = e=>{ S.showCtx=e.target.checked; render(); };
 document.querySelectorAll("#qa input[type=radio]").forEach(r=>r.onchange=e=>{ const i=+e.target.name.slice(1); a[i]=e.target.value; render(); });
 $("note").onchange = e=>{ S.notes[k]=e.target.value; saveCurrent(); };
 $("sn").onclick = async()=>{
  S.notes[k]=$("note").value; saveCurrent();
  if(last){ await openPaper(S.papers[pi+1].id); } else S.pos++;
  render();
 };
 if($("dl")) $("dl").onclick = download;
}
function download(){
 const rows = Object.values(S.reviews).sort((x,y)=>(x.arxiv_id+x.from+x.to).localeCompare(y.arxiv_id+y.from+y.to));
 const c = v=>`"${String(v||"").replace(/"/g,'""')}"`;
 const csv = FIELDS.join(",")+"\n"+rows.map(r=>FIELDS.map(f=>c(r[f])).join(",")).join("\n");
 const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
 a.download = "link_review_"+(name().replace(/[^\w-]/g,"")||"review")+".csv"; a.click();
}
function parseCsv(t){
 const out=[]; let row=[],f="",q=false;
 for(let i=0;i<t.length;i++){ const c=t[i];
  if(q){ if(c=='"'){ if(t[i+1]=='"'){f+='"';i++;} else q=false; } else f+=c; }
  else if(c=='"') q=true; else if(c==","){row.push(f);f="";}
  else if(c=="\n"){row.push(f);out.push(row);row=[];f="";} else if(c!="\r") f+=c; }
 if(f||row.length){row.push(f);out.push(row);} return out;
}
$("reviewer").onchange = async()=>{ localStorage.setItem("reviewer",name()); await restore(); render(); };
$("up").onchange = async e=>{
 const rows = parseCsv(await e.target.files[0].text()), h = rows[0];
 for(const r of rows.slice(1)){ if(r.length<h.length) continue; const o=Object.fromEntries(h.map((x,i)=>[x,r[i]])); load(o);
  if(name()) await api("/api/reviews",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...o,reviewer:name()})}); }
 S.pos=firstTodo(); render();
};
init();
