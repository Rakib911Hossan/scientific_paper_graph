let papers = {}; // arxiv_id -> final_output json (loaded lazily)
let paperMeta = []; // from /api/papers
let activeId = null;
let selected = null; // { type: 'node'|'edge', ... }
let flaggedOnly = false;
let searchTerm = "";
let pan = { x: 40, y: 30 }, zoom = 1;

const svg = document.getElementById("canvas-svg");
const canvasWrap = document.getElementById("canvas-wrap");
const canvasEmpty = document.getElementById("canvas-empty");
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

async function loadPaperList() {
  const res = await fetch("/api/papers");
  paperMeta = await res.json();
  if (paperMeta.length && !activeId) activeId = paperMeta[0].arxiv_id;
  renderTabs();
  if (activeId) await selectPaper(activeId);
  else showEmptyCanvas("No papers found under data/graph/. Run scripts/run_build_graph.py for a paper, or paste a proposition_graph_final.json above.");
}

async function selectPaper(id) {
  activeId = id;
  selected = null;
  pan = { x: 40, y: 30 };
  zoom = 1;
  renderInspectorEmpty();
  if (!papers[id]) {
    const res = await fetch(`/api/graph/${encodeURIComponent(id)}`);
    if (!res.ok) { showEmptyCanvas(`Could not load ${id}.`); return; }
    papers[id] = await res.json();
  }
  render();
}

function showEmptyCanvas(msg) {
  canvasEmpty.style.display = "flex";
  canvasEmpty.textContent = msg;
  svg.innerHTML = "";
  svg.setAttribute("width", 0);
  svg.setAttribute("height", 0);
}
function hideEmptyCanvas() { canvasEmpty.style.display = "none"; }

function buildModel(raw) {
  const nodeById = {};
  const compOf = {};
  raw.components.forEach(c => c.nodes.forEach(n => { compOf[n.id] = c.component_id; }));
  raw.isolated_nodes.forEach(n => { compOf[n.id] = "ISOLATED"; });

  const flaggedByPair = {};
  raw.components.forEach(c => (c.fidelity && c.fidelity.flagged_edges || []).forEach(f => {
    flaggedByPair[f.from + "->" + f.to] = f.reasoning;
  }));

  const allNodes = [...raw.components.flatMap(c => c.nodes), ...raw.isolated_nodes];
  allNodes.forEach(n => { nodeById[n.id] = { ...n, component: compOf[n.id] }; });

  const allEdges = raw.components.flatMap(c => c.edges.map(e => ({
    ...e,
    component: c.component_id,
    flagged: flaggedByPair.hasOwnProperty(e.from + "->" + e.to),
    reasoning: flaggedByPair[e.from + "->" + e.to] || null,
  })));

  const incoming = {}, outgoing = {};
  allEdges.forEach(e => {
    (outgoing[e.from] = outgoing[e.from] || []).push(e);
    (incoming[e.to] = incoming[e.to] || []).push(e);
  });

  return { raw, nodeById, allEdges, incoming, outgoing };
}

function computeLayout(model) {
  const componentsLayout = [];
  let yCursor = 0;
  const colW = 220, rowH = 74, bandPad = 26, bandGap = 34;

  model.raw.components.forEach(c => {
    const localOut = {};
    c.edges.forEach(e => (localOut[e.from] = localOut[e.from] || []).push(e.to));
    const rank = {};
    const idsInComp = c.nodes.map(n => n.id);
    function rankOf(id, seen) {
      if (rank[id] !== undefined) return rank[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      const outs = localOut[id] || [];
      if (!outs.length) { rank[id] = 0; return 0; }
      const r = 1 + Math.max(...outs.map(o => rankOf(o, seen)));
      rank[id] = r; return r;
    }
    idsInComp.forEach(id => rankOf(id, new Set()));

    const byRank = {};
    idsInComp.forEach(id => { (byRank[rank[id]] = byRank[rank[id]] || []).push(id); });
    const maxRank = Math.max(...Object.keys(byRank).map(Number));
    const maxRows = Math.max(...Object.values(byRank).map(a => a.length));

    const positions = {};
    for (let r = 0; r <= maxRank; r++) {
      const ids = (byRank[r] || []).sort();
      ids.forEach((id, i) => {
        positions[id] = { x: bandPad + 70 + r * colW, y: bandPad + 34 + i * rowH };
      });
    }
    const compHeight = bandPad * 2 + 34 + maxRows * rowH;
    const compWidth = bandPad * 2 + 70 + (maxRank + 1) * colW - (colW - 150);

    Object.keys(positions).forEach(id => { positions[id].y += yCursor; });

    componentsLayout.push({
      id: c.component_id, depth: c.depth, yOffset: yCursor, height: compHeight, width: compWidth, positions,
    });
    yCursor += compHeight + bandGap;
  });

  return { componentsLayout, totalHeight: yCursor, colW };
}

function confColor(flagged) {
  return flagged ? "var(--bad)" : "var(--good)";
}

function render() {
  if (!activeId || !papers[activeId]) { showEmptyCanvas("No graph loaded."); return; }
  hideEmptyCanvas();

  const model = buildModel(papers[activeId]);
  window.__model = model;
  const layout = computeLayout(model);
  window.__layout = layout;

  const width = Math.max(900, ...layout.componentsLayout.map(c => c.width + 140));
  const height = Math.max(400, layout.totalHeight);
  svg.setAttribute("width", width * zoom);
  svg.setAttribute("height", height * zoom);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const defs = el("defs", {}, svg);
  ["skip", "good", "bad"].forEach(kind => {
    const m = el("marker", { id: "arrow-" + kind, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: kind === "good" ? "var(--good)" : kind === "bad" ? "var(--bad)" : "var(--skip)" }, m);
  });

  const root = el("g", { transform: `translate(${pan.x},${pan.y})` }, svg);

  layout.componentsLayout.forEach(cl => {
    el("rect", { class: "comp-band", x: 6, y: cl.yOffset + 6, width: cl.width, height: cl.height - 12, rx: 10 }, root);
    el("text", { class: "comp-band-label", x: 20, y: cl.yOffset + 24 }, root).textContent = `${cl.id} — depth ${cl.depth}`;
  });

  const term = searchTerm.trim().toUpperCase();

  model.allEdges.forEach(e => {
    const cl = layout.componentsLayout.find(c => c.id === e.component);
    const p1 = cl.positions[e.from], p2 = cl.positions[e.to];
    if (!p1 || !p2) return;
    const color = confColor(e.flagged);
    const x1 = p1.x + 66, y1 = p1.y, x2 = p2.x - 66, y2 = p2.y;
    const midX = (x1 + x2) / 2;
    const path = `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`;
    const isDim = flaggedOnly && !e.flagged;
    const markerKind = e.flagged ? "bad" : "good";
    const p = el("path", {
      class: "edge-path" + (isDim ? " dimmed" : "") + (selected && selected.type === "edge" && selected.from === e.from && selected.to === e.to ? " selected" : ""),
      d: path, stroke: color, "stroke-width": e.flagged ? "2.2" : "1.6",
      "marker-end": `url(#arrow-${markerKind})`,
    }, root);
    p.addEventListener("click", (ev) => { ev.stopPropagation(); selectEdge(e); });
  });

  Object.entries(model.nodeById).forEach(([id, n]) => {
    if (n.component === "ISOLATED") return;
    const cl = layout.componentsLayout.find(c => c.id === n.component);
    const pos = cl.positions[id];
    if (!pos) return;
    const degree = (model.incoming[id] || []).length + (model.outgoing[id] || []).length;
    const w = 132, h = 46;
    const isSel = selected && selected.type === "node" && selected.id === id;
    const isDim = flaggedOnly && !nodeTouchesFlagged(model, id);
    const isMatch = term && id.toUpperCase() === term;
    const g = el("g", {
      class: "node-group" + (isSel ? " selected" : "") + (isDim ? " dimmed" : "") + (isMatch ? " matched" : ""),
      transform: `translate(${pos.x - w / 2},${pos.y - h / 2})`,
    }, root);
    el("rect", { class: "node-rect", width: w, height: h, rx: 8, "stroke-width": Math.min(1.3 + degree * 0.25, 3) }, g);
    el("text", { class: "node-id", x: 10, y: 17 }, g).textContent = id;
    const label = el("text", { class: "node-label", x: 10, y: 33 }, g);
    label.textContent = truncate(n.name || n.description, 22);
    g.addEventListener("click", (ev) => { ev.stopPropagation(); selectNode(id); });
  });

  renderRail(model);
  renderIsolatedStrip(model);
  renderStats(model);
  renderTabs();

  if (term) scrollToMatch(model, layout, term);
}

function nodeTouchesFlagged(model, id) {
  return [...(model.incoming[id] || []), ...(model.outgoing[id] || [])].some(e => e.flagged);
}

function truncate(s, n) { return (s || "").length > n ? s.slice(0, n - 1) + "…" : (s || ""); }

function renderStats(model) {
  const raw = model.raw;
  const flaggedCount = model.allEdges.filter(e => e.flagged).length;
  const stats = document.getElementById("stats");
  stats.innerHTML = "";
  const items = [
    [Object.keys(model.nodeById).length, "nodes"],
    [model.allEdges.length, "edges"],
    [raw.components.length, "components"],
    [flaggedCount, "flagged"],
    [raw.structural_score ?? "—", "structural"],
  ];
  items.forEach(([v, l]) => {
    const d = document.createElement("div");
    d.className = "stat";
    d.innerHTML = `<b>${v}</b><span>${l}</span>`;
    stats.appendChild(d);
  });
}

function renderTabs() {
  const tabs = document.getElementById("paper-tabs");
  tabs.innerHTML = "";
  paperMeta.forEach(p => {
    const b = document.createElement("button");
    b.className = "paper-tab" + (p.arxiv_id === activeId ? " active" : "");
    b.textContent = p.arxiv_id;
    b.title = `${p.node_count} nodes, ${p.edge_count} edges, ${p.component_count} components`;
    b.addEventListener("click", () => selectPaper(p.arxiv_id));
    tabs.appendChild(b);
  });
}

function renderRail(model) {
  const list = document.getElementById("comp-list");
  list.innerHTML = "";
  model.raw.components.forEach(c => {
    const row = document.createElement("div");
    row.className = "comp-row";
    row.innerHTML = `<span class="id">${c.component_id}</span><span class="meta">${c.nodes.length}n · depth ${c.depth}</span>`;
    row.addEventListener("click", () => panToComponent(c.component_id));
    list.appendChild(row);
  });
  document.getElementById("isolated-note").textContent =
    model.raw.isolated_nodes.length
      ? `${model.raw.isolated_nodes.length} isolated node(s) — no edges, listed below the canvas.`
      : "No isolated nodes.";
}

function renderIsolatedStrip(model) {
  const strip = document.getElementById("isolated-strip");
  strip.innerHTML = "";
  if (!model.raw.isolated_nodes.length) { strip.style.display = "none"; return; }
  strip.style.display = "flex";
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = "Isolated:";
  strip.appendChild(label);
  model.raw.isolated_nodes.forEach(n => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = n.id;
    chip.addEventListener("click", () => selectNode(n.id));
    strip.appendChild(chip);
  });
}

function panToComponent(compId) {
  const cl = window.__layout.componentsLayout.find(c => c.id === compId);
  if (!cl) return;
  pan = { x: 40, y: -cl.yOffset + 30 };
  render();
}

function scrollToMatch(model, layout, term) {
  const id = Object.keys(model.nodeById).find(k => k.toUpperCase() === term);
  if (!id) return;
  const cl = layout.componentsLayout.find(c => c.id === model.nodeById[id].component);
  if (!cl) return;
  const pos = cl.positions[id];
  if (!pos) return;
  pan = { x: -pos.x + 260, y: -pos.y + 200 };
  render();
}

function selectNode(id) {
  selected = { type: "node", id };
  render();
  const model = window.__model;
  const n = model.nodeById[id];
  const outs = model.outgoing[id] || [];
  const ins = model.incoming[id] || [];
  const insp = document.getElementById("inspector");
  insp.innerHTML = `
    <div class="insp-kicker">${n.component === "ISOLATED" ? "Isolated node" : n.component}</div>
    <div class="insp-id">${id}</div>
    <div class="insp-name">${escapeHtml(n.name || "")}</div>
    <div class="insp-body">${escapeHtml(n.description || "")}</div>
    <div class="insp-section"><h3>Incoming (${ins.length})</h3>
      <div class="edge-list">${ins.length ? ins.map(e => edgeItemHtml(e)).join("") : '<span style="color:var(--muted);font-size:12px;">none</span>'}</div>
    </div>
    <div class="insp-section"><h3>Outgoing (${outs.length})</h3>
      <div class="edge-list">${outs.length ? outs.map(e => edgeItemHtml(e)).join("") : '<span style="color:var(--muted);font-size:12px;">none</span>'}</div>
    </div>
  `;
  insp.querySelectorAll("[data-edge]").forEach(row => {
    row.addEventListener("click", () => {
      const [from, to] = row.getAttribute("data-edge").split("|");
      const e = model.allEdges.find(x => x.from === from && x.to === to);
      if (e) selectEdge(e);
    });
  });
}

function edgeItemHtml(e) {
  const kind = e.flagged ? "bad" : "good";
  return `<div class="edge-item" data-edge="${e.from}|${e.to}">
    <span class="dir">${e.from} <span class="arrow">&rarr;</span> ${e.to}</span>
    <span class="pill ${kind}">${e.flagged ? "flagged" : "reviewed"}</span>
  </div>`;
}

function selectEdge(e) {
  selected = { type: "edge", from: e.from, to: e.to };
  render();
  const kind = e.flagged ? "bad" : "good";
  const insp = document.getElementById("inspector");
  const sentences = (papers[activeId] && papers[activeId].sentences) || {};
  insp.innerHTML = `
    <div class="insp-kicker">${e.component} &middot; edge</div>
    <div class="insp-id" style="font-size:16px;">${e.from} &rarr; ${e.to}</div>
    <div class="pill-row" style="margin-top:10px;">
      <span class="pill ${kind}">${e.flagged ? "flagged" : "reviewed"}</span>
    </div>
    <div class="insp-section"><h3>Cited evidence</h3>
      <div class="evidence-list">${(e.evidence || []).map(sid => `<div class="evidence-item"><span class="evidence-id">${sid}</span>${escapeHtml(sentences[sid] || "(sentence text not available)")}</div>`).join("") || '<span style="color:var(--muted);font-size:12px;">none cited</span>'}</div>
    </div>
    ${e.reasoning ? `<div class="insp-section"><h3>Reviewer reasoning</h3><div class="reasoning-box">${escapeHtml(e.reasoning)}</div></div>` : ""}
  `;
}

function renderInspectorEmpty() {
  document.getElementById("inspector").innerHTML =
    '<div class="inspector-empty">Click a node or edge to inspect it.<br><br>Node click shows its description and every incoming/outgoing edge. Edge click shows cited evidence, vote count, and the reviewer\'s reasoning.</div>';
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// pan / zoom
let dragging = false, dragStart = null;
canvasWrap.addEventListener("mousedown", (ev) => {
  if (ev.target.closest(".zoom-btn")) return;
  dragging = true; dragStart = { x: ev.clientX - pan.x * zoom, y: ev.clientY - pan.y * zoom };
  svg.classList.add("panning");
});
window.addEventListener("mousemove", (ev) => {
  if (!dragging) return;
  pan = { x: (ev.clientX - dragStart.x) / zoom, y: (ev.clientY - dragStart.y) / zoom };
  render();
});
window.addEventListener("mouseup", () => { dragging = false; svg.classList.remove("panning"); });
canvasWrap.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  zoom = Math.min(2.2, Math.max(0.35, zoom - ev.deltaY * 0.001));
  render();
}, { passive: false });
document.getElementById("zoom-in").addEventListener("click", () => { zoom = Math.min(2.2, zoom + 0.15); render(); });
document.getElementById("zoom-out").addEventListener("click", () => { zoom = Math.max(0.35, zoom - 0.15); render(); });
document.getElementById("zoom-reset").addEventListener("click", () => { zoom = 1; pan = { x: 40, y: 30 }; render(); });

document.getElementById("search").addEventListener("input", (ev) => { searchTerm = ev.target.value; render(); });
document.getElementById("flagged-toggle").addEventListener("change", (ev) => { flaggedOnly = ev.target.checked; render(); });

svg.addEventListener("click", () => { selected = null; render(); renderInspectorEmpty(); });

document.getElementById("refresh-btn").addEventListener("click", async () => {
  await loadPaperList();
});

// paste-your-own-json
const modal = document.getElementById("paste-modal");
document.getElementById("paste-open").addEventListener("click", () => { modal.style.display = "flex"; document.getElementById("paste-error").style.display = "none"; });
document.getElementById("paste-close").addEventListener("click", () => modal.style.display = "none");
document.getElementById("paste-cancel").addEventListener("click", () => modal.style.display = "none");
document.getElementById("paste-load").addEventListener("click", () => {
  const text = document.getElementById("paste-text").value;
  try {
    const obj = JSON.parse(text);
    if (!obj.components || !obj.arxiv_id) throw new Error("missing fields");
    const id = obj.arxiv_id;
    papers[id] = obj;
    if (!paperMeta.find(p => p.arxiv_id === id)) {
      paperMeta.push({ arxiv_id: id, node_count: "?", edge_count: "?", component_count: obj.components.length, structural_score: obj.structural_score });
    }
    activeId = id;
    selected = null; pan = { x: 40, y: 30 }; zoom = 1;
    modal.style.display = "none";
    renderInspectorEmpty();
    render();
  } catch (err) {
    document.getElementById("paste-error").style.display = "block";
  }
});

loadPaperList();
