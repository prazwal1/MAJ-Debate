const state = {
  bundle: null,
  topics: [],
  activeTopicId: null,
  edgeFilter: "all",
  activeTab: "arguments",
  query: "",
  flagFilter: "all",
};

const els = {
  search: document.querySelector("#search"),
  flagFilter: document.querySelector("#flagFilter"),
  globalExperiments: document.querySelector("#globalExperiments"),
  topicList: document.querySelector("#topicList"),
  eyebrow: document.querySelector("#eyebrow"),
  topicTitle: document.querySelector("#topicTitle"),
  verdictChips: document.querySelector("#verdictChips"),
  metrics: document.querySelector("#metrics"),
  experimentSummary: document.querySelector("#experimentSummary"),
  graph: document.querySelector("#graph"),
  graphCaption: document.querySelector("#graphCaption"),
  selectedEdge: document.querySelector("#selectedEdge"),
  graphNoInternal: document.querySelector("#graphNoInternal"),
  graphCaptionNoInternal: document.querySelector("#graphCaptionNoInternal"),
  selectedEdgeNoInternal: document.querySelector("#selectedEdgeNoInternal"),
  verdictTable: document.querySelector("#verdictTable"),
  detailPanel: document.querySelector("#detailPanel"),
};

const verdictClass = (value) => String(value || "").toLowerCase();
const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

fetch("data/inspector_bundle.json")
  .then((response) => response.json())
  .then((bundle) => {
    state.bundle = bundle;
    state.topics = bundle.topics || [];
    state.activeTopicId = state.topics[0]?.topic_id;
    bindEvents();
    renderGlobalExperiments();
    render();
  })
  .catch((error) => {
    els.topicTitle.textContent = "Could not load inspector data";
    els.eyebrow.textContent = error.message;
  });

function bindEvents() {
  els.search.addEventListener("input", (event) => {
    state.query = event.target.value.toLowerCase();
    renderTopicList();
  });
  els.flagFilter.addEventListener("change", (event) => {
    state.flagFilter = event.target.value;
    renderTopicList();
  });
  document.querySelectorAll("[data-edge-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.edgeFilter = button.dataset.edgeFilter;
      document.querySelectorAll("[data-edge-filter]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      renderGraph(activeTopic());
    });
  });
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTab = button.dataset.tab;
      document.querySelectorAll("[data-tab]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      renderDetail(activeTopic());
    });
  });
}

function activeTopic() {
  return state.topics.find((topic) => topic.topic_id === state.activeTopicId) || state.topics[0];
}

function render() {
  renderTopicList();
  renderTopic(activeTopic());
}

function renderTopicList() {
  const filtered = state.topics.filter((topic) => {
    const haystack = `${topic.topic_id} ${topic.topic_text} ${topic.domain} ${topic.dataset}`.toLowerCase();
    const matchesSearch = !state.query || haystack.includes(state.query);
    const matchesFlag = state.flagFilter === "all" || (topic.flags || []).includes(state.flagFilter);
    return matchesSearch && matchesFlag;
  });

  els.topicList.innerHTML = filtered
    .map(
      (topic) => `
        <button class="topic-button ${topic.topic_id === state.activeTopicId ? "active" : ""}" data-topic-id="${esc(topic.topic_id)}">
          <strong>${esc(topic.topic_id)} · ${esc(topic.domain)}</strong>
          <span>${esc(topic.topic_text)}</span>
        </button>`
    )
    .join("");

  els.topicList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeTopicId = button.dataset.topicId;
      render();
    });
  });
}

function signedDelta(newValue, oldValue) {
  if (newValue === undefined || oldValue === undefined || newValue === null || oldValue === null) return "NA";
  const delta = Number(newValue) - Number(oldValue);
  return `${delta > 0 ? "+" : ""}${delta.toFixed(1)}`;
}

function deltaClass(newValue, oldValue) {
  if (newValue === undefined || oldValue === undefined || newValue === null || oldValue === null) return "delta-flat";
  const delta = Number(newValue) - Number(oldValue);
  if (delta > 0) return "delta-good";
  if (delta < 0) return "delta-bad";
  return "delta-flat";
}

function renderGlobalExperiments() {
  const experiment = state.bundle?.experiments?.selected10_no_internal;
  if (!experiment?.available) {
    els.globalExperiments.innerHTML = "";
    return;
  }

  const rows = experiment.rows || [];
  els.globalExperiments.innerHTML = `
    <h2>Experiments</h2>
    <article class="global-experiment-card">
      <h3>Selected 10 · no internal attacks</h3>
      <table class="global-experiment-table">
        <thead>
          <tr>
            <th>Config</th>
            <th>Graph</th>
            <th>Judge</th>
            <th>Drop</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  <td>${esc(row.config)}</td>
                  <td class="${deltaClass(row.no_internal_graph_acc_pct, row.original_graph_acc_pct)}">${esc(signedDelta(row.no_internal_graph_acc_pct, row.original_graph_acc_pct))}</td>
                  <td class="${deltaClass(row.no_internal_stage4_acc_pct, row.original_stage4_acc_pct)}">${esc(signedDelta(row.no_internal_stage4_acc_pct, row.original_stage4_acc_pct))}</td>
                  <td>${esc(row.dropped_same_stance_attack_edges)}</td>
                </tr>`
            )
            .join("")}
        </tbody>
      </table>
    </article>`;
}

function renderTopic(topic) {
  if (!topic) return;
  const full = topic.verdicts?.full || {};
  const graph = topic.stage3?.graph_verdict || {};
  els.eyebrow.textContent = `${topic.topic_id} · ${topic.dataset} · ${topic.domain}`;
  els.topicTitle.textContent = topic.topic_text;
  els.verdictChips.innerHTML = [
    chip(`Gold ${topic.benchmark_label}`, "good"),
    chip(`Full ${full.verdict || "missing"}`, full.verdict === topic.benchmark_label ? "good" : "bad"),
    chip(`Graph ${graph.winner || "missing"}`, graph.winner === topic.benchmark_label ? "good" : "bad"),
  ].join("");
  renderMetrics(topic);
  renderExperimentSummary(topic);
  renderVerdicts(topic);
  renderGraph(topic, {
    svg: els.graph,
    caption: els.graphCaption,
    detail: els.selectedEdge,
    relations: topic.relations || [],
    relationSource: topic.relation_source || "full",
  });
  renderNoInternalGraph(topic);
  renderDetail(topic);
}

function renderExperimentSummary(topic) {
  const experiment = state.bundle?.experiments?.selected10_no_internal;
  if (!experiment?.available) {
    els.experimentSummary.innerHTML = "";
    return;
  }
  const rows = experiment.rows || [];
  els.experimentSummary.innerHTML = `
    <div class="experiment-grid">
      ${rows
        .map((row) => {
          const topicResult = topic.no_internal?.[row.config] || {};
          const oldGraph = row.original_graph_acc_pct;
          const newGraph = row.no_internal_graph_acc_pct;
          const oldStage4 = row.original_stage4_acc_pct;
          const newStage4 = row.no_internal_stage4_acc_pct;
          return `
            <article class="experiment-card">
              <h3>${esc(row.config)} · no-internal</h3>
              <dl>
                <dt>Graph acc</dt><dd>${esc(oldGraph)} -> ${esc(newGraph)}</dd>
                <dt>Stage4 acc</dt><dd>${esc(oldStage4)} -> ${esc(newStage4)}</dd>
                <dt>Dropped edges</dt><dd>${esc(row.dropped_same_stance_attack_edges)}</dd>
                <dt>This graph</dt><dd>${esc(topicResult.graph_verdict || "NA")}</dd>
                <dt>This Stage4</dt><dd>${esc(topicResult.stage4_verdict || "NA")}</dd>
              </dl>
            </article>`;
        })
        .join("")}
    </div>`;
}

function chip(text, klass = "") {
  return `<span class="chip ${klass}">${esc(text)}</span>`;
}

function renderMetrics(topic) {
  const attacks = topic.relation_counts?.Attack || 0;
  const supports = topic.relation_counts?.Support || 0;
  const grounded = topic.stage3?.grounded_size ?? "0";
  const flags = (topic.flags || []).map((flag) => flag.replaceAll("_", " ")).join(", ") || "none";
  els.metrics.innerHTML = [
    metric("Arguments", topic.arguments.length),
    metric("Attacks", attacks),
    metric("Supports", supports),
    metric("Grounded", grounded),
    metric("Signals", flags),
  ].join("");
}

function metric(label, value) {
  return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
}

function renderVerdicts(topic) {
  const rows = Object.entries(topic.verdicts || {}).map(([config, verdict]) => {
    const correct = verdict.verdict === topic.benchmark_label;
    const noInternal = topic.no_internal?.[config];
    const extra = noInternal
      ? `<small>No-internal: graph ${esc(noInternal.graph_verdict || "NA")} / judge ${esc(noInternal.stage4_verdict || "NA")}</small>`
      : "";
    return `
      <div class="verdict-row">
        <div>
          <div class="config">${esc(config)}</div>
          <small>${esc(verdict.rationale || "")}</small>
          ${extra}
        </div>
        <span class="pill ${verdictClass(verdict.verdict)}">${esc(verdict.verdict || "NA")}</span>
        <span class="pill ${correct ? "pro" : "con"}">${correct ? "gold match" : "miss"}</span>
      </div>`;
  });
  els.verdictTable.innerHTML = rows.join("");
}

function renderGraph(topic, options) {
  const svg = options.svg;
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 520;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const args = topic.arguments || [];
  const argById = new Map(args.map((arg) => [arg.arg_id, arg]));
  const edges = (options.relations || []).filter(
    (edge) => state.edgeFilter === "all" || edge.label === state.edgeFilter
  );
  options.caption.textContent = `${edges.length} ${state.edgeFilter === "all" ? "kept Attack/Support" : state.edgeFilter} relations shown · source: ${options.relationSource || "full"}`;

  const pro = args.filter((arg) => arg.stance === "PRO");
  const con = args.filter((arg) => arg.stance === "CON");
  const positions = new Map();
  layoutSide(pro, width * 0.27, positions, height);
  layoutSide(con, width * 0.73, positions, height);

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrowAttack" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#b42318"></path>
    </marker>
    <marker id="arrowSupport" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#2f7d32"></path>
    </marker>`;
  svg.appendChild(defs);

  edges.forEach((edge) => {
    const source = positions.get(edge.source_arg_id);
    const target = positions.get(edge.target_arg_id);
    if (!source || !target) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("class", "edge");
    line.setAttribute("stroke", edge.label === "Attack" ? "#b42318" : "#2f7d32");
    line.setAttribute("stroke-width", String(1.2 + Number(edge.confidence || 0) * 1.7));
    line.setAttribute("marker-end", edge.label === "Attack" ? "url(#arrowAttack)" : "url(#arrowSupport)");
    line.addEventListener("click", () => showRelation(edge, argById, options.detail));
    svg.appendChild(line);
  });

  args.forEach((arg) => {
    const pos = positions.get(arg.arg_id);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "node");
    g.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    g.innerHTML = `
      <circle r="12" fill="${arg.stance === "PRO" ? "#0f8b8d" : "#c44536"}"></circle>
      <text x="17" y="4">${esc(shortArgId(arg.arg_id))}</text>`;
    g.addEventListener("click", () => showArgument(arg, topic, options.detail));
    svg.appendChild(g);
  });

  addLaneLabel(svg, width * 0.27, 24, "PRO");
  addLaneLabel(svg, width * 0.73, 24, "CON");
}

function renderNoInternalGraph(topic) {
  const variant = topic.no_internal?.full || topic.no_internal?.dung_no_agents || topic.no_internal?.two_agents;
  if (!variant) {
    els.graphNoInternal.setAttribute("viewBox", "0 0 900 520");
    els.graphNoInternal.innerHTML = "";
    els.graphCaptionNoInternal.textContent = "Selected10 no-internal graph files not loaded for this topic";
    els.selectedEdgeNoInternal.textContent = "No no-internal experiment data is available for this topic.";
    return;
  }

  renderGraph(topic, {
    svg: els.graphNoInternal,
    caption: els.graphCaptionNoInternal,
    detail: els.selectedEdgeNoInternal,
    relations: variant.relations || [],
    relationSource: variant.relation_source || "selected10 no-internal mixed stage2",
  });
  els.selectedEdgeNoInternal.innerHTML = `<strong>No-internal verdicts</strong><br>${Object.entries(topic.no_internal)
    .map(
      ([config, info]) =>
        `${esc(config)}: ${esc((info.relations || []).length)} attacks, graph ${esc(info.graph_verdict || "NA")} / judge ${esc(info.stage4_verdict || "NA")}`
    )
    .join("<br>")}`;
}

function layoutSide(args, x, positions, height) {
  const sorted = [...args].sort((a, b) => (a.round || 0) - (b.round || 0) || a.arg_id.localeCompare(b.arg_id));
  const gap = Math.max(24, (height - 80) / Math.max(1, sorted.length - 1));
  sorted.forEach((arg, index) => {
    const jitter = arg.round === 2 ? 42 : 0;
    positions.set(arg.arg_id, { x: x + jitter, y: 54 + index * gap });
  });
}

function addLaneLabel(svg, x, y, label) {
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x - 20);
  text.setAttribute("y", y);
  text.setAttribute("fill", "#667085");
  text.setAttribute("font-size", "12");
  text.setAttribute("font-weight", "700");
  text.textContent = label;
  svg.appendChild(text);
}

function shortArgId(argId) {
  return String(argId || "").split("_").pop();
}

function showArgument(arg, topic, detailEl) {
  const strength = topic.argument_strength?.[arg.arg_id];
  detailEl.innerHTML = `<strong>${esc(arg.arg_id)}</strong> · ${esc(arg.stance)} · round ${esc(arg.round)} · ${esc(arg.persona)}<br>${esc(arg.text)}${strength ? `<br>Strength: ${esc(strength.strength)} · ${esc(strength.rationale)}` : ""}`;
}

function showRelation(edge, argById, detailEl) {
  const source = argById.get(edge.source_arg_id);
  const target = argById.get(edge.target_arg_id);
  detailEl.innerHTML = `<strong>${esc(edge.label)}</strong> ${esc(edge.source_arg_id)} -> ${esc(edge.target_arg_id)} · confidence ${esc(edge.confidence)}<br>${esc(edge.justification || edge.premise || "")}<br><strong>Source:</strong> ${esc(source?.text || "")}<br><strong>Target:</strong> ${esc(target?.text || "")}`;
}

function renderDetail(topic) {
  if (state.activeTab === "relations") return renderRelations(topic);
  if (state.activeTab === "rationales") return renderRationales(topic);
  renderArguments(topic);
}

function renderArguments(topic) {
  const column = (stance) => `
    <div>
      <h3 class="section-title">${stance}</h3>
      <div class="arg-list">
        ${topic.arguments
          .filter((arg) => arg.stance === stance)
          .map(
            (arg) => `
              <article class="arg-card">
                <header><strong>${esc(arg.arg_id)}</strong><span>round ${esc(arg.round)} · ${esc(arg.persona)}</span></header>
                <p>${esc(arg.text)}</p>
              </article>`
          )
          .join("")}
      </div>
    </div>`;
  els.detailPanel.innerHTML = `<div class="arg-grid">${column("PRO")}${column("CON")}</div>`;
}

function renderRelations(topic) {
  const rows = [...(topic.relations || [])]
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .slice(0, 120)
    .map(
      (rel) => `
        <article class="relation-card">
          <header>
            <strong>${esc(rel.label)} · ${esc(rel.source_arg_id)} -> ${esc(rel.target_arg_id)}</strong>
            <span>${esc(rel.confidence)}</span>
          </header>
          <p>${esc(rel.justification || rel.premise || "No rationale captured.")}</p>
        </article>`
    );
  els.detailPanel.innerHTML = `<div class="relation-list">${rows.join("")}</div>`;
}

function renderRationales(topic) {
  const rows = Object.entries(topic.verdicts || {}).map(([config, verdict]) => {
    const noInternal = topic.no_internal?.[config];
    return `
      <article class="rationale-card">
        <header>
          <strong>${esc(config)} · ${esc(verdict.verdict)}</strong>
          <span>confidence ${esc(verdict.confidence ?? "NA")}</span>
        </header>
        <p>${esc(verdict.rationale || "No rationale captured.")}</p>
        ${
          noInternal
            ? `<p><strong>No-internal:</strong> graph ${esc(noInternal.graph_verdict || "NA")}, judge ${esc(noInternal.stage4_verdict || "NA")}. ${esc(noInternal.stage4_rationale || "")}</p>`
            : ""
        }
      </article>`;
  });
  els.detailPanel.innerHTML = `<div class="rationale-list">${rows.join("")}</div>`;
}
