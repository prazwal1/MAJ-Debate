/* ── MAJ Debate — frontend logic ─────────────────────────────────────────── */

let _network    = null;
let _physicsOn  = true;
let _labelsOn   = true;
let _allRels    = [];
let _argStrength= {};
let _stage3Data = null;
let _allArgs    = [];

/* ── Init ────────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => { loadModels(); });

async function loadModels() {
  try {
    const { models } = await fetch('/api/models').then(r => r.json());
    const sel = document.getElementById('model-select');
    sel.innerHTML = models.map(m => `<option value="${m.id}">${m.label}</option>`).join('');
    updateHeaderModel(models[0]?.id ?? '');
    sel.addEventListener('change', () => updateHeaderModel(sel.value));
  } catch (e) { console.error('Could not load models', e); }
}

function updateHeaderModel(id) {
  const el = document.getElementById('header-model');
  el.textContent = id.split('/').pop() || id;
  el.title = id;
}

/* ── Start debate ────────────────────────────────────────────────────────── */
async function startDebate() {
  const topic = document.getElementById('topic-input').value.trim();
  if (!topic) { alert('Enter a debate topic.'); return; }

  const model  = document.getElementById('model-select').value;
  const apiKey = document.getElementById('api-key-input').value.trim();

  const checkedPro = [...document.querySelectorAll('input[name="pro"]:checked')].map(i => i.value);
  const checkedCon = [...document.querySelectorAll('input[name="con"]:checked')].map(i => i.value);
  if (!checkedPro.length || !checkedCon.length) {
    alert('Select at least one PRO and one CON persona.');
    return;
  }

  const stages = [...document.querySelectorAll('.stage-cb:checked')].map(i => parseInt(i.value));

  const config = {
    topic,
    model,
    api_key:              apiKey || null,
    n_pro:                checkedPro.length,
    n_con:                checkedCon.length,
    r1_args:              parseInt(document.getElementById('r1-args').value)      || 3,
    r2_args:              parseInt(document.getElementById('r2-args').value)      || 2,
    confidence_threshold: parseFloat(document.getElementById('conf-threshold').value) || 0.65,
    pair_batch_size:      parseInt(document.getElementById('batch-size').value)   || 40,
    targeted_attacks:     document.getElementById('targeted-attacks').checked,
    run_stage1: stages.includes(1),
    run_stage2: stages.includes(2),
    run_stage3: stages.includes(3),
    run_stage4: stages.includes(4),
  };

  _allRels = []; _argStrength = {}; _stage3Data = null; _allArgs = [];
  resetProgress();
  showState('running');
  document.getElementById('run-topic').textContent = topic;
  document.getElementById('run-btn').disabled = true;
  log('Starting debate analysis…');

  try {
    const res = await fetch('/api/debate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to start');
    }
    const { job_id } = await res.json();
    log(`Job: ${job_id}`);
    connectSSE(job_id, topic, config);
  } catch (e) { showError(e.message); }
}

/* ── SSE ─────────────────────────────────────────────────────────────────── */
function connectSSE(jobId, topic, config) {
  const es = new EventSource(`/api/events/${jobId}`);
  es.onmessage = ev => {
    try { handleEvent(JSON.parse(ev.data), topic, config, es); }
    catch (e) { console.error('Event parse error', e); }
  };
  es.onerror = () => {
    es.close();
    showError('Connection lost. Analysis may still be running — refresh to check.');
    document.getElementById('run-btn').disabled = false;
  };
}

function handleEvent(event, topic, config, es) {
  switch (event.type) {
    case 'start':
      log(`Model: ${event.model}`);
      break;

    case 'progress': {
      const pi = document.getElementById(`pi-${event.stage}`);
      if (pi) {
        pi.classList.add('active');
        document.getElementById(`pm-${event.stage}`).textContent = event.message || '';
      }
      log(event.message || '');
      break;
    }

    case 'stage_done': {
      const pi = document.getElementById(`pi-${event.stage}`);
      if (pi) {
        pi.classList.remove('active');
        pi.classList.add('done');
        document.getElementById(`pm-${event.stage}`).textContent = 'Done';
      }
      if (event.stage === 1 && event.data?.arguments) _allArgs = event.data.arguments;
      if (event.stage === 2 && event.data) {
        _allRels     = event.data.relations         || [];
        _argStrength = event.data.argument_strength || {};
        _allArgs     = event.data.arguments         || _allArgs;
      }
      if (event.stage === 3 && event.data) _stage3Data = event.data;
      log(`Stage ${event.stage} complete.`);
      break;
    }

    case 'complete':
      es.close();
      document.getElementById('run-btn').disabled = false;
      showResults(topic, event.data, config);
      break;

    case 'error':
      es.close();
      document.getElementById('run-btn').disabled = false;
      showError(event.message || 'Unknown error');
      break;

    case 'ping': break;
    default: log(`[${event.type}]`);
  }
}

/* ── Render results ──────────────────────────────────────────────────────── */
function showResults(topic, data, config) {
  const s2 = data.stage2 || {};
  const s3 = data.stage3 || _stage3Data;
  const s4 = data.stage4;

  const allArgs  = s2.arguments        || _allArgs    || [];
  const allRels  = s2.relations        || _allRels    || [];
  const strength = s2.argument_strength|| _argStrength|| {};
  const summary  = s2.summary          || {};

  document.getElementById('res-topic').textContent = topic;

  const chips = [];
  chips.push(`<span class="tag">${(data.model || '').split('/').pop()}</span>`);
  chips.push(`<span class="tag">${allArgs.length} arguments</span>`);
  if (summary.kept_relations != null)
    chips.push(`<span class="tag">${summary.kept_relations} relations</span>`);
  if (s3)
    chips.push(`<span class="tag">Grounded: ${(s3.grounded_extension || []).length}</span>`);
  if (s4) {
    const v = (s4.verdict || 'TIE').toLowerCase();
    chips.push(`<span class="tag ${v}">${s4.verdict}</span>`);
  }
  document.getElementById('res-tags').innerHTML = chips.join('');

  renderVerdict(s4, s3);
  renderArguments(allArgs, strength);
  renderRelations(allRels, allArgs);
  renderGraph(allArgs, allRels, strength, s3);

  showState('results');
  switchTabByName('verdict');
}

/* ── Verdict tab ─────────────────────────────────────────────────────────── */
function renderVerdict(s4, s3) {
  const vw = document.getElementById('v-word');
  if (!s4) { vw.textContent = '—'; vw.className = 'verdict-word'; return; }

  const verdict = s4.verdict || 'TIE';
  vw.textContent = verdict;
  vw.className = `verdict-word ${verdict}`;

  const pct = Math.round((s4.confidence || 0) * 100);
  document.getElementById('conf-fill').style.width = `${pct}%`;
  document.getElementById('conf-num').textContent  = `${pct}%`;
  document.getElementById('v-rationale').textContent = s4.rationale || '—';

  const attacks = s4.killing_attacks || [];
  const kills = document.getElementById('v-kills');
  if (attacks.length) {
    kills.style.display = '';
    document.getElementById('v-kills-list').innerHTML =
      attacks.map(a => `<li>${escHtml(a)}</li>`).join('');
  } else {
    kills.style.display = 'none';
  }

  /* Graph verdict card */
  const gvBody = document.getElementById('gv-body');
  if (s3?.graph_verdict) {
    const gv  = s3.graph_verdict;
    const cls = (gv.winner || 'TIE').toLowerCase();
    gvBody.innerHTML = `
      <div class="gv-winner ${cls}">${escHtml(gv.winner || 'TIE')}</div>
      <div class="gv-row"><span>PRO score</span><span class="gv-val">${gv.pro_score ?? '—'}</span></div>
      <div class="gv-row"><span>CON score</span><span class="gv-val">${gv.con_score ?? '—'}</span></div>
      <div class="gv-row"><span>Basis</span><span class="gv-val">${escHtml(gv.basis ?? '—')}</span></div>
      <div class="gv-row"><span>LLM agrees</span><span class="gv-val" style="color:var(--${gv.winner === s4.verdict ? 'green' : 'con'})">${gv.winner === s4.verdict ? 'Yes ✓' : 'No ✗'}</span></div>`;
  } else {
    gvBody.textContent = 'Stage 3 not run.';
  }

  /* Semantics card */
  const semBody = document.getElementById('sem-body');
  if (s3) {
    const grounded   = s3.grounded_extension || [];
    const acceptance = s3.acceptance || {};
    const skeptical  = Object.entries(acceptance).filter(([,v]) => v.skeptical).map(([k]) => k);
    const credulous  = Object.entries(acceptance).filter(([,v]) => v.credulous && !v.skeptical).map(([k]) => k);

    const chips = (ids, cls) =>
      ids.length
        ? ids.map(id => `<span class="sem-chip ${cls}">${shortId(id)}</span>`).join('')
        : '<span class="sem-chip">none</span>';

    semBody.innerHTML = `
      <div class="sem-group">
        <div class="sem-group-label">Grounded ext. (${grounded.length})</div>
        <div class="sem-chips">${chips(grounded, 'grnd')}</div>
      </div>
      <div class="sem-group">
        <div class="sem-group-label">Skeptically accepted</div>
        <div class="sem-chips">${chips(skeptical, 'skep')}</div>
      </div>
      <div class="sem-group">
        <div class="sem-group-label">Credulous only</div>
        <div class="sem-chips">${chips(credulous, '')}</div>
      </div>
      <div class="gv-row"><span>Preferred exts</span><span class="gv-val">${s3.n_preferred ?? '—'}</span></div>
      <div class="gv-row"><span>Stable exts</span><span class="gv-val">${s3.n_stable ?? '—'}</span></div>
      <div class="gv-row"><span>Attack edges</span><span class="gv-val">${s3.n_attack_edges ?? '—'}</span></div>`;
  } else {
    semBody.textContent = 'Stage 3 not run.';
  }
}

/* ── Arguments tab ───────────────────────────────────────────────────────── */
function renderArguments(args, strength) {
  const pro = args.filter(a => a.stance === 'PRO');
  const con = args.filter(a => a.stance === 'CON');
  document.getElementById('pro-count').textContent = pro.length;
  document.getElementById('con-count').textContent = con.length;
  document.getElementById('pro-list').innerHTML = pro.map(a => argCard(a, strength, 'pro')).join('');
  document.getElementById('con-list').innerHTML = con.map(a => argCard(a, strength, 'con')).join('');
}

function argCard(arg, strength, cls) {
  const s   = strength[arg.arg_id]?.strength ?? 0.5;
  const pct = Math.round(s * 100);
  const tgt = arg.targets_arg != null
    ? `<span class="arg-rnd" style="color:var(--amber)">→${arg.targets_arg}</span>` : '';
  return `<div class="arg-card ${cls}">
    <div class="arg-meta">
      <span class="arg-id">${escHtml(arg.arg_id)}</span>
      <span class="arg-per">${escHtml(arg.persona || '')}</span>
      <span class="arg-rnd">R${arg.round ?? 1}</span>${tgt}
    </div>
    <div class="arg-txt">${escHtml(arg.text || '')}</div>
    <div class="str-bar" title="Strength ${pct}%"><div class="str-fill" style="width:${pct}%"></div></div>
  </div>`;
}

/* ── Relations tab ───────────────────────────────────────────────────────── */
function renderRelations(rels, args) {
  _allRels = rels;
  const kept = rels.filter(r => r.kept && r.label !== 'None' && r.label !== 'Neutral');
  document.getElementById('rel-lbl').textContent = `${kept.length} relations`;

  const argMap = {};
  (args || []).forEach(a => { argMap[a.arg_id] = a.text || ''; });

  document.getElementById('rel-tbody').innerHTML = kept.map(r => {
    const src = shortId(r.source_arg_id);
    const tgt = shortId(r.target_arg_id);
    return `<tr data-label="${r.label}">
      <td><span class="ref" title="${escHtml(argMap[r.source_arg_id] || '')}">${src}</span></td>
      <td><span class="lbl-badge ${r.label}">${r.label}</span></td>
      <td><span class="ref" title="${escHtml(argMap[r.target_arg_id] || '')}">${tgt}</span></td>
      <td><span class="mini-conf">${r.confidence?.toFixed(2) ?? '—'}</span></td>
      <td class="premise-td">${r.premise ? escHtml(r.premise) : ''}</td>
    </tr>`;
  }).join('');
}

function filterRels(btn) {
  document.querySelectorAll('.filter').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const f = btn.dataset.f;
  document.querySelectorAll('#rel-tbody tr').forEach(row =>
    row.classList.toggle('hidden', f !== 'all' && row.dataset.label !== f)
  );
}

/* ── Graph tab ───────────────────────────────────────────────────────────── */
function renderGraph(args, rels, strength, s3) {
  const container = document.getElementById('graph-canvas');
  if (_network) { _network.destroy(); _network = null; }

  const grounded   = new Set(s3?.grounded_extension || []);
  const acceptance = s3?.acceptance || {};

  /* Build stance lookup */
  const stanceOf = {};
  args.forEach(a => { stanceOf[a.arg_id] = a.stance; });

  /* Nodes — bipartite initial positions */
  const nodes = new vis.DataSet(args.map(arg => {
    const isPro = arg.stance === 'PRO';
    const inGnd = grounded.has(arg.arg_id);
    const s     = strength[arg.arg_id]?.strength ?? 0.5;
    const alpha = 0.45 + s * 0.55;
    return {
      id:    arg.arg_id,
      label: shortId(arg.arg_id) + '\n' + truncate(arg.text || '', 28),
      title: nodeTooltip(arg, strength[arg.arg_id], acceptance[arg.arg_id]),
      x:     isPro ? -220 : 220,
      color: {
        background: isPro ? `rgba(96,165,250,${alpha})` : `rgba(248,113,113,${alpha})`,
        border:     inGnd ? '#fbbf24' : (isPro ? '#1d4ed8' : '#991b1b'),
        highlight:  { background: isPro ? '#93c5fd' : '#fca5a5', border: '#fbbf24' },
        hover:      { background: isPro ? '#93c5fd' : '#fca5a5', border: '#fbbf24' },
      },
      borderWidth:         inGnd ? 3 : 1,
      borderWidthSelected: 4,
      font: { color: '#e2e2e2', size: 11, face: 'Inter' },
      shape: 'box',
      margin: { top: 7, bottom: 7, left: 9, right: 9 },
    };
  }));

  /* Edges — cross-stance Attacks only; all Support edges */
  const keptRels = rels.filter(r => r.kept && r.label !== 'None' && r.label !== 'Neutral');
  const graphRels = keptRels.filter(r => {
    if (r.label === 'Attack') {
      const ss = r.source_stance || stanceOf[r.source_arg_id];
      const ts = r.target_stance || stanceOf[r.target_arg_id];
      return ss && ts && ss !== ts;  // cross-stance only
    }
    return true; // Support edges always shown
  });

  const edges = new vis.DataSet(graphRels.map((r, i) => {
    const isAtk = r.label === 'Attack';
    return {
      id:     i,
      from:   r.source_arg_id,
      to:     r.target_arg_id,
      arrows: { to: { enabled: true, scaleFactor: 0.55 } },
      color:  { color: isAtk ? '#f87171' : '#4ade80', opacity: 0.8 },
      width:  Math.max(1, (r.confidence || 0.5) * 3),
      dashes: !isAtk,
      title:  `${r.label} (${r.confidence?.toFixed(2) ?? '?'})${r.premise ? '\n' + r.premise : ''}`,
      smooth: { type: 'curvedCW', roundness: 0.2 },
    };
  }));

  _network = new vis.Network(container, { nodes, edges }, {
    physics: {
      enabled:       _physicsOn,
      stabilization: { iterations: 200, updateInterval: 25 },
      barnesHut:     { gravitationalConstant: -5500, springLength: 170, springConstant: 0.04 },
    },
    layout:      { improvedLayout: true, randomSeed: 42 },
    interaction: { hover: true, tooltipDelay: 120 },
    nodes:       { shape: 'box', widthConstraint: { minimum: 55, maximum: 155 } },
    edges:       { smooth: { type: 'curvedCW', roundness: 0.2 } },
  });

  const atkEdges = graphRels.filter(r => r.label === 'Attack').length;
  const supEdges = graphRels.filter(r => r.label === 'Support').length;
  document.getElementById('leg-stats').innerHTML = `
    <div>${args.length} arguments</div>
    <div>${atkEdges} attacks &nbsp; ${supEdges} support</div>
    <div>${grounded.size} grounded</div>`;
}

function nodeTooltip(arg, si, accept) {
  const s = si?.strength;
  const lines = [
    `<strong>${arg.arg_id}</strong>`,
    `Persona: ${arg.persona || '—'}`,
    `Stance: ${arg.stance}  Round: ${arg.round ?? '—'}`,
    s != null ? `Strength: ${(s * 100).toFixed(0)}%` : '',
    '', arg.text || '',
  ];
  if (accept) {
    lines.push('');
    if (accept.grounded)       lines.push('✓ Grounded');
    if (accept.skeptical)      lines.push('✓ Skeptically accepted');
    else if (accept.credulous) lines.push('◑ Credulously accepted');
  }
  return lines.filter(l => l !== null).join('<br/>');
}

/* ── Graph controls ──────────────────────────────────────────────────────── */
function fitGraph() {
  _network?.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
}

function togglePhysics() {
  _physicsOn = !_physicsOn;
  _network?.setOptions({ physics: { enabled: _physicsOn } });
}

function toggleLabels() {
  _labelsOn = !_labelsOn;
  if (!_network) return;
  _network.body.data.nodes.get().forEach(n => {
    const line1 = n.label?.split('\n')[0] || n.id;
    _network.body.data.nodes.update({
      id: n.id,
      label: _labelsOn ? n.label : line1,
    });
  });
}

/* ── UI helpers ──────────────────────────────────────────────────────────── */
function showState(name) {
  ['empty', 'running', 'results', 'error'].forEach(s =>
    document.getElementById(`state-${s}`).classList.toggle('hidden', s !== name)
  );
}

function switchTab(btn) {
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const tab = btn.dataset.tab;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
  document.getElementById(`tab-${tab}`).classList.remove('hidden');
  if (tab === 'graph' && _network)
    setTimeout(() => _network.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } }), 50);
}

function switchTabByName(name) {
  const btn = document.querySelector(`.tab[data-tab="${name}"]`);
  if (btn) switchTab(btn);
}

function resetProgress() {
  for (let i = 1; i <= 4; i++) {
    const pi = document.getElementById(`pi-${i}`);
    if (pi) pi.classList.remove('active', 'done', 'error');
    const pm = document.getElementById(`pm-${i}`);
    if (pm) pm.textContent = '';
  }
  const box = document.getElementById('log-box');
  if (box) box.innerHTML = '';
}

function log(msg) {
  if (!msg) return;
  const box   = document.getElementById('log-box');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.textContent = `${ts()} ${msg}`;
  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;
}

function showError(msg) {
  showState('error');
  document.getElementById('error-msg').textContent = msg;
  const box   = document.getElementById('log-box');
  const entry = document.createElement('div');
  entry.className = 'log-entry err';
  entry.textContent = `${ts()} ERROR: ${msg}`;
  box?.appendChild(entry);
}

function resetApp() {
  showState('empty');
  document.getElementById('run-btn').disabled = false;
  if (_network) { _network.destroy(); _network = null; }
}

/* ── Utilities ───────────────────────────────────────────────────────────── */
function shortId(id) {
  const m = id.split('_A');
  return m.length > 1 ? 'A' + m[m.length - 1] : id;
}

function ts() {
  const d = new Date();
  return `[${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}]`;
}

function pad(n) { return String(n).padStart(2, '0'); }

function truncate(s, n) { return s.length <= n ? s : s.slice(0, n) + '…'; }

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
