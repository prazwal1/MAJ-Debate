const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function streamTopLevelObjectEntries(filePath, onEntry, onDone) {
  const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
  let started = false;
  let readingKey = false;
  let inString = false;
  let escape = false;
  let depth = 0;
  let keyBuffer = '';
  let valueBuffer = '';
  let currentKey = null;
  let expectingValue = false;

  function flushEntry() {
    const raw = valueBuffer.trim();
    if (!currentKey || !raw) return;
    onEntry(currentKey, raw);
    currentKey = null;
    valueBuffer = '';
    expectingValue = false;
  }

  stream.on('data', (chunk) => {
    for (let i = 0; i < chunk.length; i += 1) {
      const ch = chunk[i];

      if (!started) {
        if (ch === '{') {
          started = true;
        }
        continue;
      }

      if (readingKey) {
        if (escape) {
          keyBuffer += ch;
          escape = false;
          continue;
        }
        if (ch === '\\') {
          keyBuffer += ch;
          escape = true;
          continue;
        }
        if (ch === '"') {
          readingKey = false;
          currentKey = JSON.parse(`"${keyBuffer}"`);
          keyBuffer = '';
          continue;
        }
        keyBuffer += ch;
        continue;
      }

      if (!expectingValue && depth === 0) {
        if (ch === '"') {
          readingKey = true;
          keyBuffer = '';
          continue;
        }
        if (ch === ':') {
          expectingValue = true;
          continue;
        }
        if (ch === '}') {
          onDone();
          stream.destroy();
          return;
        }
        continue;
      }

      if (expectingValue && depth === 0 && /\s/.test(ch)) {
        continue;
      }

      valueBuffer += ch;

      if (inString) {
        if (escape) {
          escape = false;
          continue;
        }
        if (ch === '\\') {
          escape = true;
          continue;
        }
        if (ch === '"') {
          inString = false;
        }
        continue;
      }

      if (ch === '"') {
        inString = true;
        continue;
      }

      if (ch === '{' || ch === '[') {
        depth += 1;
        continue;
      }

      if (ch === '}' || ch === ']') {
        depth -= 1;
        continue;
      }

      if (ch === ',' && depth === 0) {
        valueBuffer = valueBuffer.slice(0, -1);
        flushEntry();
      }
    }
  });

  stream.on('end', () => onDone());
}

function normalizeWinner(value) {
  if (value == null) return null;
  const text = String(value).trim().toLowerCase();
  if (!text) return null;
  if (['pro', 'for', 'affirmative', 'yes', '1'].includes(text)) return 'PRO';
  if (['con', 'against', 'negative', 'no', '2'].includes(text)) return 'CON';
  return null;
}

function inferWinner(record) {
  if (record.participant_1_points != null || record.participant_2_points != null) {
    const p1 = Number(record.participant_1_points || 0);
    const p2 = Number(record.participant_2_points || 0);
    if (Number.isFinite(p1) && Number.isFinite(p2) && p1 !== p2) {
      const winningPosition = p1 > p2 ? record.participant_1_position : record.participant_2_position;
      const label = normalizeWinner(winningPosition);
      if (label) return label;
    }
  }

  const directFields = [
    'winner',
    'winning_side',
    'winningSide',
    'winner_side',
    'decision',
    'result'
  ];
  for (const field of directFields) {
    const label = normalizeWinner(record[field]);
    if (label) return label;
  }

  if (Array.isArray(record.votes)) {
    const counts = { PRO: 0, CON: 0 };
    for (const vote of record.votes) {
      const label = normalizeWinner(vote && (vote.vote || vote.winner || vote.side || vote.position));
      if (label) counts[label] += 1;
    }
    if (counts.PRO > counts.CON) return 'PRO';
    if (counts.CON > counts.PRO) return 'CON';
  }

  if (record.pro_votes != null || record.con_votes != null) {
    const pro = Number(record.pro_votes || 0);
    const con = Number(record.con_votes || 0);
    if (pro > con) return 'PRO';
    if (con > pro) return 'CON';
  }

  return null;
}

function inferTopicText(record) {
  return record.title || record.topic || record.motion || record.resolution || null;
}

function inferDomain(record) {
  return record.category || record.domain || 'unknown';
}

function buildSample(records, targetSize) {
  const byCategory = new Map();
  for (const rec of records) {
    if (!byCategory.has(rec.domain)) byCategory.set(rec.domain, []);
    byCategory.get(rec.domain).push(rec);
  }

  const categories = [...byCategory.keys()].sort();
  const selected = [];
  const used = new Set();
  let madeProgress = true;

  while (selected.length < targetSize && madeProgress) {
    madeProgress = false;
    for (const category of categories) {
      const list = byCategory.get(category);
      while (list.length > 0 && used.has(list[0].topic_id)) {
        list.shift();
      }
      if (list.length === 0) continue;
      const next = list.shift();
      if (!used.has(next.topic_id)) {
        selected.push(next);
        used.add(next.topic_id);
        madeProgress = true;
        if (selected.length >= targetSize) break;
      }
    }
  }

  return selected;
}

function isUsableRecord(record) {
  const rounds = Number(record.number_of_rounds || 0);
  if (rounds < 3) return false;
  const p1 = Number(record.participant_1_points || 0);
  const p2 = Number(record.participant_2_points || 0);
  if (!Number.isFinite(p1) || !Number.isFinite(p2)) return false;
  if (p1 === p2) return false;
  return true;
}

function runInspect(inputPath, limit) {
  let seen = 0;
  streamTopLevelObjectEntries(
    inputPath,
    (key, raw) => {
      if (seen >= limit) return;
      const record = JSON.parse(raw);
      const snapshot = {
        key,
        fields: Object.keys(record),
        title: record.title,
        category: record.category,
        inferred_winner: inferWinner(record),
        debate_status: record.debate_status,
        number_of_rounds: record.number_of_rounds,
        participant_1_position: record.participant_1_position,
        participant_1_points: record.participant_1_points,
        participant_2_position: record.participant_2_position,
        participant_2_points: record.participant_2_points,
        pro_votes: record.pro_votes,
        con_votes: record.con_votes,
        votes_sample: Array.isArray(record.votes) ? record.votes.slice(0, 2) : record.votes
      };
      console.log(JSON.stringify(snapshot, null, 2));
      seen += 1;
    },
    () => {}
  );
}

function runExtract(inputPath, outputPath, targetSize) {
  const candidates = [];
  let total = 0;
  let kept = 0;

  streamTopLevelObjectEntries(
    inputPath,
    (key, raw) => {
      total += 1;
      const record = JSON.parse(raw);
      const topicText = inferTopicText(record);
      const benchmarkLabel = inferWinner(record);
      const domain = inferDomain(record);
      if (!topicText || !benchmarkLabel || !isUsableRecord(record)) return;
      const normalized = {
        topic_id: `DDO_${String(total).padStart(5, '0')}`,
        topic_text: topicText,
        domain,
        benchmark_label: benchmarkLabel,
        source_dataset: 'DDO',
        source_ref: key
      };
      candidates.push(normalized);
      kept += 1;
    },
    () => {
      const sample = buildSample(candidates, targetSize);
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, sample.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
      const summary = {
        total_entries_seen: total,
        usable_entries: kept,
        sample_size: sample.length,
        categories: [...new Set(sample.map((x) => x.domain))].length,
        pro_count: sample.filter((x) => x.benchmark_label === 'PRO').length,
        con_count: sample.filter((x) => x.benchmark_label === 'CON').length
      };
      console.log(JSON.stringify(summary, null, 2));
    }
  );
}

const args = parseArgs(process.argv);
const mode = args.mode || 'inspect';
const input = args.input;
if (!input) {
  throw new Error('Missing --input path');
}

if (mode === 'inspect') {
  runInspect(input, Number(args.limit || 3));
} else if (mode === 'extract') {
  if (!args.output) throw new Error('Missing --output path');
  runExtract(input, args.output, Number(args.size || 500));
} else {
  throw new Error(`Unknown mode: ${mode}`);
}
