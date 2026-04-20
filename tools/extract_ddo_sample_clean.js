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
        if (ch === '{') started = true;
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

      if (expectingValue && depth === 0 && /\s/.test(ch)) continue;

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
        if (ch === '"') inString = false;
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
  if (['pro', 'for', 'affirmative', 'yes', '1'].includes(text)) return 'PRO';
  if (['con', 'against', 'negative', 'no', '2'].includes(text)) return 'CON';
  return null;
}

function inferWinner(record) {
  const p1 = Number(record.participant_1_points);
  const p2 = Number(record.participant_2_points);
  if (!Number.isFinite(p1) || !Number.isFinite(p2) || p1 === p2) return null;
  const winningPosition = p1 > p2 ? record.participant_1_position : record.participant_2_position;
  return normalizeWinner(winningPosition);
}

function cleanText(text) {
  return String(text || '')
    .replace(/^[.\-"\s]+/, '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([?!.,;:])/g, '$1')
    .trim();
}

function inferDomain(record) {
  return cleanText(record.category || record.domain || 'unknown');
}

function isUsableRecord(record) {
  const rounds = Number(record.number_of_rounds || 0);
  const p1 = Number(record.participant_1_points);
  const p2 = Number(record.participant_2_points);
  if (rounds < 3) return false;
  if (!Number.isFinite(p1) || !Number.isFinite(p2) || p1 === p2) return false;
  return true;
}

const EXCLUDED_CATEGORIES = new Set([
  'Arts',
  'Cars',
  'Entertainment',
  'Fashion',
  'Funny',
  'Games',
  'Miscellaneous',
  'Music',
  'Movies',
  'News',
  'People',
  'Places-Travel',
  'Religion',
  'Sports'
  , 'TV'
]);

const SENSITIVE_PATTERNS = [
  /\babort/i,
  /\brape\b/i,
  /\bsuicide\b/i,
  /\bself[- ]?harm\b/i,
  /\bdomestic violence\b/i,
  /\bviolence\b/i,
  /\bdeadly force\b/i,
  /\bkill(s|er|ers|ing|ings)?\b/i,
  /\btargeted killings?\b/i,
  /\bvictim(s)?\b/i,
  /\bpedoph/i,
  /\bincest\b/i,
  /\bporn/i,
  /\bpenis\b/i,
  /\bsex\b/i,
  /\bsexual\b/i,
  /\bcontracept/i,
  /\balcohol\b/i,
  /\bcannabis\b/i,
  /\bmarijuana\b/i,
  /\bgay marriage\b/i,
  /\bgay\b/i,
  /\bgays\b/i,
  /\bhomosexual\b/i,
  /\bhomosexuality\b/i,
  /\bhomosexuals\b/i,
  /\bbisexual\b/i,
  /\bheterosexuals?\b/i,
  /\btrans(gender)?\b/i,
  /\brace\b/i,
  /\bracist\b/i,
  /\bslavery\b/i,
  /\bholocaust\b/i,
  /\bterror/i,
  /\bmuslim\b/i,
  /\bmuslims\b/i,
  /\bchristian\b/i,
  /\bchristians\b/i,
  /\bislam\b/i,
  /\batheis/i,
  /\bgod\b/i,
  /\bcreator\b/i,
  /\bnazi\b/i,
  /\bhitler\b/i,
  /\bgun control\b/i,
  /\bdeath penalty\b/i,
  /\beuthanasia\b/i,
  /\bdrug/i
];

const LOW_QUALITY_PATTERNS = [
  /\bdebate me\b/i,
  /\bdebate challenge\b/i,
  /\bwho would win\b/i,
  /\bbest artist\b/i,
  /\brhymes with\b/i,
  /\byou('| a)re going down\b/i,
  /\bi like\b/i,
  /\bhotter than\b/i,
  /\bcuter than\b/i,
  /\bprove me wrong\b/i,
  /\bfunny word\b/i,
  /\bawesome\b/i,
  /\bchange my view\b/i,
  /\byes or no\b/i,
  /\bplease click\b/i,
  /\bchange my mind\b/i,
  /\bvs\.?\b/i,
  /\bmy opponent\b/i
];

function seemsSeriousTopic(topicText, domain) {
  if (!topicText) return false;
  if (EXCLUDED_CATEGORIES.has(domain)) return false;
  const text = cleanText(topicText);
  if (text.length < 18 || text.length > 180) return false;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 4) return false;
  if ((text.match(/[!?]/g) || []).length > 2) return false;
  if (/^[^A-Za-z0-9]+/.test(text)) return false;
  if (/^[^A-Za-z0-9]+$/.test(text)) return false;
  if (/[A-Z]{6,}/.test(text) && text === text.toUpperCase()) return false;
  if (LOW_QUALITY_PATTERNS.some((re) => re.test(text))) return false;
  if (SENSITIVE_PATTERNS.some((re) => re.test(text))) return false;
  return true;
}

function qualityScore(topicText) {
  const text = cleanText(topicText);
  let score = 0;
  const words = text.split(/\s+/).filter(Boolean);

  if (/^[A-Za-z]/.test(text)) score += 4;
  if (words.length >= 6 && words.length <= 18) score += 4;
  if (words.length >= 4 && words.length <= 24) score += 2;

  if (/\b(should|should be|ought|ought to|is|are|can|cannot|will|would|must|better|worse|more|less)\b/i.test(text)) {
    score += 6;
  }
  if (/\b(government|education|school|tax|climate|economy|policy|internet|technology|science|health|trade|democracy|market|rights)\b/i.test(text)) {
    score += 4;
  }

  if (/[?]/.test(text)) score -= 6;
  if (/["'|/#()]/.test(text)) score -= 4;
  if (/^\d/.test(text)) score -= 3;
  if (/\b(ddo|debate|challenge|round|prove|awesome|funny|word|furry|hex|kosher)\b/i.test(text)) score -= 8;
  if (/\b(genocide|serial|military intelligence|mental disease)\b/i.test(text)) score -= 8;
  if (/[=:]/.test(text)) score -= 2;
  if ((text.match(/[.,!?]/g) || []).length > 2) score -= 2;

  return score;
}

function buildSample(records, targetSize) {
  const byCategory = new Map();
  for (const rec of records) {
    if (!byCategory.has(rec.domain)) byCategory.set(rec.domain, []);
    byCategory.get(rec.domain).push(rec);
  }
  for (const list of byCategory.values()) {
    list.sort((a, b) => {
      const scoreDiff = qualityScore(b.topic_text) - qualityScore(a.topic_text);
      if (scoreDiff !== 0) return scoreDiff;
      return a.topic_text.localeCompare(b.topic_text);
    });
  }
  const categories = [...byCategory.keys()].sort();
  const selected = [];
  const used = new Set();
  let progressed = true;
  while (selected.length < targetSize && progressed) {
    progressed = false;
    for (const category of categories) {
      const list = byCategory.get(category);
      while (list.length > 0 && used.has(list[0].source_ref)) list.shift();
      if (list.length === 0) continue;
      const next = list.shift();
      if (used.has(next.source_ref)) continue;
      selected.push(next);
      used.add(next.source_ref);
      progressed = true;
      if (selected.length >= targetSize) break;
    }
  }
  return selected;
}

function runExtract(inputPath, outputPath, targetSize) {
  let total = 0;
  let usable = 0;
  let cleanCandidates = 0;
  const candidates = [];

  streamTopLevelObjectEntries(
    inputPath,
    (key, raw) => {
      total += 1;
      const record = JSON.parse(raw);
      if (!isUsableRecord(record)) return;
      usable += 1;
      const topicText = cleanText(record.title || record.topic || record.motion || record.resolution);
      const domain = inferDomain(record);
      const benchmarkLabel = inferWinner(record);
      if (!benchmarkLabel || !seemsSeriousTopic(topicText, domain)) return;
      cleanCandidates += 1;
      candidates.push({
        topic_id: `DDO_${String(total).padStart(5, '0')}`,
        topic_text: topicText,
        domain,
        benchmark_label: benchmarkLabel,
        source_dataset: 'DDO',
        source_ref: key
      });
    },
    () => {
      const sample = buildSample(candidates, targetSize);
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, sample.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
      const domainCounts = {};
      for (const row of sample) domainCounts[row.domain] = (domainCounts[row.domain] || 0) + 1;
      const summary = {
        total_entries_seen: total,
        usable_entries: usable,
        clean_candidates: cleanCandidates,
        sample_size: sample.length,
        domains: Object.keys(domainCounts).length,
        pro_count: sample.filter((x) => x.benchmark_label === 'PRO').length,
        con_count: sample.filter((x) => x.benchmark_label === 'CON').length,
        top_domains: Object.entries(domainCounts).sort((a, b) => b[1] - a[1]).slice(0, 15)
      };
      console.log(JSON.stringify(summary, null, 2));
    }
  );
}

const args = parseArgs(process.argv);
if (!args.input || !args.output) {
  throw new Error('Usage: node tools/extract_ddo_sample_clean.js --input <debates.json> --output <jsonl> [--size 500]');
}

runExtract(args.input, args.output, Number(args.size || 500));
