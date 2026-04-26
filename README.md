# MAJ-Debate: Multi-Agent LLM Argumentation Framework for Automated Debate Judgment

**Status: Project complete — final report submitted.**

MAJ-Debate is a four-stage multi-agent pipeline for automated debate judgment, developed as the final project for the NLP course (Master's Program, Asian Institute of Technology). The pipeline combines persona-driven argument generation, pairwise attack-relation labelling, Dung Abstract Argumentation Framework graph reasoning, and an LLM judge brain to produce explainable debate verdicts.

---

## Team

| Name | Student ID |
|---|---|
| Prajwal Bhandary | st126380 |
| Saugat Shakya | st125974 |
| Prabidhi Pyakurel | st125982 |
| Rahul Shakya | st125986 |

Asian Institute of Technology (AIT), Pathum Thani, Thailand

---

## Key Results

| Configuration | DDO Acc. (%) | Logic Acc. (%) |
|---|---|---|
| **Single-LLM Baseline** | **56.4** | **78.0** |
| + CoT Baseline | 49.2 | 74.0 |
| + Direct Judge | 40.8 | 60.0 |
| + 2 Agents | 42.6 | 66.0 |
| + 6 Agents | 40.4 | 58.0 |
| + Targeted Attacks | 40.0 | 58.0 |
| + Dung Graph (no agents) | 43.8 | 62.0 |
| Full Pipeline | 41.8 | 56.0 |

DDO benchmark: n=500 crowd-voted debates. Logic benchmark: n=50 hand-crafted topics.  
The single-LLM baseline outperforms all complex configurations on DDO persuasion accuracy.  
Human evaluation on 10 curated failure topics: human majority 100%, full pipeline 0%.  
Large-scale replication (Llama-3.1-405B via NVIDIA NIM): single-LLM 100%, CoT 90% on same topics.

---

## Pipeline Architecture

```
Stage 1 — Side-Picking Agents
  6 persona-driven agents (3 Pro: Rationalist, Ethics Advocate, Futurist;
  3 Con: Skeptic, Rights Defender, Pragmatist) generate arguments over
  two rounds using BM25 retrieval from DebateSum (187k arguments).
  Reference: Debate-to-Write (Hu et al., 2025)

Stage 2 — Attack-Relation Brain
  Every ordered argument pair is classified as Attack / Support / Neutral
  (confidence threshold ≥ 0.65). Strength calibration uses "Which Side
  Are You On?" (Li et al., 2024).
  Reference: Ozaki et al. (2025)

Stage 3 — Dung Argumentation Framework Engine
  A directed attack graph is built from Stage 2 labels. Grounded, preferred,
  and stable extensions are computed. When the grounded extension is empty
  (43.6% of full-pipeline topics), the system falls back to preferred-majority.
  Reference: Dung (1995)

Stage 4 — Judge Brain (LLM-as-Judge)
  The Stage 3 verdict is passed as a hard constraint to an LLM judge that
  produces a natural-language explanation and final verdict.
```

Model: **Qwen2.5-3B-Instruct** served via vLLM on 4× NVIDIA RTX 2080 Ti (11 GB each).  
Large-scale replication: **Llama-3.1-405B-Instruct** via NVIDIA NIM API.

---

## Repository Structure

```
MAJ-Debate/
├── report/                          # Final academic report (ACL style)
│   ├── report.tex                   # Main LaTeX source
│   ├── generate_figures.py          # Generates all report figures
│   ├── check_report.py              # Pre-submission sanity checker
│   └── figures/                     # All 10 generated PNG figures
│
├── proposal/                        # Project proposal (submitted)
│   ├── MAJ_Debate_Proposal.tex
│   ├── acl.sty
│   └── references.bib
│
├── progress_report/                 # Progress report (submitted)
│   └── MAJ_Debate_Progress_Report.tex
│
├── notebooks/
│   ├── eda/                         # Exploratory data analysis
│   ├── Stage 1/                     # Side-picking agents notebook
│   ├── Stage 2/                     # Attack-relation labelling notebook
│   ├── Stage 3/                     # Argumentation framework notebook
│   ├── Stage 4/                     # Judge brain notebook
│   └── Experiments/
│       ├── Run_All_Proposal_Experiments.ipynb
│       ├── Run_Selected10_Groq_Experiments.ipynb   # 10 failure topics, Qwen-3B
│       └── Run_Selected10_Nvidia_Experiments.ipynb # 10 failure topics, Llama-405B
│
├── scripts/
│   ├── stage1_vllm.py               # Stage 1: argument generation
│   ├── stage2_vllm_shard.py         # Stage 2: pairwise relation labelling
│   ├── stage3_graph.py              # Stage 3: Dung graph + extensions
│   ├── stage4_judge.py              # Stage 4: LLM judge brain
│   ├── run_all_ablations.py         # Orchestrates 8-config ablation suite
│   ├── evaluate_ablations.py        # Computes accuracy / persuasion metrics
│   ├── inspect_ablations.py         # Cross-config diagnostic inspector
│   ├── score_form_responses.py      # Scores human evaluation Google Form CSV
│   ├── groq_selected10_notebook.py  # Shared notebook driver (Groq / NVIDIA)
│   └── generate_progress_report_figures.py
│
├── data/
│   ├── eval/
│   │   ├── ddo_sample_topics.jsonl        # 500-topic DDO benchmark
│   │   ├── logic_test_topics.jsonl        # 50-topic correctness benchmark
│   │   └── google_form/
│   │       └── form_topics.jsonl          # 10-topic human failure-case set
│   └── raw/                               # Raw corpora (gitignored if large)
│
├── outputs/
│   ├── ablations/
│   │   ├── ddo_sample/ablation_table.csv  # Main ablation results (DDO)
│   │   ├── logic_test/ablation_table.csv  # Main ablation results (Logic)
│   │   └── human_form/scorecard.md        # Human evaluation scorecard
│   ├── stage1/ … stage4/                  # Per-stage JSON outputs
│   ├── eda/figures/                       # EDA figures
│   ├── groq_selected10/                   # 10-topic Groq (Qwen-3B) run
│   ├── nvidia_selected10/                 # 10-topic NVIDIA (Llama-405B) run
│   └── progress_report_bundle.md          # Full data dump for progress report
│
├── configs/                               # Ablation and baseline manifests
├── tools/                                 # Shared utility modules
├── pyproject.toml
├── requirements.txt
└── .env                                   # API keys — NOT committed
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Or with `uv` (faster):

```bash
uv sync
```

### 2. API keys

Copy `.env.example` to `.env` and fill in your keys:

```
OPENROUTER_API_KEY=...
NVIDIA_API_KEY=...          # for NVIDIA NIM replication
GROQ_API_KEY=...            # for Groq-hosted runs
VLLM_BASE_URL=http://localhost:8000/v1   # local vLLM server
```

`.env` is gitignored. Never commit it.

### 3. vLLM server (for local Qwen-3B runs)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tensor-parallel-size 4 \
  --port 8000
```

Requires 4× GPU with ≥11 GB VRAM each.

---

## Running the Pipeline

### Full ablation suite (DDO + Logic)

```bash
python scripts/run_all_ablations.py --benchmark ddo_sample
python scripts/run_all_ablations.py --benchmark logic_test
python scripts/evaluate_ablations.py
```

### Individual stages

```bash
python scripts/stage1_vllm.py   --topic-file data/eval/ddo_sample_topics.jsonl
python scripts/stage2_vllm_shard.py
python scripts/stage3_graph.py
python scripts/stage4_judge.py  --config full
```

### Human evaluation scoring

```bash
python scripts/score_form_responses.py \
  --responses-csv outputs/MAJ-Debate_Selected_Failure_Review_Responses.csv \
  --topics data/eval/google_form/form_topics.jsonl
# Output: outputs/ablations/human_form/scorecard.md
```

### Diagnostics

```bash
python scripts/inspect_ablations.py   # cross-config duplicate check, graph usage
```

---

## Generating Report Figures

```bash
python report/generate_figures.py
# Writes 10 PNG figures to report/figures/
```

Figures generated:
1. `fig_ablation_actual.png` — dual-benchmark accuracy bar chart
2. `fig_persuasion_vs_correctness.png` — DDO vs Logic scatter per config
3. `fig_pro_bias.png` — PRO/CON/TIE verdict distribution
4. `fig_running_example.png` — 4-stage LOGIC_002 walkthrough
5. `fig_human_eval.png` — human evaluation agreement per config
6. `fig_attack_diversity.png` — attack diversity vs accuracy
7. `fig_nvidia_vs_groq.png` — Qwen-3B vs Llama-405B on 10 failure topics
8. `fig_graph_stability.png` — empty-grounded-extension rate
9. `fig_domain_breakdown.png` — per-domain DDO accuracy
10. `fig_error_taxonomy.png` — failure mode counts

---

## Compiling the Final Report

```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Requires TeX Live or MiKTeX with `acl.sty` (provided in `proposal/`).  
Pre-submission checker:

```bash
python report/check_report.py
```

---

## Ablation Configurations

| Config | Agents | Targeted Attacks | Dung Graph |
|---|---|---|---|
| Single-LLM | No | No | No |
| CoT | No (CoT prompt) | No | No |
| Direct Judge | No | No | No (strong judge) |
| 2 Agents | 2 (1 Pro, 1 Con) | No | Yes |
| 6 Agents | 6 (3 Pro, 3 Con) | No | Yes |
| Targeted Attacks | 6 | Yes | Yes |
| Dung (no agents) | No | No | Yes only |
| Full | 6 | Yes | Yes |

---

## Error Analysis Summary

| ID | Failure Mode | Affected Configs | Count |
|---|---|---|---|
| E1 | PRO-verdict bias (Support-edge surplus in Stage 2) | All complex | 472/500 PRO |
| E2 | Empty grounded extension → fallback | Full pipeline | 43.6% of topics |
| E3 | Stage 4 judge overrides graph verdict | Full, 6-agent | 3/10 failure topics |
| E4 | Cross-config rationale duplication (orchestrator bug) | 6-agents | 327 topics |

---

## Datasets Used

| Dataset | Size | Role |
|---|---|---|
| DebateSum (Roush & Balaji, 2020) | 187k argument triples | Stage 1 retrieval corpus |
| Which Side Are You On? (Li et al., 2024) | 28k examples | Stage 2 strength calibration |
| DDO / Debate.org (Durmus & Cardie, 2019) | 500-debate sample | Persuasion evaluation benchmark |
| Logic-Test Set (hand-crafted) | 50 topics | Correctness evaluation benchmark |
| Human Failure-Case Set | 10 topics, 10 annotators | Failure mode validation |
