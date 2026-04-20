# MAJ-Debate

MAJ-Debate is a research codebase for automated debate judgment with a four-stage pipeline:

1. Stage 1: multi-agent Pro/Con argument generation
2. Stage 2: pairwise relation labeling between arguments
3. Stage 3: Dung-style argumentation graph construction and winner inference
4. Stage 4: verdict explanation and judgment

The repository also contains:
- dataset preparation and EDA notebooks
- ablation and benchmark evaluation scripts
- LaTeX sources for the proposal and progress report

## Current Status

The project is no longer in EDA-only mode. The repo currently includes:

- real DDO-based evaluation data preparation
- a cleaned `500`-topic DDO benchmark split
- a `50`-topic logic/correctness benchmark split
- Stage 1 to Stage 4 notebooks and scripts
- ablation outputs for the main eight experiment configurations
- generated progress-report figures and compiled report/proposal PDFs

The current experimental story in the repo is:
- the single-model baseline is strongest on both DDO persuasion agreement and the logic-test correctness benchmark
- the full multi-stage pipeline underperforms that baseline at the current model scale
- Stage 3 shows a strong PRO-side skew on DDO, which is one of the main error signals documented in the report

## Repository Layout

- `notebooks/eda/`
  - exploratory data analysis and dataset inspection
- `notebooks/Stage 1/`
  - side-picking multi-agent generation notebook
- `notebooks/Stage 2/`
  - attack/relation labeling notebook
- `notebooks/Stage 3/`
  - argumentation framework notebook
- `notebooks/Stage 4/`
  - judge/explanation notebook
- `notebooks/Experiments/`
  - experiment orchestration and human-eval analysis notebooks
- `scripts/`
  - runnable Python utilities for stages, ablations, scoring, diagnostics, and report figure generation
- `data/eval/`
  - benchmark topic files and human-eval inputs
- `configs/`
  - ablation and baseline manifests
- `outputs/`
  - generated stage artifacts, ablation tables, logs, and experiment results
- `proposal/`
  - main proposal LaTeX source and compiled PDF
- `progress_report/`
  - progress report LaTeX source, generated figures, and compiled PDF

## Main Data Artifacts

- `data/eval/ddo_sample_topics.jsonl`
  - cleaned `500`-topic DDO benchmark split
- `data/eval/human_eval_topics.jsonl`
  - human/logic evaluation topic file
- `outputs/stage1/`
  - generated argument pools
- `outputs/stage2/`
  - relation labels and Stage 2 summaries
- `outputs/stage3/`
  - graph outputs and graph-level verdicts
- `outputs/stage4/`
  - final judgments for each configuration
- `outputs/ablations/`
  - final ablation tables used by the report

## Recommended Workflow

### 1. Environment

Create a local virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Secrets Locally

Use `.env` for local API keys and endpoint settings. Do not commit `.env`.

Typical local settings include:
- `OPENROUTER_API_KEY`
- model names and endpoint URLs
- stage-specific knobs such as topic limits and resume flags

### 3. Run the Pipeline

You can run either the notebooks or the scripts.

Notebook-first path:
- `notebooks/Stage 1/Stage1_SidePickingAgents.ipynb`
- `notebooks/Stage 2/Stage2_AttackRelationBrain.ipynb`
- `notebooks/Stage 3/Stage3_ArgumentationFramework.ipynb`
- `notebooks/Stage 4/Stage4_JudgeBrain.ipynb`
- `notebooks/Experiments/Run_All_Proposal_Experiments.ipynb`

Script-first path:
- `scripts/stage1_vllm.py`
- `scripts/stage2_vllm_shard.py`
- `scripts/stage3_graph.py`
- `scripts/stage4_judge.py`
- `scripts/run_all_ablations.py`
- `scripts/evaluate_ablations.py`
- `scripts/inspect_ablations.py`

### 4. Generate Report Figures

The report figure generator uses the saved experiment outputs:

```powershell
.\.venv\Scripts\python.exe scripts\generate_progress_report_figures.py
```

### 5. Compile LaTeX Reports

Proposal:

```powershell
cd proposal
pdflatex -interaction=nonstopmode MAJ_Debate_Proposal.tex
bibtex MAJ_Debate_Proposal
pdflatex -interaction=nonstopmode MAJ_Debate_Proposal.tex
pdflatex -interaction=nonstopmode MAJ_Debate_Proposal.tex
```

Progress report:

```powershell
cd progress_report
pdflatex -interaction=nonstopmode MAJ_Debate_Progress_Report.tex
bibtex MAJ_Debate_Progress_Report
pdflatex -interaction=nonstopmode MAJ_Debate_Progress_Report.tex
pdflatex -interaction=nonstopmode MAJ_Debate_Progress_Report.tex
```

## Current Experiment Outputs

The repo currently contains saved ablation summaries for:
- `ddo_sample`
- `logic_test`

Key result tables live in:
- `outputs/ablations/ddo_sample/ablation_table.json`
- `outputs/ablations/logic_test/ablation_table.json`

The current progress report and proposal PDFs live in:
- `progress_report/MAJ_Debate_Progress_Report.pdf`
- `proposal/MAJ_Debate_Proposal.pdf`

## Important Notes

- Some outputs in `outputs/experiments/` are older orchestration manifests and may not reflect the newest ablation runs. Prefer `outputs/ablations/` as the authoritative summary layer.
- The current "full" ablation label in some saved artifacts historically referred to `6 agents + graph`, while the targeted-attack variant is stored separately. Use the latest report wording rather than older manifest labels when writing up conclusions.
- Notebook checkpointing and partial stage outputs are expected during long runs; only finalized ablation tables should be treated as report-ready.

## Team

- Prajwal Bhandary
- Saugat Shakya
- Prabidhi Pyakurel
- Rahul Shakya

Asian Institute of Technology (AIT)  
NLP Course, Master's Program
