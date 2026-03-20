# MAJ-Debate

MAJ-Debate is a research project on explainable automated debate judgment using a multi-agent argumentation pipeline.

This repository currently contains **Step 1 work**: exploratory data analysis (EDA), preprocessing decisions, and experiment-preparation artifacts.

## Current Status

The repository is in the **EDA and preparation phase**.

Completed:
- Dataset-level EDA design and reproducible notebook workflow
- Topic, stance, and token-length diagnostics
- Complexity forecasting for pairwise relation labeling
- Exportable EDA statistics for proposal alignment

Planned next:
- Controlled argument generation experiments
- Pairwise relation-label quality experiments
- Graph-based extension and end-to-end judgment experiments

## Research Focus

The broader MAJ-Debate direction investigates:
- Multi-agent argument generation (Pro/Con perspectives)
- Pairwise argument relation analysis
- Graph-grounded reasoning for explainable verdicts

At this stage, this repository is intentionally focused on **data understanding and methodological readiness**.

## Repository Structure

- `notebooks/0_MAJ_Debate_EDA_Pipeline.ipynb`: Main EDA notebook
- `notebooks/outputs/`: Generated EDA artifacts (figures + JSON outputs)
- `proposal/`: Main proposal sources in LaTeX format (`MAJ_Debate_Proposal.tex`, bibliography, style files)
- `requirements.txt`: Python dependencies
- `.env`: Local environment variables (not required for current EDA-only workflow)

## Quick Start

### 1. Create and activate a Python environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Launch Jupyter and run the notebook

```powershell
jupyter notebook
```

Open:
- `notebooks/0_MAJ_Debate_EDA_Pipeline.ipynb`

Run cells from top to bottom.

## Outputs

After execution, the notebook writes outputs under:
- `notebooks/outputs/figures/` (plots)
- `notebooks/outputs/eda_stats.json` (key metrics for reporting)

## Reproducibility Notes

- The current notebook workflow is offline and does not require API keys.
- Random seeds are fixed in the notebook for stable synthetic simulations where applicable.
- If preprocessing thresholds are changed, rerun the notebook and regenerate `eda_stats.json` before reporting results.

## Data and Evaluation Notes

The current EDA phase references:
- DebateSum
- Which Side Are You On?
- A planned internal human-evaluated topic set

Please follow each dataset's license and citation requirements when sharing derived artifacts.

## Team

- Prajwal Bhandary
- Saugat Shakya
- Prabidhi Pyakurel
- Rahul Shakya

Asian Institute of Technology (AIT), NLP Course (Master's Program)

## Citation

If you use this repository in academic work, cite the MAJ-Debate project materials and the underlying benchmark datasets used in the EDA notebook and LaTeX proposal sources.
