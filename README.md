# Antithrombotic resumption after traumatic intracranial haemorrhage — analysis code

This repository contains the analysis code and aggregate result tables for the manuscript:

> **Antithrombotic resumption after traumatic intracranial haemorrhage: timing, thrombosis and mortality across five critical care databases.**
> Zhang X. *Neurocritical Care*, 2026.

## Contents

- `scripts/` — extraction scripts for the five source databases, cohort construction, Firth logistic regression, propensity-score matching, inverse-probability weighting, landmark and competing-risk analyses, restricted cubic splines, and figure code.
- `results/` — aggregate result tables and JSON outputs underlying Tables 1–2 and Figures 3–5.
- `LICENSE` — MIT license.
- `CITATION.cff` — citation metadata for this archived version.

## Data

Individual-level patient data are **not** included. MIMIC-IV, eICU-CRD, NWICU, SICdb and AmsterdamUMCdb remain governed by their data-use agreements and must be obtained from the respective data owners. The scripts contain absolute path placeholders (for example `F:\...`) that must be changed to the local licensed copy before execution.

## Environment

- Python 3.12 (pandas, numpy, statsmodels, lifelines, scipy, duckdb)
- R 4.6.1 (rms, MatchIt, survival, cmprsk, ggplot2, patchwork)

## Order of execution

1. `scripts/extract_*.py` — extract the source-specific cohorts.
2. `scripts/analyze_t3_v3_pool.py` — harmonise the sources into the pooled analysis table.
3. `scripts/analyze_review_response_v2.py` and `analyze_review_response_v3.py` — reversal composition, early-exit, subgroup and bleeding-proxy analyses.
4. `scripts/analyze_t3_competing.py` — cause-specific and Fine–Gray models in the MIMIC-IV subset.
5. `scripts/sensitivity_covariates.py` — covariate-specification sensitivity models.
6. `scripts/mimic_post_restart_hb.py` — post-restart haemoglobin proxy.
7. `scripts/compute_tables.py`, `make_figures.py`, `make_figure4_forest.R`, `analyze_v3_enhanced.R`, `make_enhanced_figures.R` — tables and figures.

## Citation

If you use this code, please cite:

> Zhang X. *Neurocritical Care*, 2026. DOI: [to be added upon publication].

## Version

- v1.0.0 — initial release accompanying the revised manuscript (November 2026).
