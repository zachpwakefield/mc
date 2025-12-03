# MLP-Cox training outputs and workflow

This repository trains a Cox proportional hazards multilayer perceptron (MLP-Cox) for any TCGA cancer/modality pair and writes a fully self-describing run directory so figures, tables, and weights can be reused without re-running the model. The sections below document the workflow, model architecture, and every artifact produced during a run.

## End-to-end workflow
1. **Data loading**: `load_omics` pulls gene-expression or splicing event matrices plus optional clinical covariates. Modality-specific transforms (logit for PSI-like events, atanh for HIT scores, identity for expression) are applied before converting to Torch tensors.
2. **Train/dev split**: A stratified split (`train_val_test_split`) separates a development set (train+val) from a blind test hold-out. Low-event cancers use a larger hold-out (35%) to keep validation counts stable; others hold out 15%. Within the dev split, `fit_full` creates a 67/33 train/val partition stratified by event status and time quantiles.
3. **Feature scaling**: Non-clinical features plus `clin::age` are z-scored using means/stds fit on the dev-train subset only. These `mu`/`sigma` tensors are reused for validation, test, SHAP, and downstream plots to avoid leakage.
4. **Hyperparameter search (optional)**: Optuna TPE explores layer depth, first-layer width, dropout, learning rate, weight decay, and epochs. Up to `--max_trials` trials are stored in a per-run SQLite DB under `optuna/` so interrupted searches can resume.
5. **Model training**: The final MLP-Cox is trained on the dev split with AdamW, ReduceLROnPlateau, gradient clipping, and early stopping (patience=50) using the Cox partial log-likelihood. Best validation concordance dictates checkpoint restoration.
6. **Evaluation**: After training once, dev/test risk scores are cached and reused to compute C-index, ROC/PR AUCs, risk distributions, risk-quantile event rates, and Kaplan–Meier curves. All metrics are written as CSV plus publication-ready PNGs.
7. **Explainability**: SHAP values are generated on a fixed background of randomly sampled patients to anchor a global risk baseline; beeswarm plots and per-feature importances are saved under `shap/`.
8. **Bookkeeping**: Key run metadata, hyperparameters, scaling stats, and paths are written to `metrics.json`; an append-only `run_summary.csv` tracks all runs. Risk scores are saved both as NumPy arrays and CSV with observed times/events.

## Model architecture
- **Backbone**: `MLPCox` stacks `n_layers` blocks of Linear → ReLU → Dropout with geometric width decay from the first-layer width (never below 32 hidden units).
- **Head**: A single linear layer without bias outputs a scalar log-risk per sample. No output activation is used, matching Cox PH assumptions.
- **Initialization**: All linear weights use Kaiming normal initialization with zero bias.
- **Loss**: The negative partial log-likelihood (`cox_ph_loss`) over sorted event times; gradients are clipped to `max_norm=20` each step.

## Run directory layout
Each invocation writes a timestamped `ROOT_DIR` (or `--dir_name`) with structured subfolders:

- `models/<cancer>/<modality>/`
  - `best_model.pt`: Best-epoch weights restored after early stopping.
  - `metrics.json`: Summary of dev/test metrics (C-index, ROC/PR AUCs, risk-quantile stats, KM summaries), hyperparameters, and z-scoring stats (`mu`, `sigma`).
  - `risk_test.csv`, `risk_scores.npy`: Per-patient test risks with observed survival times and event flags (CSV) plus a NumPy dump of risks.
  - `run_summary.csv`: Append-only ledger of run timestamps, dev/test concordance, and model paths.
  - `loss_curve.png`, `cidx_curve.png`: Training/validation loss and concordance traces across epochs.
  - **Discrimination curves** (per split, prefixes `dev` / `test`):
    - `<prefix>_roc_curve.csv` / `.png`: ROC coordinates with AUC annotation and chance line.
    - `<prefix>_pr_curve.csv` / `.png`: Precision–recall coordinates (recall-sorted) with baseline prevalence overlay.
  - **Risk diagnostics** (per split):
    - `<prefix>_risk_hist.png`: Overlaid risk histograms for events vs censored patients.
    - `<prefix>_risk_quantiles.csv` / `.png`: Counts, event rates, and bar plot across equal-frequency risk bins.
  - **Survival calibration** (per split):
    - `<prefix>_km_q#.csv`: Kaplan–Meier survival tables per risk quantile.
    - `<prefix>_km_summary.csv`: Aggregated survival at key time points per quantile.
    - `<prefix>_km_plot.png`: KM curves with confidence bands for all quantiles.
- `logs/`: Optuna best-trial JSON (`<study>_best.json`) plus any auxiliary logs.
- `optuna/`: SQLite database backing the hyperparameter search (`optuna_<cancer>_<modality>.db`).
- `shap/<cancer>/<modality>/`
  - `shap_values.npy`: SHAP matrix for the first `nsamples` examples (default 200) using a fixed background baseline (default 100 patients).
  - `summary_beeswarm.png`: Beeswarm of top features by SHAP magnitude.
  - `shap_mean_abs.csv`: Mean absolute SHAP values sorted descending for feature ranking.

## Methods detail (suitable for a paper)
- **Data preprocessing**: Omics inputs are modality-transformed (logit/atanh/identity) and concatenated with optional clinical variables. Non-clinical features and age are standardized using dev-train statistics only to prevent leakage. Clinical categorical variables remain untouched.
- **Training regime**: Models are optimized with AdamW and ReduceLROnPlateau on validation loss, early-stopped on validation C-index. Stratified splits maintain balanced event/censor composition across folds. Gradient clipping stabilizes optimization.
- **Hyperparameter tuning**: TPE search samples layer depth (2–8), first-layer width (4096→128 options), dropout (0.1–0.8), learning rate (1e-5–1e-3 log-scale), weight decay (1e-6–1e-3 log-scale), and epochs (100–1000). Completed trials are tracked to avoid duplication on restart.
- **Evaluation metrics**: Concordance index (dev/test) measures ranking quality. Binary discrimination uses ROC AUC and average precision on event labels with full curve exports. Risk diagnostics summarize calibration through KM curves by risk quantile and event-rate bar plots. Risk histograms stratify event vs censor distributions. All plots are publication-formatted (gridlines, legends, axis labels, and dpi=200).
- **Explainability**: DeepExplainer computes SHAP values relative to a fixed global mean-risk baseline. Background patients are randomly sampled without replacement; the explained foreground defaults to the first 200 samples. Outputs include the SHAP matrix, beeswarm visualization, and per-feature importance table to support narrative feature attributions.
- **Reproducibility**: Random seeds are fixed (`SEED=100`) for Torch, NumPy, and Python’s RNG. Each run records the exact hyperparameters, scaling stats, and file paths in `metrics.json`; run-level CSV ledgers allow audit across experiments. The Optuna database enables deterministic resumption of searches.

## Practical usage
Run a job with, for example:

```bash
python mlp_cox_job_11_25.py --cancer BRCA --modality gex --max_trials 20 --with_clin --transform_data --gpus 1
```

The command will create a timestamped run directory under the configured base path, populate it with all artifacts listed above, and produce ready-to-use figures and tables for reporting without needing to regenerate metrics.

### Comparing multiple runs
Use `compare_runs.py` to summarise and plot side-by-side performance for several completed runs. Example:

```bash
python compare_runs.py \
  --run-dirs \
    /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_mad10000 \
    /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_unique10 \
    /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/model_outputs_12_1_nonzero10 \
  --run-labels mad10000 unique10 nonzero10 \
  --plot-dir ./plots/comparisons
```

Outputs include overall C-index distributions, per-cancer boxplots (one box per run per cancer), and per-cancer win counts to make it clear which configuration performs best overall and by cancer type.
