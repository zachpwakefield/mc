#!/usr/bin/env python

"""
Train an MLP-Cox model on one (cancer, modality) pair.

Usage
-----
python mlp_cox_job.py --cancer BRCA --modality AFE \
                      --max_trials 20 --gpus 1
Valid --modality:  gex | afe | ale | hit | mxe | se | ri | a5ss | a3ss
"""

import os, sys, json, math, time, copy, random, argparse
from typing import Dict, List, Tuple
import datetime, csv
import numpy as np
import numpy
import pandas as pd
import torch, torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
import optuna, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn.functional import logsigmoid
from sklearn.model_selection import train_test_split
import pycox
from filelock import FileLock
from optuna.trial import TrialState


from lifelines import KaplanMeierFitter
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PRELOADED_DATA = None
# ------------------------------  constants / paths
EVENT_KEYS = ["afe","ale","hit","mxe","se","ri","a5ss","a3ss"]
GEX_CSV   = "/projectnb2/evolution/zwakefield/tcga/cancer_learning/data/small_harmonized/gex.csv"

ARNAP_CSV  = {k: f"/projectnb2/evolution/zwakefield/tcga/cancer_learning/data/small_harmonized/{k}.csv"
              for k in EVENT_KEYS}
CLIN_CSV   = "/projectnb2/evolution/zwakefield/tcga/cancer_learning/clinical/clinical_data.csv"

SEED = 100
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

base_root = "/projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_fs_layout(root: str,
                   cancer: str,
                   modality: str) -> Dict[str, str]:
    paths = {
        "root"      : root,
        "model_dir" : f"{root}/models/{cancer}/{modality}/",
        "log_dir"   : f"{root}/logs/",
        "shap_dir"  : f"{root}/shap/{cancer}/{modality}",
    }

    for p in set(map(os.path.dirname, paths.values())):
        if p:
            os.makedirs(p, exist_ok=True)

    return paths

def get_non_clin_idx(feat_names: List[str]) -> torch.Tensor:
    """
    Indices of features to z-score:
      - all non-clinical features (no 'clin::' prefix)
      - plus clin::age (continuous)
    """
    idx = []
    for i, n in enumerate(feat_names):
        if n == "clin::age":
            idx.append(i)
        elif not n.startswith("clin::"):
            idx.append(i)
    if not idx:
        raise RuntimeError("get_non_clin_idx: no non-clinical features found.")
    return torch.tensor(idx, dtype=torch.long)


def fit_zscore_subset(x: torch.Tensor, idx: torch.Tensor):
    """
    Fit mu/sigma only on the selected columns (idx).
    x is (N, D), idx is 1-D LongTensor of column indices.
    """
    x_sub = x[:, idx]
    mu    = x_sub.mean(dim=0, keepdim=True)
    sigma = x_sub.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mu, sigma


def apply_zscore_inplace(x: torch.Tensor, idx: torch.Tensor, mu, sigma):
    """
    In-place z-score on x[:, idx] using provided mu, sigma.
    """
    x[:, idx] = (x[:, idx] - mu) / sigma


def _binary_ranking_metrics(y_true: np.ndarray,
                            scores: np.ndarray,
                            prefix: str,
                            out_dir: str) -> Dict[str, float]:
    """
    Compute ROC AUC / PR AUC and write full curves to CSV.
    Catches degenerate label sets (all 0 or all 1) and returns NaN metrics.
    """
    metrics = {}

    # ROC -------------------------------------------------------------
    try:
        auc = roc_auc_score(y_true, scores)
        fpr, tpr, thr = roc_curve(y_true, scores)
        pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thr,
        }).to_csv(f"{out_dir}/{prefix}_roc_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", lw=1, label="Chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{prefix.upper()} ROC")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{prefix}_roc_curve.png", dpi=200)
        plt.close(fig)
    except ValueError:
        auc = float("nan")
    metrics[f"{prefix}_roc_auc"] = float(auc)

    # PR --------------------------------------------------------------
    try:
        ap = average_precision_score(y_true, scores)
        pr_p, pr_r, pr_thr = precision_recall_curve(y_true, scores)
        # precision_recall_curve returns thresholds of len(n_points - 1)
        thr_pad = np.concatenate([pr_thr, [np.nan]])
        pd.DataFrame({
            "precision": pr_p,
            "recall": pr_r,
            "threshold": thr_pad,
        }).to_csv(f"{out_dir}/{prefix}_pr_curve.csv", index=False)

        base_rate = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.step(pr_r, pr_p, where="post", color="#d62728", lw=2, label=f"AP = {ap:.3f}")
        ax.hlines(base_rate, 0, 1, colors="#999999", linestyles="--", lw=1, label=f"Baseline = {base_rate:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{prefix.upper()} Precision–Recall")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{prefix}_pr_curve.png", dpi=200)
        plt.close(fig)
    except ValueError:
        ap = float("nan")
    metrics[f"{prefix}_avg_precision"] = float(ap)

    return metrics


def _km_by_risk_quantile(risk: np.ndarray,
                         t: np.ndarray,
                         e: np.ndarray,
                         prefix: str,
                         out_dir: str,
                         q: int = 4) -> pd.DataFrame:
    """
    Save per-quantile Kaplan–Meier survival tables + summary stats.
    Falls back to equal-width bins if q-quantiles are not unique.
    """
    risk_series = pd.Series(risk, name="risk")
    try:
        bins = pd.qcut(risk_series, q=q, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(risk_series, bins=q, labels=False)

    km = KaplanMeierFitter()
    rows = []
    curves = []
    for g in sorted(np.unique(bins.dropna().astype(int))):
        mask = (bins == g).to_numpy()
        if mask.sum() == 0:
            continue
        km.fit(t[mask], e[mask])
        surv = km.survival_function_.reset_index()
        surv.columns = ["time", "survival"]
        surv.to_csv(f"{out_dir}/{prefix}_km_q{g+1}.csv", index=False)

        curves.append(
            (
                f"Q{g+1} (n={int(mask.sum())}, events={int(e[mask].sum())})",
                surv,
            )
        )
        rows.append({
            "quantile": int(g + 1),
            "n": int(mask.sum()),
            "events": int(e[mask].sum()),
            "median_survival": float(km.median_survival_time_),
            "risk_min": float(risk[mask].min()),
            "risk_max": float(risk[mask].max()),
            "risk_mean": float(risk[mask].mean()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{out_dir}/{prefix}_km_summary.csv", index=False)
    if curves:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(curves)))
        for color, (label, surv) in zip(colors, curves):
            ax.step(
                surv["time"],
                surv["survival"],
                where="post",
                lw=2,
                color=color,
                label=label,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{prefix.upper()} Kaplan–Meier by risk quantile")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Risk groups", loc="best")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{prefix}_km_plot.png", dpi=200)
        plt.close(fig)
    return summary




def train_val_test_split(X, t, e, test_size=0.3, seed=42):
    """
    Return train/test indices with stratification on (event × time quantile).
    Works even if t/e end up on GPU later.
    """
    # make sure we're on CPU for _strat_labels / sklearn
    y_strat = _strat_labels(t.cpu(), e.cpu())

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        stratify=y_strat,
        random_state=seed,
    )
    return train_idx, test_idx

def _read_matrix(path:str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")


def _append_status(project: str, modality: str, status: str) -> None:
    ts_now  = datetime.datetime.now().isoformat(timespec="seconds")
    header  = ["timestamp","project","modality","status"]
    path    = STATUS_CSV
    lock    = FileLock(str(path)+".lock", timeout=60)

    with lock:
        rows = []
        if path.exists():
            with path.open(newline="") as f:
                rows = list(csv.reader(f))

        if not rows or rows[0] != header:
            rows = [header]

        for i in range(1, len(rows)):
            if len(rows[i]) >= 3 and rows[i][1] == project and rows[i][2] == modality:
                rows[i] = [ts_now, project, modality, status]
                break
        else:
            rows.append([ts_now, project, modality, status])

        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        tmp.replace(path)


def load_omics(cancer_code: int,
               modality: str,
               with_clin: bool,
               transform_data: bool) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:

    # -------- clinical table -----------------
    clin = pd.read_csv(CLIN_CSV)
    clin = clin.set_index("File.ID")
    clin = clin[clin["Project.ID_code"] == cancer_code]
    if clin.empty:
        raise RuntimeError(f"No samples for code={cancer_code}")

    # drop “Solid Tissue Normal”
    norm_mask = clin["Sample.Type"] != "Solid Tissue Normal"
    clin = clin[norm_mask]

    mats, names = [], []

    if modality in ("gex", "all"):
        gex = _read_matrix(GEX_CSV)[clin.index].T
        mats.append(gex)
        names += [f"gex::{c}" for c in gex.columns]
        

    # -------- ARP blocks ---------------------
    arp_keys = EVENT_KEYS if modality == "all" \
               else [modality] if modality in EVENT_KEYS else []
    for k in arp_keys:
        df = _read_matrix(ARNAP_CSV[k])[clin.index].T
        # df = _logit_clip(df, 1e-3)
        if df.empty:
            continue
        mats.append(df)
        names += [f"{k}::{c}" for c in df.columns]

    # Concat all feats
    X = np.hstack(mats)
    print(X.shape)

    X = np.nan_to_num(X, nan=0.0)
    
    # if transform_data:
    #     X = apply_transform(X, select_transform(modality))
    #     TOP_K_EVENTS = 1000
    #     mod_lower = modality.lower()

    #     # Only do this for splicing event modalities, not for gex
    #     if mod_lower in EVENT_KEYS and X.shape[1] > TOP_K_EVENTS:
    #         # variance per feature (across samples)
    #         var = X.var(axis=0)            # shape (n_features,)
    #         # indices of top-K by variance, high → low
    #         top_idx = np.argsort(var)[::-1][:TOP_K_EVENTS]

    #         X     = X[:, top_idx]
    #         names = [names[i] for i in top_idx]
    #         print(f"[FEATURE SELECT] modality={modality}, "
    #             f"keeping {TOP_K_EVENTS} / {len(var)} highest-variance events")

    if with_clin:
        clin_frames, clin_names = [], []

        gender_cols = ["gender_female", "gender_male", "gender_nan"]
        present_gender = [c for c in gender_cols if c in clin.columns]
        if len(present_gender) == len(gender_cols):
            clin_frames.append(clin[present_gender].astype(np.float32).values)
            clin_names.extend([f"clin::{c}" for c in present_gender])

        race_cols = ["race_Asian", "race_Black", "race_White", "race_is_missing"]
        present_race = [c for c in race_cols if c in clin.columns]
        if len(present_race) == len(race_cols):
            clin_frames.append(clin[present_race].astype(np.float32).values)
            clin_names.extend([f"clin::{c}" for c in present_race])

        stage_cols = ["stage_i", "stage_ii", "stage_iii", "stage_iv", "stage_nan"]
        present_stage = [c for c in stage_cols if c in clin.columns]
        if len(present_stage) == len(stage_cols):
            clin_frames.append(clin[present_stage].astype(np.float32).values)
            clin_names.extend([f"clin::{c}" for c in present_stage])

        if "age_at_diagnosis" in clin.columns:
            z = clin["age_at_diagnosis"].astype(np.float32)
            clin_frames.append(z.values[:, None])
            clin_names.append("clin::age")
            
        if clin_frames:
            X = np.hstack([X] + clin_frames)
            names += clin_names

    # -------- targets ------------------------
    t = clin["OS.time"].values.astype(np.float32)
    e = clin["OS.event"].values.astype(np.float32)

    # -------- final NaN scrub ----------------
    keep = (~np.isnan(X).any(1)) & ~np.isnan(t) & ~np.isnan(e)
    X, t, e = X[keep], t[keep], e[keep]

    sample_ids = clin.index.values[keep]
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(e, dtype=torch.float32),
            names,
            sample_ids)


def _strat_labels(t: torch.Tensor, e: torch.Tensor, n_q: int = 4) -> np.ndarray:
    """
    Build stratification labels for survival splitting.

    Label encodes:
        time-quantile (0 .. n_q-1 or fewer if duplicates are dropped)
        × event status (0 = censored, 1 = event)

    Returns
    -------
    labels : np.ndarray of shape (N,)
        Integer labels in [0, 2*n_q - 1] (or fewer if qcut collapses bins).
    """
    # Ensure we are on CPU and detached for sklearn/pandas
    t_np = t.detach().cpu().numpy().astype(np.float32)
    e_np = e.detach().cpu().numpy().astype(int)

    # Quantile bins over time; duplicates='drop' avoids crash if few unique times
    q = pd.qcut(t_np, n_q, labels=False, duplicates="drop")
    q = np.asarray(q, dtype=int)

    return q * 2 + e_np

def cox_ph_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative partial log-likelihood for Cox PH.

    Parameters
    ----------
    risk  : (N,) tensor
        Linear predictors f(x).
    time  : (N,) tensor
        Survival / follow-up time.
    event : (N,) tensor
        1 if event occurred, 0 if censored.

    Returns
    -------
    loss : scalar tensor
        Mean negative partial log-likelihood.
    """
    # Flatten + ensure float
    time  = time.view(-1)
    event = event.view(-1)

    # Sort by descending time so risk sets are cumulative
    order = torch.argsort(time, descending=True)
    risk_ord  = risk[order]
    event_ord = event[order]

    # log sum exp of cumulative risk
    log_cumsum = torch.logcumsumexp(risk_ord, dim=0)
    pll = risk_ord - log_cumsum

    # Only contribute from events
    num_events = event_ord.sum()
    return -(pll * event_ord).sum() / (num_events + eps)

class MLPCox(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_layers: int,
        first_width: int,
        p_drop: float,
    ):
        super().__init__()

        # Geometric shrink of widths, but never below 32
        widths = [max(first_width // (2 ** i), 32) for i in range(n_layers)]
        layers = []
        last = in_dim
        for w in widths:
            layers += [
                nn.Linear(last, w),
                nn.ReLU(),
                nn.Dropout(p_drop),
            ]
            last = w

        self.body = nn.Sequential(*layers)
        self.beta = nn.Linear(last, 1, bias=False)

        # Optional: a reasonable default init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, D) tensor

        Returns
        -------
        risk : (B,) tensor
            Linear risk score for Cox PH.
        """
        return self.beta(self.body(x)).squeeze(1)
        
def select_transform(modality: str) -> str:
    """
    Map a single-modality string to the transform name applied to
    the *omics* block in `load_omics`.

    Returns one of: "none", "logit", "atanh".
    """
    m = modality.lower()

    # raw log2 CPM / counts etc.
    if m == "gex":
        return "none"

    # HIT index in [-1, 1]
    if m == "hit":
        return "atanh"

    # PSI-like in [0, 1]
    event_mods = {"afe", "ale", "se", "ri", "mxe", "a5ss", "a3ss"}
    if m in event_mods:
        return "logit"

    # For "all" or anything unknown we do nothing here
    return "none"

def apply_transform(x, mode: str, eps: float = 1e-3):
    """
    Apply a simple elementwise transform to a numpy array-like `x`.

    mode = "none"  → identity
    mode = "logit" → log(x / (1 - x)), clipped to [eps, 1-eps]
    mode = "atanh" → 0.5 * log((1 + x) / (1 - x)), clipped to (-1+eps, 1-eps)
    """
    if mode == "none":
        return x

    # we only call this before converting to torch, so go through numpy
    x = np.asarray(x, dtype=np.float32)

    if mode == "logit":
        x_clipped = np.clip(x, eps, 1.0 - eps)
        return np.log(x_clipped / (1.0 - x_clipped))

    if mode == "atanh":
        x_clipped = np.clip(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * np.log((1.0 + x_clipped) / (1.0 - x_clipped))

    raise ValueError(f"Unknown transform: {mode}")

def fit_zscore(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std for z-scoring.

    Parameters
    ----------
    x : (N, D) float tensor

    Returns
    -------
    mu    : (1, D) tensor
    sigma : (1, D) tensor, std with floor at 1e-9
    """
    x = x.float()
    mu = x.mean(dim=0, keepdim=True)
    sigma = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-9)
    return mu, sigma

def apply_zscore(x: torch.Tensor,
                 mu: torch.Tensor,
                 sigma: torch.Tensor) -> torch.Tensor:
    """
    Apply precomputed z-score stats featurewise.

    All tensors are broadcast along dim=0.
    """
    return (x - mu) / sigma


def train_once(
    X: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    feat_names: List[str],
    cfg: dict,
    modality: str,
    transform_data: bool = True,
    folds: int = 5,
    seed: int = SEED
) -> float:
    """
    One Optuna evaluation:
    k-fold CV on (X, t, e), returning mean validation C-index.
    """

    # Stratification labels (event × time-quantile)
    y_strat = _strat_labels(t, e)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    # indices of features to z-score (non-clin OR clin::age)
    non_clin_idx = torch.tensor(
        [i for i, n in enumerate(feat_names)
         if (not n.startswith("clin::")) or (n == "clin::age")],
        dtype=torch.long,
        device=X.device,
    )

    cidx_scores: List[float] = []
    idxs = np.arange(len(X))

    for fold, (tr, va) in enumerate(skf.split(idxs, y_strat), start=1):
        # --- split & clone so we don't mutate the full X ---
        X_tr = X[tr].clone()
        X_va = X[va].clone()

        # --- per-fold z-score on non-clinical features ---
        if non_clin_idx.numel() > 0:
            mu, sigma = fit_zscore(X_tr[:, non_clin_idx])
            if transform_data:
                X_tr[:, non_clin_idx] = apply_zscore(X_tr[:, non_clin_idx], mu, sigma)
                X_va[:, non_clin_idx] = apply_zscore(X_va[:, non_clin_idx], mu, sigma)

        # move to device
        X_tr_d = X_tr.to(DEVICE)
        X_va_d = X_va.to(DEVICE)
        t_tr_d = t[tr].to(DEVICE)
        e_tr_d = e[tr].to(DEVICE)
        t_va_d = t[va].to(DEVICE)
        e_va_d = e[va].to(DEVICE)

        # --- model / opt per fold ---
        net = MLPCox(
            in_dim=X.shape[1],
            n_layers=cfg["layers"],
            first_width=cfg["width"],
            p_drop=cfg["drop"],
        ).to(DEVICE)

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["wd"],
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )

        best_c = -np.inf
        patience = 100
        stall = 0

        for ep in range(1, cfg["epochs"] + 1):
            # ---------------------- train ----------------------
            net.train()
            opt.zero_grad()

            risk_tr = net(X_tr_d)
            loss = cox_ph_loss(risk_tr, t_tr_d, e_tr_d)
            if torch.isnan(loss) or torch.isinf(loss):
                return float("-inf")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()

            # ---------------------- val ------------------------
            net.eval()
            with torch.no_grad():
                risk_val = net(X_va_d)
                loss_val = cox_ph_loss(risk_val, t_va_d, e_va_d)

            # C-index on CPU
            c_val = concordance_index(
                t[va].detach().cpu().numpy(),
                -risk_val.detach().cpu().numpy(),
                e[va].detach().cpu().numpy(),
            )

            if c_val > best_c:
                best_c = c_val
                stall = 0
            else:
                stall += 1

            sched.step(-c_val)
            if stall >= patience:
                break

        cidx_scores.append(best_c)

    return float(np.mean(cidx_scores))


def objective(trial, X, t, e, feat_names, modality, transform_data):
    cfg = {
        "layers" : trial.suggest_int("layers", 2, 8),
        "width"  : trial.suggest_categorical("width", [4096, 2048, 1024, 512, 256, 128]),
        "drop"   : trial.suggest_float("drop", 0.1, 0.8),
        "lr"     : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "wd"     : trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "epochs" : trial.suggest_categorical("epochs", [100, 200, 300, 400, 500, 750, 1000, 1250]),
    }
    return train_once(X, t, e, feat_names, cfg, modality, transform_data)



def fit_full(
    X: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    feat_names: List[str],
    cfg: dict,
    modality: str,
    model_dir: str,
    transform_data: bool,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Train final MLP-Cox model on a single train/val split using the
    best hyper-parameters, and return the fitted network plus z-score
    stats (mu, sigma) for non-clinical features.

    X, t, e are assumed to be **unscaled** versions coming out of
    load_omics (i.e. only logit/atanh transforms applied if requested).
    """

    os.makedirs(model_dir, exist_ok=True)

    # --- build model -------------------------------------------------------
    net = MLPCox(
        in_dim=X.shape[1],
        n_layers=cfg["layers"],
        first_width=cfg["width"],
        p_drop=cfg["drop"],
    ).to(device)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    # keep tensors on CPU here; we’ll move train/val splits to device below
    X_cpu = X.detach().cpu()
    t_cpu = t.detach().cpu()
    e_cpu = e.detach().cpu()

    N = X_cpu.shape[0]

    # stratified split (event × time-quantile)
    y_strat = _strat_labels(t_cpu, e_cpu)
    train_idx_np, val_idx_np = train_test_split(
        np.arange(N),
        test_size=0.33,
        stratify=y_strat,
        random_state=SEED,
    )

    train_idx = torch.from_numpy(train_idx_np)
    val_idx   = torch.from_numpy(val_idx_np)

    # --- compute z-score stats on non-clinical + age (CPU) -----------------
    non_clin_idx_cpu = torch.tensor(
        [i for i, name in enumerate(feat_names)
         if (not name.startswith("clin::")) or (name == "clin::age")],
        dtype=torch.long,
    )

    # stats on TRAIN subset only
    mu_cpu, sigma_cpu = fit_zscore(X_cpu[train_idx][:, non_clin_idx_cpu])

    # --- move split + stats to device and apply scaling --------------------
    X_train = X_cpu[train_idx].to(device)
    X_val   = X_cpu[val_idx].to(device)
    t_train = t_cpu[train_idx].to(device)
    e_train = e_cpu[train_idx].to(device)
    t_val   = t_cpu[val_idx].to(device)
    e_val   = e_cpu[val_idx].to(device)

    non_clin_idx_dev = non_clin_idx_cpu.to(device)
    mu_dev    = mu_cpu.to(device)
    sigma_dev = sigma_cpu.to(device)

    print("X_train mean/std:", 
      X_train[:, non_clin_idx_dev].mean().item(), 
      X_train[:, non_clin_idx_dev].std().item())

    # z-score only the selected features
    if transform_data:
        X_train[:, non_clin_idx_dev] = apply_zscore(X_train[:, non_clin_idx_dev],
                                                    mu_dev, sigma_dev)
        X_val[:, non_clin_idx_dev]   = apply_zscore(X_val[:, non_clin_idx_dev],
                                                    mu_dev, sigma_dev)

    print("X_train mean/std:", 
      X_train[:, non_clin_idx_dev].mean().item(), 
      X_train[:, non_clin_idx_dev].std().item())

    n_events_val = int(e_val.sum().item())
    print("VAL events:", n_events_val)
    n_events_train = int(e_train.sum().item())
    print("TRAIN events:", n_events_train)

    len_val = len(e_val)
    print("VAL total:", len_val)
    len_train = len(e_train)
    print("TRAIN total:", len_train)

    # --- training loop with early stopping ---------------------------------
    train_losses: List[float] = []
    val_losses:   List[float] = []
    train_cidxs:  List[float] = []
    val_cidxs:    List[float] = []

    best_val_c = -float("inf")
    best_state = None
    stall      = 0
    patience   = 100

    train_losses, val_losses = [], []
    train_cidxs, val_cidxs   = [], []

    for ep in range(cfg["epochs"]):
        # ---- update step ----
        net.train()
        opt.zero_grad()

        risk_train = net(X_train)
        loss_train = cox_ph_loss(risk_train, t_train, e_train)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20.0)
        opt.step()

        # ---- metrics in eval mode (dropout OFF) ----
        net.eval()
        with torch.no_grad():
            # recompute train metrics without dropout
            risk_train_eval = net(X_train)
            loss_train_eval = cox_ph_loss(risk_train_eval, t_train, e_train)
            c_train = concordance_index(
                t_train.cpu().numpy(),
                -risk_train_eval.cpu().numpy(),
                e_train.cpu().numpy(),
            )

            # validation
            risk_val = net(X_val)
            loss_val = cox_ph_loss(risk_val, t_val, e_val)
            c_val = concordance_index(
                t_val.cpu().numpy(),
                -risk_val.cpu().numpy(),
                e_val.cpu().numpy(),
            )

        # log per-epoch summary
        print(
            f"[ep {ep+1:03d}] "
            f"Train loss: {loss_train_eval.item():.4f}, "
            f"Train C: {c_train:.4f}  |  "
            f"Val loss: {loss_val.item():.4f}, "
            f"Val C: {c_val:.4f}"
        )

        # store history
        train_losses.append(loss_train_eval.item())
        val_losses.append(loss_val.item())
        train_cidxs.append(c_train)
        val_cidxs.append(c_val)

        # LR schedule + early stopping
        sched.step(loss_val.item())
        if c_val > best_val_c:
            best_val_c = c_val
            best_state = copy.deepcopy(net.state_dict())
            stall = 0
        else:
            stall += 1

        if stall >= patience:
            print(f"[EARLY STOP] Stopping at epoch {ep+1}, best val C={best_val_c:.4f}")
            break

    # --- plot loss + C-index curves ----------------------------------------
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Cox PH Loss")
    plt.legend()
    plt.title("Train vs Val Loss")
    plt.savefig(f"{model_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_cidxs, label="Train")
    plt.plot(val_cidxs,   label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Concordance Index")
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.title("Train vs Val C-index")
    plt.savefig(f"{model_dir}/cidx_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

        # --- load best weights --------------------------------------------------
    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        print("[WARN] best_state is None; using final epoch weights")

    # IMPORTANT: return mu, sigma on CPU so they can be reused on X_test, SHAP, etc.
    return net, mu_cpu, sigma_cpu

def run_shap(
    best_net: nn.Module,
    mu,
    sigma,
    X: torch.Tensor,
    feat_names: List[str],
    out_dir: str,
    sample_ids,
    nsamples: int = 200,
    background_size: int = 200,
    seed: int = SEED,
):
    """
    DeepExplainer with a *fixed* risk baseline
    -----------------------------------------
    • Pick `background_size` random samples         → X_back
    • Baseline µ = mean( risk(X_back) )
    • Explain the first `nsamples` samples          → X_expl
    The wrapper returns  r(x) − µ, so every SHAP value
    reflects deviation from that population-average risk.
    """
    ## max out for speed and csv output
    nsamples = min(int(nsamples), 1000)
    background_size = min(int(background_size), 1000)

    best_net.eval()
    device = next(best_net.parameters()).device

    # 1 ─── choose background + foreground ---------------------------------
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
    fg_idx = np.arange(min(nsamples, len(X)))

    X_back = X[bg_idx].to(device)
    X_expl = X[fg_idx].to(device)

    # 2 ─── compute global baseline µ --------------------------------------
    with torch.no_grad():
        mu = best_net(X_back).mean().item()        # scalar baseline

    # 3 ─── wrapper that subtracts µ (NOT per-batch mean) ------------------
    class RiskShift(nn.Module):
        def __init__(self, base, mu):
            super().__init__()
            self.base, self.mu = base, mu

        def forward(self, x):
            r = self.base(x).unsqueeze(1)          # shape (B,1)
            return r - self.mu                     # fixed shift

    wrapped = RiskShift(best_net, mu).to(device)

    # 4 ─── run DeepExplainer (additivity check off) -----------------------
    explainer = shap.DeepExplainer(wrapped, X_back)
    sv = explainer.shap_values(X_expl, check_additivity=False)

    # Normalize output shapes from DeepExplainer so downstream plotting
    # always receives a 2-D (N, D) matrix of SHAP values.
    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()

    # squeeze any singleton output dimension regardless of position
    if sv.ndim == 3:
        if sv.shape[-1] == 1:
            sv = sv[..., 0]
        elif sv.shape[0] == 1:
            sv = sv[0]
        elif sv.shape[1] == 1:
            sv = sv[:, 0, :]

    if sv.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP shape {sv.shape}; expected (N, D) after squeezing singleton outputs"
        )

    # 5 ─── save + plot ----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/shap_values.npy", sv)

    fg_sample_ids = [sample_ids[i] for i in fg_idx]
    shap_df = pd.DataFrame(sv, 
                           index = fg_sample_ids,
                           columns=feat_names).T
    shap_df.index.name = "feature"
    shap_df.to_csv(f"{out_dir}/shap_values.csv")

    print(sv.shape)
    fig = plt.figure(figsize=(15, 15))
    shap.summary_plot(
        sv,
        X_expl.cpu().numpy(),
        feature_names=feat_names,
        max_display=20,
        show=False
    )
    fig.tight_layout()
    plt.savefig(f"{out_dir}/summary_beeswarm.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(0).reshape(-1)
    pd.Series(mean_abs, index=feat_names) \
      .sort_values(ascending=False) \
      .to_csv(f"{out_dir}/shap_mean_abs.csv")



def main():

    p = argparse.ArgumentParser()
    p.add_argument("--cancer",     required=True,
                   help="TCGA short code e.g. BRCA")
    p.add_argument("--modality",   required=True,
                   choices=["gex", *EVENT_KEYS, "ALL"])
    p.add_argument("--max_trials", type=int, default=30)
    p.add_argument("--with_clin", action="store_true",
               help="append stage, gender, race, dx code")
    p.add_argument("--transform_data", action="store_true")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--use_preloaded", action="store_true",
               help="If set, expect PRELOADED_DATA tuple in mlp_cox_job namespace")
    p.add_argument("--dir_name", default = 'none')
    args = p.parse_args()

    global DEVICE, ROOT_DIR, STATUS_CSV

    if args.gpus == 0:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")
    print("Using device:", DEVICE)
    if args.dir_name == 'none':
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ROOT_DIR = f"{base_root}/{stamp}"
    else:
        ROOT_DIR = f"{base_root}/{args.dir_name}"
    
    os.makedirs(ROOT_DIR, exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/models", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/logs", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/shap", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/optuna", exist_ok=True)

    STATUS_CSV = Path(ROOT_DIR) / "job_status.csv"

    _append_status(args.cancer, args.modality, "Running")
    
    
    try:
        print(f"[INFO] Job (cancer={args.cancer}, modality={args.modality})")
    
        # map TCGA project code → numeric code used in your clinical table
        project_map = {"ACC":1, "BLCA":2, "BRCA":3, "CESC":4, "CHOL":5, "COAD":6, "DLBC":7, "ESCA":8, "GBM":9, "HNSC":10, 
                       "KICH":11, "KIRC":12, "KIRP":13, "LAML":14, "LGG":15, "LIHC":16, "LUAD":17, "LUSC":18, "MESO":19, "OV":20, 
                       "PAAD":21, "PCPG":22, "PRAD":23, "READ":24, "SARC":25, "SKCM":26, "STAD":27, "TGCT":28, "THCA":29, "THYM":30, 
                       "UCEC":31, "UCS":32, "UVM":33}
        cancer_code = project_map[args.cancer]
        if args.use_preloaded:
            assert PRELOADED_DATA is not None, "PRELOADED_DATA is empty!"
            X, t, e, feat_names, sample_ids = PRELOADED_DATA
            print("[INFO] Using tensors from PRELOADED_DATA")
        else:
            X, t, e, feat_names, sample_ids = load_omics(cancer_code, args.modality, with_clin=args.with_clin, transform_data = args.transform_data)
            
        
    
        init_fs_layout(ROOT_DIR, args.cancer, args.modality)

        LOW_EVENT_CANCERS = {
            "DLBC",   # 9 deaths, 38 censored
            "CHOL",   # 23 deaths, 21 censored
            "MESO",   # 73 deaths, 12 censored
            "PCPG",   # 8 deaths, 179 censored
            "PRAD",   # 10 deaths, 543 censored
            "TGCT",   # 4 deaths, 135 censored
            "THYM"    # 9 deaths, 112 censored
        }
        
        if args.cancer in LOW_EVENT_CANCERS:
            split_prop = 0.35
        else:
            split_prop = 0.15
        train_idx, test_idx = train_val_test_split(X, t, e, split_prop, SEED)
    
        X_dev,  t_dev,  e_dev  = X[train_idx], t[train_idx], e[train_idx]
        X_test, t_test, e_test = X[test_idx],  t[test_idx], e[test_idx]


        ## optuna init
        study_name = f"{args.cancer}_{args.modality}"
        storage    = f"sqlite:///{ROOT_DIR}/optuna/optuna_{study_name}.db"
        if args.max_trials!=0:

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                study_name=study_name,
                storage=storage,
                load_if_exists=True
            )

            completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
            remaining = max(0, args.max_trials - completed)
            print(f"[OPTUNA] {completed} / {args.max_trials} trials finished; scheduling {remaining} more")
            
            if remaining > 0:
                

                study.optimize(
                    lambda tr: objective(tr, X_dev, t_dev, e_dev, feat_names, args.modality, args.transform_data),
                    n_trials=remaining,
                    timeout=None
                )
            else:
                print("[OPTUNA] Target reached – skipping new trials")
                
            best_params = study.best_trial.params
            json.dump(best_params,
                      open(f"{ROOT_DIR}/logs/{study_name}_best.json","w"), indent=2)
        
            best_cfg        = best_params
            print("Best hyper-params:", best_cfg)
            best_score_dev = study.best_value
    
    
        else:
            best_score_dev = 1
            # best_params=optuna.load_study(study_name=study_name,
            #                               storage=storage
            #                               ).best_params
    
            best_params = {
                "layers": 3,
                "width": 2048,
                "drop": 0.3,
                "lr": 5e-3,
                "wd": 1e-3,
                "epochs": 200,
            }
            best_cfg = best_params
            print("Best hyper-params:", best_cfg)
    
        model_dir = f"{ROOT_DIR}/models/{args.cancer}/{args.modality}"
    
        feat_idx = torch.arange(X_dev.shape[1])
        print(f"[INFO] Samples={len(X_test)+len(X_dev)}, Features={X_dev.shape[1]}")

        best_net, mu, sigma = fit_full(X_dev, t_dev, e_dev, feat_names, best_cfg, args.modality, model_dir, args.transform_data)

        best_net.eval()
        non_clin_idx = torch.tensor(
            [i for i, n in enumerate(feat_names)
             if (not n.startswith("clin::")) or (n == "clin::age")],
            dtype=torch.long
        )
        X_dev_scaled  = X_dev.clone()
        X_test_scaled = X_test.clone()
        
        apply_zscore_inplace(X_dev_scaled,  non_clin_idx, mu, sigma)
        apply_zscore_inplace(X_test_scaled, non_clin_idx, mu, sigma)    
        
        best_net.eval()

        with torch.no_grad():
            risk_dev  = best_net(X_dev_scaled.to(DEVICE)).cpu()
            risk_test = best_net(X_test_scaled.to(DEVICE)).cpu()

        c_dev = concordance_index(
            t_dev.detach().cpu().numpy(),
            -risk_dev.detach().cpu().numpy(),
            e_dev.detach().cpu().numpy(),
        )
        c_test = concordance_index(
            t_test.detach().cpu().numpy(),
            -risk_test.detach().cpu().numpy(),
            e_test.detach().cpu().numpy(),
        )

        print(f"[C-INDEX] Dev={c_dev:.3f} | Test={c_test:.3f}")
    
        risk_df = pd.DataFrame({
            "risk": risk_test.numpy(),
            "OS.time": t_test.numpy(),
            "OS.event": e_test.numpy()
        })
        risk_df.to_csv(f"{model_dir}/risk_test.csv", index=False)
        
        
        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_net.state_dict(),
                   f"{ROOT_DIR}/models/{args.cancer}/{args.modality}/best_model.pt")
        model_path = f"{model_dir}/best_model.pt"
    
        np.save(f"{model_dir}/risk_scores.npy", risk_test.numpy())
    
        # additional discrimination / calibration summaries
        y_dev = e_dev.detach().cpu().numpy().reshape(-1)
        y_test = e_test.detach().cpu().numpy().reshape(-1)
        risk_dev_np = risk_dev.numpy().reshape(-1)
        risk_test_np = risk_test.numpy().reshape(-1)

        extra_metrics = {}
        extra_metrics.update(
            _binary_ranking_metrics(y_dev, risk_dev_np, "dev", model_dir)
        )
        extra_metrics.update(
            _binary_ranking_metrics(y_test, risk_test_np, "test", model_dir)
        )

        km_dev = _km_by_risk_quantile(
            risk_dev_np, t_dev.numpy(), y_dev, "dev", model_dir
        )
        km_test = _km_by_risk_quantile(
            risk_test_np, t_test.numpy(), y_test, "test", model_dir
        )

        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        meta = {
            "timestamp" : stamp,
            "cancer"    : args.cancer,
            "modality"  : args.modality,
            "dev_cidx"  : round(best_score_dev, 4),
            "test_cidx" : round(c_test, 4),
            "dev_km_summary": km_dev.to_dict(orient="list"),
            "test_km_summary": km_test.to_dict(orient="list"),
            "params"    : best_cfg,
            "model_pt"  : model_path,
            "feat_names": feat_names,
            "non_clin_idx": non_clin_idx.detach().cpu().numpy().tolist(),
            "mu"        : mu.detach().cpu().numpy().tolist(),
            "sigma"     : sigma.detach().cpu().numpy().tolist(),

        }
    
        json_path = f"{model_dir}/metrics.json"
        with open(json_path, "w") as jf:
            json.dump(meta, jf, indent=2)
        
        # 3) append to global CSV ledger
        ledger = f"{model_dir}/run_summary.csv"
        first  = not os.path.exists(ledger)
        with open(ledger, "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(["timestamp","cancer","modality",
                                 "dev_cidx","test_cidx","model_path"])
            writer.writerow([stamp, args.cancer, args.modality,
                             f"{best_score_dev:.4f}", f"{c_test:.4f}", model_path])
        
        print(f"[SAVE] model → {model_path}")
        print(f"[SAVE] metrics→ {json_path}")
        print(f"[APPEND] ledger→ {ledger}")
        out_dir = f"{ROOT_DIR}/shap/{args.cancer}/{args.modality}"
        os.makedirs(out_dir, exist_ok=True)


        

        

        # ------------------  SHAP ---------------------------------
        out_dir = f"{ROOT_DIR}/shap/{args.cancer}/{args.modality}"
        os.makedirs(out_dir, exist_ok=True)

        X_scaled = X.clone()

        apply_zscore_inplace(X_scaled, non_clin_idx, mu, sigma)
        feat_names = [feat_names[i] for i in feat_idx.tolist()]

        run_shap(best_net, mu, sigma, X_scaled[:, feat_idx], feat_names, out_dir, sample_ids, 100, 100)#X.shape[0], X.shape[0])

        print(f"[DONE] {args.cancer}-{args.modality}  meanC={c_test:.3f}")
        _append_status(args.cancer, args.modality, "Completed")
    

    except Exception as ex:
        _append_status(args.cancer, args.modality,
                       f"Terminated: {type(ex).__name__}")
        raise



if __name__ == "__main__":
    main()
