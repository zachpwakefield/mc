import json
import subprocess
import sys
from pathlib import Path
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "seaborn"])
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────
# 1.  Locate and parse every metrics.json
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/projectnb/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/models")

records = []

for metrics_file in BASE_DIR.glob("*/*/metrics.json"):
    # metrics_file = …/models/{cancer}/{modality}/metrics.json
    cancer_type = metrics_file.parents[1].name     # folder two levels up
    modality    = metrics_file.parent.name         # folder one level up

    try:
        with metrics_file.open() as fp:
            metrics = json.load(fp)
        test_cidx = metrics.get("test_cidx")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️  Skipping unreadable file {metrics_file}: {exc}")
        continue

    # In case the key is missing or the value is None
    if test_cidx is None:
        print(f"⚠️  No test_cidx in {metrics_file}")
        continue

    records.append(
        {
            "cancer":   cancer_type,
            "modality": modality,
            "test_cidx": float(test_cidx),
            "path":      str(metrics_file),   # helpful for debugging
        }
    )

if not records:
    raise RuntimeError("No metrics.json files were successfully read!")

df = pd.DataFrame.from_records(records)
print(f"Loaded {len(df):,} result files "
      f"({df['cancer'].nunique()} cancers, {df['modality'].nunique()} modalities).")
print(f"cancers included in analyses: {df['cancer'].unique()}")
# ──────────────────────────────────────────────────────────────────────────
# 2.  Make the box-plots
# ──────────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")         # nice defaults; comment out if using matplotlib only
sns.set_context("talk", font_scale=0.9)

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Hard-coded root for all plots
PLOT_DIR = Path("/projectnb/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)   # make sure it exists

def save_fig(name: str,
             subdir: str | None = None,
             dpi: int = 300,
             bbox_inches: str = "tight",
             **savefig_kwargs) -> Path:
    """
    Save the active Matplotlib figure under `PLOT_DIR` (or a child folder).
    
    Parameters
    ----------
    name : str
        File name, with or without extension. '.png' is added if omitted.
    subdir : str, optional
        Optional subfolder inside `PLOT_DIR` (e.g. cancer code, date).
    dpi : int
        Resolution of the saved image.
    bbox_inches : str
        Passed to plt.savefig; "tight" removes extra whitespace.
    **savefig_kwargs
        Any other plt.savefig keyword—e.g. transparent=True.
    
    Returns
    -------
    Path to the written file (useful for logging).
    """
    # Ensure extension
    if not Path(name).suffix:
        name += ".png"

    # Optional nested folder
    target_dir = PLOT_DIR / subdir if subdir else PLOT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp to avoid overwriting
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{Path(name).stem}_{stamp}{Path(name).suffix}"
    fpath = target_dir / filename

    plt.savefig(fpath, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    print(f"[saved plot] {fpath}")
    return fpath

    # %matplotlib inline             # or `%matplotlib widget` for interactive zoom

import json, subprocess, sys
from pathlib import Path

# ── 1. Ensure seaborn is available ───────────────────────────────
try:
    import seaborn as sns
except ModuleNotFoundError:
    print("Installing seaborn …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "seaborn"])
    import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

# ── 2. Point to the directory that actually holds your runs ─────
# CHANGE THIS if your files are really under /projectnb2/…
BASE_DIR = Path("/projectnb/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/models")
# Example of alternative:
# BASE_DIR = Path("/projectnb2/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/models")

print(f"Scanning under: {BASE_DIR}")

# ── 3. Collect metrics.json files (recursive) ────────────────────
records = []
for metrics_file in BASE_DIR.rglob("metrics.json"):
    try:
        cancer   = metrics_file.parents[1].name   # {cancer}/{modality}/metrics.json
        modality = metrics_file.parent.name

        with metrics_file.open() as fp:
            metrics = json.load(fp)

        test_cidx = metrics.get("test_cidx")
        if test_cidx is None:
            continue

        records.append(
            {"cancer": cancer,
             "modality": modality,
             "test_cidx": float(test_cidx)}
        )
    except Exception as exc:
        print(f"⚠️  skipping {metrics_file}: {exc}")

df = pd.DataFrame(records)
print(f"\n➡  {len(df)} result files loaded "
      f"({df['cancer'].nunique()} cancers, {df['modality'].nunique()} modalities)")

if df.empty:
    raise RuntimeError("No data found – double-check BASE_DIR")

import seaborn as sns
sns.set_style("whitegrid"); sns.set_context("talk", font_scale=0.9)

# ── determine sort order by median ──────────────────────────────
order_mod = (
    df.groupby("modality")["test_cidx"]
      .median()
      .sort_values(ascending=False)     # highest median first
      .index
)

order_can = (
    df.groupby("cancer")["test_cidx"]
      .median()
      .sort_values(ascending=False)
      .index
)

# ── box-plot: across modalities ─────────────────────────────────
fig_mod, ax_mod = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="modality", y="test_cidx",
            order=order_mod,
            ax=ax_mod, linewidth=1)
ax_mod.set(title="Test c-index across modalities (ordered by median)",
           xlabel="Modality", ylabel="Test c-index")
ax_mod.tick_params(axis="x", rotation=45)
plt.tight_layout()
# plt.show()

save_fig("ModalitySummary.pdf")

# ── box-plot: across cancer types ───────────────────────────────
fig_can, ax_can = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df, x="cancer", y="test_cidx",
            order=order_can,
            ax=ax_can, linewidth=1)

ax_can.set(title="Test c-index across cancer types (ordered by median)",
           xlabel="Cancer type", ylabel="Test c-index")
ax_can.tick_params(axis="x", rotation=90)

# ▸ annotate: show which modality produced the max c-index per cancer
y_offset = 0.01                    # lift labels a bit above the whisker
for xpos, cancer in enumerate(order_can):
    cancer_slice = df.loc[df["cancer"] == cancer]
    idx_best     = cancer_slice["test_cidx"].idxmax()
    best_val     = cancer_slice.loc[idx_best, "test_cidx"]
    best_modal   = cancer_slice.loc[idx_best, "modality"]
    ax_can.text(xpos, best_val + y_offset, best_modal,
                ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
# plt.show()

save_fig("CancerTypeSummary.pdf")

# ── bar-plot: winner counts, sorted descending ───────────────────────────
# 1. Which modality wins in each cancer?
best_rows = (
    df.loc[df.groupby("cancer")["test_cidx"].idxmax()]   # one max row per cancer
)

best_count = (
    best_rows["modality"]
      .value_counts()
      .sort_values(ascending=False)      # ← sort by frequency ↓
)

# 2. Plot
fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
sns.barplot(x=best_count.index, y=best_count.values, ax=ax_bar)
ax_bar.set(
    title="Modality that achieves the highest c-index (count of cancers)",
    xlabel="Modality",
    ylabel="Number of cancers"
)
ax_bar.bar_label(ax_bar.containers[0], padding=3)        # add counts on bars
ax_bar.tick_params(axis="x", rotation=45)
plt.tight_layout()
# plt.show()

save_fig("HighestCindex.pdf")

# ── rank-sum plot: lower = better ──────────────────────────────────────
# 1. Rank modalities inside each cancer (highest c-index → rank 1)
df["rank"] = (
    df.groupby("cancer")["test_cidx"]
      .rank(method="min", ascending=False)
)

# 2. Sum the ranks over cancers
rank_sum = (
    df.groupby("modality")["rank"]
      .sum()
      .sort_values()              # ascending: best overall first
)

# 3. Plot
fig_rank, ax_rank = plt.subplots(figsize=(8, 4))
sns.barplot(x=rank_sum.index, y=rank_sum.values, ax=ax_rank,
            order=rank_sum.index)           # already sorted
ax_rank.set(
    title="Sum of modality ranks across cancers (lower = better)",
    xlabel="Modality",
    ylabel="Sum of per-cancer ranks"
)
ax_rank.bar_label(ax_rank.containers[0], padding=3)      # show numbers
ax_rank.tick_params(axis="x", rotation=45)
plt.tight_layout()
# plt.show()

save_fig("SummedScores.pdf")

# ── 1. Rank each modality inside each cancer  (1 = best) ───────────────
df["rank"] = (
    df.groupby("cancer")["test_cidx"]
      .rank(method="min", ascending=False)
)

# ── 2. Median rank per modality → ordering vector ──────────────────────
median_rank = (
    df.groupby("modality")["rank"]
      .median()
      .sort_values()             # lower median = better overall
)

order_mod_rank = median_rank.index     # modalities sorted by median rank

# ── 3. Box-plot of per-cancer ranks  ───────────────────────────────────
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="modality", y="rank",
            order=order_mod_rank, linewidth=1)
plt.axhline(1, ls="--", c="grey", lw=0.8)   # visual ‘perfect’ line (optional)
plt.xlabel("Modality")
plt.ylabel("Per-cancer rank (1 = best c-index)")
plt.title("Distribution of modality ranks across cancers\n(sorted by median rank)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
# plt.show()

save_fig("MedianScores.pdf")

import subprocess, sys, re
from pathlib import Path

def ensure(pkg):
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
        __import__(pkg)

ensure("pandas");       import pandas as pd
ensure("matplotlib");   import matplotlib.pyplot as plt
ensure("seaborn");      import seaborn as sns
ensure("upsetplot");    from upsetplot import UpSet, from_memberships
import re
gene_re = re.compile(r"(ENSG\d{11})")

BASE_SHAP  = Path("/projectnb/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/shap")
CSV_NAME  = "shap_mean_abs.csv"   # adapt if your file has a different name
TOP_K     = 150                    # keep this many features per model
TOP_N_BAR = 15                    # bar-plot size


def grab_gene(raw):
    """Return ENSG00000… if present, else the original string."""
    m = gene_re.search(str(raw))
    return m.group(1) if m else str(raw)

shap_records = []

for csv_path in BASE_SHAP.rglob(CSV_NAME):
    cancer, modality = csv_path.parents[1].name, csv_path.parent.name
    try:
        # 1️⃣  read with pandas auto-detect
        df_raw = pd.read_csv(csv_path, engine="python")          # handles commas / tabs

        # 2️⃣  force at least two columns: feature (first) + shap (last)
        if df_raw.shape[1] < 2:
            raise ValueError("needs ≥2 columns (feature, shap)")
        df = df_raw.iloc[:, [0, -1]].copy()
        df.columns = ["feature", "shap"]

        # 3️⃣  numeric coercion of SHAP column
        df["shap"] = pd.to_numeric(df["shap"], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=["shap"])
        if len(df) < n_before:
            print(f"    ↳ {csv_path.name}: dropped {n_before-len(df)} non-numeric rows")

        # 4️⃣  gene extraction
        df["gene_id"] = df["feature"].apply(grab_gene)

        # 5️⃣  keep top-k
        top = df.nlargest(TOP_K, "shap")
        for rank, row in top.iterrows():
            shap_records.append({
                "cancer":   cancer,
                "modality": modality,
                "feature":  row["feature"],
                "gene_id":  row["gene_id"],
                "rank":     rank + 1,
                "shap_val": row["shap"],
            })

    except Exception as exc:
        print(f"⚠️  skipping {csv_path.relative_to(BASE_SHAP)} → {exc}")

shap_df = pd.DataFrame(shap_records)
print("SHAP loaded...")

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "mygene"])
import mygene

# ▸ function to add HGNC symbols to shap_df ─────────────────────────
def add_hgnc_symbols(df, ensembl_col="gene_id"):
    """
    Return a copy of `df` with a new `hgnc_symbol` column.
    Rows whose Ensembl IDs cannot be mapped keep NaN in that column.
    """
    mg = mygene.MyGeneInfo()
    
    # grab unique ENSG IDs that look legit
    unique_ensg = df[ensembl_col].str.match(r"^ENSG\d{11}$", na=False)
    ensg_list   = df.loc[unique_ensg, ensembl_col].unique().tolist()
    
    if not ensg_list:
        print("No ENSG IDs detected – skipping symbol lookup.")
        return df.assign(hgnc_symbol=np.nan)
    
    print(f"Querying MyGene.info for {len(ensg_list)} Ensembl IDs …")
    out = mg.querymany(ensg_list, scopes="ensembl.gene", fields="symbol", species="human")
    
    # build dict → quick lookup
    id2sym = {d["query"]: d.get("symbol") for d in out if not d.get("notfound")}
    
    df = df.copy()
    df["hgnc_symbol"] = df[ensembl_col].map(id2sym)
    return df

# ▸ apply right after `shap_df` is created ──────────────────────────
shap_df = add_hgnc_symbols(shap_df)

top_df = shap_df.copy()
if top_df.empty:
    raise RuntimeError("No SHAP CSVs found – check BASE_DIR / CSV_NAME.")

print(f"Aggregated {len(top_df):,} rows "
      f"({top_df.cancer.nunique()} cancers, {top_df.modality.nunique()} modalities)")

# ── visual 1: UpSet plot (gene × modality membership) ───────────────────
membership = (
    top_df.groupby(["hgnc_symbol", "modality"])
          .size()
          .unstack(fill_value=0)        # gene_id × modality bool matrix
          .astype(bool)
          .apply(lambda row: tuple(row.index[row].tolist()), axis=1)
)

plt.figure(figsize=(10, 6))
UpSet(from_memberships(membership), subset_size='count', show_counts=True,
      min_subset_size=90).plot()
plt.suptitle(f"UpSet – gene membership across modalities (top {TOP_K})")
# plt.show()

save_fig("top150Upset.pdf")


# ── visual 2: recurrent genes across cancers ────────────────────────────
recurrent = (top_df.groupby("hgnc_symbol")["cancer"]
                    .nunique()
                    .sort_values(ascending=False))

sns.set_style("whitegrid"); sns.set_context("talk", font_scale=0.9)
plt.figure(figsize=(8, 6))
sns.barplot(y=recurrent.head(TOP_N_BAR).index,
            x=recurrent.head(TOP_N_BAR).values,
            palette="viridis")
plt.xlabel("Cancers where gene is top-k")
plt.ylabel("Gene ID")
plt.title(f"Top {TOP_N_BAR} recurrent genes across cancers")
plt.tight_layout()
# plt.show()

save_fig("top150Shap.pdf")
# ── export tidy table for enrichment etc. ───────────────────────────────
top_df.to_csv("top_k_shap_features.csv", index=False)
print("✓ Saved tidy table → top_k_shap_features.csv")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1️⃣  Tell me which identifier column to use
ID_COL = "hgnc_symbol"          # ← use "gene_id" if symbols not available

# 2️⃣  Pick the genes you care about  (case-insensitive for symbols)
genes_of_interest = [
    # canonical tumour suppressors / oncogenes
    "TP53", "KRAS", "BRAF", "PIK3CA", "PTEN", "MYC",
    # context-specific or pathway drivers
    "EGFR", "ERBB2", "ALK", "FLT3", "NPM1",
    # DNA-repair & BRCA axis
    "BRCA1", "BRCA2",
    # metabolic mutations
    "IDH1", "IDH2",
    # TERT promoter / telomerase
    "TERT",
    # immune checkpoint / micro-environment
    "CD274"   # PD-L1
]

# harmonise case for symbols
if ID_COL == "hgnc_symbol":
    shap_df[ID_COL] = shap_df[ID_COL].str.upper()

shap_sub = shap_df[shap_df[ID_COL].isin([g.upper() for g in genes_of_interest])]

if shap_sub.empty:
    raise ValueError("None of the chosen genes appear in the SHAP top-k tables.")

# 3️⃣  Summary table
summary = (
    shap_sub.groupby(ID_COL)
            .agg(n_models = ("rank", "size"),
                 mean_rank = ("rank", "mean"),
                 median_rank = ("rank", "median"),
                 best_rank = ("rank", "min"))
            .sort_values("median_rank")
)

display(summary)

# 4️⃣  Violin + strip plot of ranks
sns.set_style("whitegrid"); sns.set_context("talk", font_scale=0.9)
plt.figure(figsize=(8, 5))
sns.violinplot(data=shap_sub, x=ID_COL, y="rank", order=summary.index,
               inner=None, cut=0)
sns.stripplot(data=shap_sub, x=ID_COL, y="rank", order=summary.index,
              hue="modality", dodge=False, size=3, alpha=0.7)
plt.gca().invert_yaxis()          # rank 1 at top
plt.ylabel("Rank (1 = most important)")
plt.title(f"SHAP rank distribution for selected prognosis genes\n(top-{TOP_K} per model)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
# plt.show()

save_fig("top150MajorGenes.pdf")

import json, subprocess, sys, warnings
from pathlib import Path
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────
# 1.  Collect MADfeatures values
BASE_MET = Path("/projectnb/evolution/zwakefield/tcga/sir_analysis/survivalModel/mlpCOX/models")
records  = []

for mj in BASE_MET.glob("*/*/metrics.json"):
    cancer, modality = mj.parents[1].name, mj.parent.name
    try:
        with mj.open() as fp:
            meta = json.load(fp)
        k = meta["params"].get("MADfeatures")
        if k is None:
            raise KeyError("MADfeatures missing")
        records.append({"cancer": cancer, "modality": modality, "MADfeatures": int(k)})
    except Exception as e:
        warnings.warn(f"skip {mj.relative_to(BASE_MET)} → {e}")

mad_df = pd.DataFrame(records)
if mad_df.empty:
    raise RuntimeError("No MADfeatures found—check path or file schema.")

print(f"{len(mad_df):,} model runs parsed "
      f"({mad_df.cancer.nunique()} cancers, {mad_df.modality.nunique()} modalities)")
display(mad_df.describe(include='all'))

# ───────────────────────────────────────────────────────────────
# 2.  Plot overall histogram
sns.set_style("whitegrid"); sns.set_context("talk", font_scale=0.9)

plt.figure(figsize=(8,5))
sns.histplot(mad_df["MADfeatures"], bins=30, kde=True, alpha=0.8)
plt.xlabel("MADfeatures (features kept after MAD filtering)")
plt.title("Overall distribution of MADfeatures across all models")
plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────
# 3.  Box-plot by modality
plt.figure(figsize=(10,5))
order_mod = (mad_df.groupby("modality")["MADfeatures"]
                     .median()
                     .sort_values(ascending=False).index)
sns.boxplot(data=mad_df, x="modality", y="MADfeatures", order=order_mod)
plt.xticks(rotation=45, ha="right")
plt.title("MADfeatures distribution by modality")
plt.tight_layout(); plt.show()

# ───────────────────────────────────────────────────────────────
# 4.  Box-plot by cancer type
plt.figure(figsize=(14,5))
order_can = (mad_df.groupby("cancer")["MADfeatures"]
                     .median()
                     .sort_values(ascending=False).index)
sns.boxplot(data=mad_df, x="cancer", y="MADfeatures", order=order_can)
plt.xticks(rotation=90)
plt.title("MADfeatures distribution by cancer type")
plt.tight_layout(); plt.show()

# How many distinct (cancer, modality) combinations have metrics.json ?
n_pairs = shap_df[["cancer", "modality"]].drop_duplicates().shape[0]
print(f"Unique cancer-modality pairs: {n_pairs}")
print("\nModalities per cancer:")
display(shap_df.groupby("cancer")["modality"].nunique().sort_values(ascending=False))

print("\nCancers per modality:")
display(shap_df.groupby("modality")["cancer"].nunique().sort_values(ascending=False))

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gseapy"])

import pandas as pd
import gseapy as gp

# 1. ─ Load SHAP table (tidy format we saved earlier)
shap_df = pd.read_csv("top_k_shap_features.csv")

# 2. ─ Select genes you want to test
TOP_RANK  = 100                      # top-k per model
query     = shap_df["rank"] <= TOP_RANK
genes     = (shap_df.loc[query, "hgnc_symbol"]
                     .dropna()
                     .str.upper()   # Enrichr expects upper-case symbols
                     .unique()
                     .tolist())

print(f"Submitting {len(genes)} unique genes to Enrichr…")

# 3.1 ─ Run Enrichr (GO BP as example; pick any collection you like)
enrGOBP = gp.enrichr(
    gene_list   = genes,
    gene_sets   = "GO_Biological_Process_2023",
    organism    = "Human",
    outdir      = None,             # no files on disk
    cutoff      = 0.05,             # adjust-P threshold
)

# 4.1 ─ Show top results
cols = ["Term", "Adjusted P-value", "Overlap", "Combined Score"]
display(enrGOBP.results.sort_values("Adjusted P-value").head(5)[cols])

# 3.2 ─ Run Enrichr (GO BP as example; pick any collection you like)
enrKEGG = gp.enrichr(gene_list=genes, # or "./tests/data/gene_list.txt",
                 gene_sets=['KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None,
                 cutoff      = 0.05
                )

# 4.2 ─ Show top results
cols = ["Term", "Adjusted P-value", "Overlap", "Combined Score"]
display(enrKEGG.results.sort_values("Adjusted P-value").head(5)[cols])

# 3.3 ─ Run Enrichr (GO BP as example; pick any collection you like)
enrHALL = gp.enrichr(gene_list=genes, # or "./tests/data/gene_list.txt",
                 gene_sets=['MSigDB_Hallmark_2020'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None,
                 cutoff      = 0.05
                )

# 4.3 ─ Show top results
cols = ["Term", "Adjusted P-value", "Overlap", "Combined Score"]
display(enrHALL.results.sort_values("Adjusted P-value").head(5)[cols])

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

def plot_enrichment(df: pd.DataFrame,
                    p_col: str = "Adjusted P-value",
                    geneset_col: str = "Term",
                    size_col: str = "Overlap",      # or "Genes"
                    color_col: str = "Combined Score",
                    top_n: int = 15,
                    figsize: tuple = (9, 5),
                   fig_name: str = "enrichment"):
    """
    Visualises an enrichment result DataFrame with:
    1. horizontal bar plot  (−log10 p-value)
    2. dot plot             (−log10 p-value × gene-set size × color)
    
    Parameters
    ----------
    df : DataFrame
        The result table from gseapy.enrichr() or gseapy.gsea().
    p_col : str
        Column with (adjusted) p-values.
    geneset_col : str
        Column with gene-set names.
    size_col : str
        Column encoding gene-set size (e.g. "Overlap" or "Genes").
    color_col : str
        Column to colour-map in the dot plot.
    top_n : int
        How many top rows (by p-value) to display.
    figsize : tuple
        Figure size for each plot.
    """
    if df.empty:
        raise ValueError("Enrichment DataFrame is empty.")

    # keep only numeric rows
    df = df.copy()
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    df = df.dropna(subset=[p_col])
    df = df[pd.to_numeric(df[p_col], errors="coerce") < 0.1]
    if df.empty:
        raise ValueError(f"No numeric {p_col} values to plot.")

    df_top = df.sort_values(p_col).head(top_n)
    y_labels = df_top[geneset_col]

    log_p = -np.log10(df_top[p_col].values)

    # ── 1. horizontal bar plot ─────────────────────────────────────
    plt.figure(figsize=figsize)
    plt.barh(y_labels, log_p)
    plt.xlabel(f"-log10 {p_col}")
    plt.title(f"Top {top_n} enriched terms")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()

    save_fig(f"{fig_name}_bar.pdf")
    # ── 2. dot (bubble) plot ───────────────────────────────────────
    # ensure size & colour columns numeric
    df_top[size_col]   = pd.to_numeric(df_top[size_col], errors="coerce").fillna(0)
    df_top[color_col]  = pd.to_numeric(df_top[color_col], errors="coerce").fillna(0)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=df_top,
                    x=log_p,
                    y=y_labels,
                    size=size_col,
                    hue=color_col,
                    palette="viridis",
                    sizes=(50, 600),
                    legend="brief")
    plt.xlabel(f"-log10 {p_col}")
    plt.ylabel("")
    plt.title(f"Dot plot – {top_n} terms")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.show()
    save_fig(f"{fig_name}_dot.pdf")

# ────────────────────────────────────────────────────────────────
# Usage example   (assuming you already have `enr` from gseapy.enrichr)
plot_enrichment(
    df           = enrGOBP.results,
    p_col        = "Adjusted P-value",
    geneset_col  = "Term",
    size_col     = "Overlap",
    color_col    = "Combined Score",
    top_n        = 5,
    fig_name = "enrGOBP150"
)


# plot_enrichment(
#     df           = enrKEGG.results,
#     p_col        = "Adjusted P-value",
#     geneset_col  = "Term",
#     size_col     = "Overlap",
#     color_col    = "Combined Score",
#     top_n        = 5
# )
plot_enrichment(
    df           = enrHALL.results,
    p_col        = "Adjusted P-value",
    geneset_col  = "Term",
    size_col     = "Overlap",
    color_col    = "Combined Score",
    top_n        = 5,
    fig_name = "enrHall150"
)

import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
SHAP_ROOT = Path(
    "/projectnb/evolution/zwakefield/tcga/sir_analysis/"
    "survivalModel/mlpCOX/shap"
)
MODALITIES = ["AFE","ALE","MXE","SE","RI","A3SS","A5SS","HIT","GEX"]
FIG_SUBDIR = "ModalityCorrelation"
# ────────────────────────────────────────────────────────────────────

# Gather every <CANCER>/<MODALITY>/shap_mean_abs.csv two levels deep
mod_lists = {m: [] for m in MODALITIES}

for csv in SHAP_ROOT.glob("*/*/shap_mean_abs.csv"):
    cancer, modality = csv.parts[-3:-1]           # e.g. LUAD, AFE
    if modality not in MODALITIES:
        continue

    df = (
        pd.read_csv(csv, index_col=0, header=None, names=["abs"])
          .iloc[1:]                               # drop header row
    )
    # tag feature with cancer so duplicates keep both entries
    df.index = [f"{idx}::{cancer}" for idx in df.index]
    mod_lists[modality].append(df["abs"])

# Concatenate to one Series per modality
mod_series = {
    m: pd.concat(lst) if lst else pd.Series(dtype=float)
    for m, lst in mod_lists.items()
}

# Ensure we have data
if not any(len(s) for s in mod_series.values()):
    raise RuntimeError("No SHAP files found – check SHAP_ROOT path.")

# Align into a DataFrame (feature rows × modality cols)
all_idx = pd.Index(sorted(set().union(*(s.index for s in mod_series.values()))))
X = pd.DataFrame({m: s.reindex(all_idx, fill_value=0) for m, s in mod_series.items()})

# Spearman ρ correlation matrix
corr = X.corr(method="spearman")

# ─────────────────────────── plot ────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1,
            cmap="vlag", cbar_kws={"label": "Spearman ρ"})
plt.title("Cross-modality concordance of feature importance")
plt.tight_layout()

save_fig("Modality_SHAP_Correlation", subdir=FIG_SUBDIR)

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────── settings ──────────────────────────────────────
SHAP_ROOT = Path(
    "/projectnb/evolution/zwakefield/tcga/sir_analysis/"
    "survivalModel/mlpCOX/shap"
)
MODS   = ["AFE","ALE","MXE","SE","RI","A3SS","A5SS","HIT","GEX"]
TOP_K  = 25
FIGDIR = "JaccardOverlap"
# ───────────────────────────────────────────────────────────────

# 1 · collect sets of top-25 genes (collapse across cancers)
gene_sets = {m: set() for m in MODS}
for csv in SHAP_ROOT.glob("*/*/shap_mean_abs.csv"):
    _, mod = csv.parts[-3:-1]
    if mod not in MODS:
        continue
    df = (pd.read_csv(csv, index_col=0, header=None, names=["abs"])
            .iloc[1:]
            .sort_values("abs", ascending=False)
            .head(TOP_K))
    genes = {idx.split("::")[0] for idx in df.index}
    gene_sets[mod].update(genes)

gene_sets = {m: s for m, s in gene_sets.items() if s}

# 2 · Jaccard matrix
jac = pd.DataFrame(index=MODS, columns=MODS, dtype=float)
for i, j in itertools.product(MODS, repeat=2):
    if i not in gene_sets or j not in gene_sets:
        jac.loc[i, j] = np.nan
        continue
    inter = len(gene_sets[i] & gene_sets[j])
    union = len(gene_sets[i] | gene_sets[j])
    jac.loc[i, j] = inter / union if union else np.nan

# 3 · heat-map
plt.figure(figsize=(8, 6))
sns.heatmap(jac, annot=True, fmt=".2f", vmin=0, vmax=0.4,
            cmap="Blues", cbar_kws={"label": "Jaccard index"})
plt.title("Top-25 gene overlap across modalities")
plt.tight_layout()

save_fig("Jaccard_Top25_Genes", subdir=FIGDIR)   # vector PDF
print("[done] → see plots/JaccardOverlap/")