#!/bin/bash -l
#─────────────────────────────────────────────────────────────────────
# run_array_sge.sh   – Sun Grid Engine version
# 330 tasks = 33 cancers × 9 modalities (GEX + 8 ARP)
#─────────────────────────────────────────────────────────────────────
#$ -N single_modality_cox_nonzero10
#$ -P evolution
#$ -cwd                       # run in the current directory
#$ -j y                       # merge stdout + stderr
#$ -o /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/qsub_output/$JOB_ID.$TASK_ID.log
#$ -l h_rt=72:00:00
#$ -pe omp 8
# -l mem_per_core=8G
#$ -t 1-297
# -tc 50
#$ -m ea
#$ -M zachpwakefield@gmail.com
#─────────────────────────────────────────────────────────────────────

module load miniconda cuda gcc
cd /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/
conda activate /projectnb/evolution/zachpw/.conda/envs/survival_env_full

PAIR_CSV="/projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/tcga_job_pairs.csv"

# ── pick the line that matches this array index (SGE is 1-based) ──
pair=$(sed -n "${SGE_TASK_ID}p" "$PAIR_CSV")
cancer=${pair%%,*}             # part before the comma
mod=${pair##*,}                # part after  the comma

echo "[INFO] Task ${SGE_TASK_ID} → ${cancer} / ${mod}"

# ran:
# mad 10000
#
# nonzero_vals 0.1
#
# unique_vals 10
#

python -u /projectnb2/evolution/zwakefield/tcga/cancer_learning/single_input/mlp_cox_job.py \
       --cancer   "${cancer}" \
       --modality "${mod//$'\r'/}" \
       --with_clin            \
       --gpus 0 \
       --transform_data \
       --dir_name "model_outputs_12_1_nonzero10" \
       --max_trials 500 \
       --filter_mode "nonzero_vals" \
       --filter_num 0.1
