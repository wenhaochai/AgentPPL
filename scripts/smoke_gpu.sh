#!/bin/bash
#SBATCH -J agenticppl-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/KARTHIKN/wc9403/project/AgenticPPL/logs/slurm/%x-%j.out
#FETCHABLE
# The record of the GPU smoke test on the Della cluster (its paths and queue are that cluster's). GPU smoke test of
# `agenticppl score` (2026-11-24): 8 dev models side by side, one per GPU, split dev;
# then scripts/smoke_compare.py checks plain PPL and AgenticPPL against the harness (acceptance: |diff| <= 1e-3).
# Submit: /scratch/gpfs/KARTHIKN/wc9403/tools/wq/wq submit --prio 20 scripts/smoke_gpu.sh
set -uo pipefail
R=/scratch/gpfs/KARTHIKN/wc9403/project/AgenticPPL
T=/scratch/gpfs/KARTHIKN/wc9403/project/token-ppl
cd $R
JOB=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/KARTHIKN/wc9403/tmp/job/$JOB
mkdir -p $TMPDIR $R/logs/slurm
trap 'rm -rf $TMPDIR' EXIT
module load proxy/default 2>/dev/null || true
module load cudatoolkit/13.0 2>/dev/null || true
source $T/.venv/bin/activate   # vLLM 0.30.0, transformers 5.23.0
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 WANDB_MODE=disabled
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor XDG_CACHE_HOME=$TMPDIR/xdg
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=$TMPDIR/vllm
export AGENTICPPL_DATA=$R/release PACKED_LEDGER_DIR=$R/logs/packed
OUT=$R/logs/smoke/$JOB
mkdir -p $OUT
# dev models whose weights are cached and that load without trust_remote_code; tp 1 each, as their dumps were made
MODELS=(Qwen/Qwen3.5-4B mistralai/Ministral-3-8B-Instruct-2512 google/gemma-4-12B-it ibm-granite/granite-4.2-3b
        ibm-granite/granite-4.2-8b Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B mistralai/Ministral-3-14B-Instruct-2512)
IFS=, read -ra ALL <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
for i in "${!MODELS[@]}"; do
  m=${MODELS[$i]}; tag=${m//\//__}
  ( export CUDA_VISIBLE_DEVICES=${ALL[$i]}
    bash /scratch/gpfs/KARTHIKN/wc9403/tools/packed_task.sh "score-$tag" - other 1 50m -- \
      python -m agenticppl score --model $m --tp 1 --split dev --out $OUT/$tag.json > $OUT/$tag.log 2>&1
    echo "$(date +%T) $m exit $?" ) &
done
wait
python scripts/smoke_compare.py $OUT/*.json | tee $OUT/compare.txt
