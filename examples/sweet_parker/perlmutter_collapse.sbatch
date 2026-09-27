#!/bin/bash
# Sweet-Parker X-point collapse run on Perlmutter GPUs.
#
# Submit from the repository root (edit -A first):
#     sbatch examples/sweet_parker/perlmutter_collapse.sbatch
# Override parameters without editing, e.g. a short debug-queue check:
#     sbatch -q debug -t 00:30:00 --export=ALL,NX=3072,NY=768,TEND=40 \
#         examples/sweet_parker/perlmutter_collapse.sbatch
# Four GPUs on one node (whole node, so use the regular queue):
#     sbatch -q regular --gpus 4 -c 128 --export=ALL,DEVICES=4 \
#         examples/sweet_parker/perlmutter_collapse.sbatch
#
#SBATCH -J mhx-sp-collapse
#SBATCH -A <PROJECT>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus 1
#SBATCH -t 10:00:00
#SBATCH -o %x-%j.out

set -eo pipefail

ETA=${ETA:-1e-3}
NX=${NX:-6144}
NY=${NY:-768}
TEND=${TEND:-400}
SEED=${SEED:-1e-3}
LX=${LX:-25.132741228718345}  # 8*pi
LY=${LY:-25.132741228718345}  # 8*pi (ka = 0.125, Delta'a = 15.75)
CFL=${CFL:-0.25}
DEVICES=${DEVICES:-1}
EXTRA=${EXTRA:-}              # extra collapse_check.py flags, e.g. "--sap"

module load python
conda activate mhx

export JAX_ENABLE_X64=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

LX_PI=$(awk "BEGIN { printf \"%g\", ${LX} / 3.141592653589793 }")
LY_PI=$(awk "BEGIN { printf \"%g\", ${LY} / 3.141592653589793 }")
TAG=$(echo "${EXTRA}" | tr -d ' -')
RUN="collapse_eta${ETA}_${NX}x${NY}_lx${LX_PI}pi_ly${LY_PI}pi_seed${SEED}_cfl${CFL}_gpu${DEVICES}_hold${TAG:+_${TAG}}"
OUTDIR="${SCRATCH}/mhx_outputs/sweet_parker/${RUN}"
mkdir -p "${OUTDIR}"
echo "[sbatch] job ${SLURM_JOB_ID} -> ${OUTDIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

srun -n 1 --gpus "${DEVICES}" python -u "${SLURM_SUBMIT_DIR}/examples/sweet_parker/collapse_check.py" \
    --nx "${NX}" --ny "${NY}" --lx "${LX}" --ly "${LY}" \
    --eta "${ETA}" --seed "${SEED}" --t-end "${TEND}" --cfl "${CFL}" \
    --devices "${DEVICES}" --hold-equilibrium --outdir "${OUTDIR}" ${EXTRA}
