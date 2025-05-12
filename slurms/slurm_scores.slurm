#!/bin/sh
#SBATCH --job-name="score_logging"
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --account=share-iv-mtp
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G

#SBATCH --output=logs/%x_%j.out  # %x: job name, %j: job ID
#SBATCH --error=logs/%x_%j.err

#SBATCH --mail-user=mortenhs@stud.ntnu.no
#SBATCH --mail-type=ALL

OUTPUT_DIR="/cluster/work/mortenhs/Janner/diffuser/"
mkdir -p ${OUTPUT_DIR}

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running file
module load Anaconda3/2024.02-1
conda activate diffuser
python /cluster/work/mortenhs/Janner/diffuser/scripts/log_scores_optim.py
