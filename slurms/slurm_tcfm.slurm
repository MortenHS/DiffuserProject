#!/bin/sh
#SBATCH --job-name="TCFM_medium_Training"
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --account=share-iv-mtp
#SBATCH --time=0-15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64G

#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

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
python /cluster/work/mortenhs/Janner/diffuser/scripts/train.py --dataset maze2d-medium-v1 --config config.maze2d
