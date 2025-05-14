#!/bin/sh
#SBATCH --job-name="pos_error_calcs"
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --account=share-iv-mtp
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G

#SBATCH --output=pos_error_output.txt      # Specifying 'stdout'
#SBATCH --error=pos_error_output.err        # Specifying 'stderr'

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
python /cluster/work/mortenhs/Janner/diffuser/scripts/pos_error_no_plot.py