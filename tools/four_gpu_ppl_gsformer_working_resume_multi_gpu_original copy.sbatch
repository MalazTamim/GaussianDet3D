#!/bin/bash

#SBATCH --job-name=GausnFSDV2  # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --cpus-per-task=20         # Number of CPUs per task
#SBATCH --gres=gpu:4             # Request 4 GPUs
#SBATCH --mem=200GB                # Memory per node
#SBATCH --time=220:00:00          # Time limit (199 hours)
#SBATCH --mail-type=ALL           # Notifications (ALL: start, end, fail)
#SBATCH --output=/home/stud/tama/FSDV-SST/lSLURM_sept/slurm-%j_fs.out  # Standard output
#SBATCH --error=/home/stud/tama/FSDV-SST/logs-SLURM_sept/slurm-%j_fs.err   # Standard error
#SBATCH --nodelist=node[15,16,17,18,19,20]  # Specify nodes

# Load necessary modules
module avail cuda
module load cuda/11.3.0

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Activate your Conda environment
source /home/stud/tama/miniconda3/etc/profile.d/conda.sh
conda activate fs

# Find the latest epoch file in the checkpoint directory
CHECKPOINT_DIR="logs/FSDV2/N_by_28_gaussians_25600_1_prev_sample_bs8_4GPUS_modified_lr_no_object_samples_ts_index27_exp3"
LATEST_EPOCH_FILE=$(ls -v ${CHECKPOINT_DIR}/epoch_*.pth | tail -n 1)

if [ -z "$LATEST_EPOCH_FILE" ]; then
  echo "No checkpoint files found, starting from sHratch."
  bash tools/dist_train.sh configs/fsdv2/fsdv2_nusc_2x_N_by_28_multi_samples_exp3.py 4 --work-dir ${CHECKPOINT_DIR}
else
  echo "Resuming from the latest epoch: $LATEST_EPOCH_FILE"
  bash tools/dist_train.sh configs/fsdv2/fsdv2_nusc_2x_N_by_28_multi_samples_exp3.py 4 --work-dir ${CHECKPOINT_DIR} --resume-from $LATEST_EPOCH_FILE
fi
