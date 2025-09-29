#!/bin/bash
#SBATCH --partition mllab  # the partition for the Lab course
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 0-02:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/ceph/hdd/students/weea/slurm_log/slurm-%j.out"  # where the output log will be stored
#SBATCH --mem=8000 # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed -- but don't set it too large since it will block resources and will lead to your job being given a low priority by the scheduler.
#SBATCH --qos=interactivelab   # this line ensures a very high priority (e.g. start a Jupyter notebook) but only one job per user can run under this mode (remove for normal compute jobs).

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary

export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318

# Example
python train.py --model simple_gnn --dataset geodrive --datapath "data/GeoLife/merged.h5" --n_samples 5000 --min_length 5 --max_length 30 --min_history 3 --no_test --feature_dim 16 --epochs 10
