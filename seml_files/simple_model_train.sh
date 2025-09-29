#!/bin/bash
#SBATCH --partition=gpu_gtx1080 # There was the partition for the Lab course using 
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 0-0:20 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/ceph/hdd/students/zhzo/slurm-output/slurm-%j.out"  # where the output log will be stored
#SBATCH --mem=4G # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed -- but don't set it too large since it will block resources and will lead to your job being given a low priority by the scheduler.
#SBATCH --qos=default   # this line ensures a very high priority (e.g. start a Jupyter notebook) but only one job per user can run under this mode (remove for normal compute jobs).

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

# Activate your conda environment if necessary

export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318

# Example
python ../train.py --model markovchain --dataset tdrive  --n_samples 200 --min_length 5 --max_length 5 --min_history 2  --epochs 10 --patience 100 --local_loss --rot 0.0 --trans_x 0.1 --trans_y 0.1 
