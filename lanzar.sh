#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

#SBATCH --gres=gpu:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Script to launch CUDA transpose program under Nsight Systems profiling

# Default sizes
ROWS=${1:-1024}
COLS=${2:-1024}

OUTPUT="nsys_report_${ROWS}x${COLS}.qdrep"

# Run profiling
nsys profile --stats=true -o ${OUTPUT} ./programa ${ROWS} ${COLS}

echo "Profiling complete. Report saved to ${OUTPUT}"
