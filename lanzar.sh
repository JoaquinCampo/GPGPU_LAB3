#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

# module load cuda   # uncomment if your cluster supports it
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Build
make clean && make

# Matrix dims
ROWS=${1:-1024}
COLS=${2:-1024}

# Prepare output folder
OUTPUT_DIR=output
NVPROF_OUTPUT="${OUTPUT_DIR}/NVPROF"
NSYS_OUTPUT="${OUTPUT_DIR}/NSYS"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${NVPROF_OUTPUT}"
mkdir -p "${NSYS_OUTPUT}"

# Run nvprof for profiling
echo "=== Nsight Profiling (nvprof) ==="
nvprof --metrics all --o "${NVPROF_OUTPUT}/run_${ROWS}x${COLS}.nvprof" -f ./ej1_part1 "${ROWS}" "${COLS}"
echo "Done. Files in ${OUTPUT_DIR}:"

# Run just once for now
echo "=== Single Run ==="
nsys profile --output "${NSYS_OUTPUT}/run_${ROWS}x${COLS}.nsys-rep" --stats=true ./ej1_part1 "${ROWS}" "${COLS}"
echo "Done"

echo "Done"