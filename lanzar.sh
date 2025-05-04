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
mkdir -p "${OUTPUT_DIR}"

# Collect times
times=()

for i in $(seq 1 10); do
  echo "=== Run $i ==="
  BASE="${OUTPUT_DIR}/run${i}_${ROWS}x${COLS}"
  nsys profile --stats=true -o "${BASE}" ./ej1_part1 "${ROWS}" "${COLS}" &> "${BASE}.log"
done

echo
echo "Done. Files in ${OUTPUT_DIR}:"
ls -1 "${OUTPUT_DIR}"