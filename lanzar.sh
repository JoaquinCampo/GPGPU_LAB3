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
# First, run Nsight Compute for detailed kernel analysis (only on first run to avoid overhead)
BASE_NCU="${OUTPUT_DIR}/run1_${ROWS}x${COLS}_ncu"
echo "=== Nsight Compute Profiling (ncu) ==="
echo "Done. Files in ${OUTPUT_DIR}:"
# ncu --list-chips
nvprof -o "${OUTPUT_DIR}/run_${ROWS}x${COLS}.nvprof" -f ./ej1_part1 "${ROWS}" "${COLS}" &> "${OUTPUT_DIR}/run_${ROWS}x${COLS}_nvprof.log"
# ncu -o test -f ej1_part1 1024 1024 &> test.log
#ncu -o "${BASE_NCU}" --set full ./ej1_part1 1024 1024 --list-chips &> "${BASE_NCU}.log"


# Run just once for now
echo "=== Single Run ==="
BASE="${OUTPUT_DIR}/run1_${ROWS}x${COLS}"
nsys profile --stats=true -o "${BASE}" ./ej1_part1 "${ROWS}" "${COLS}" &> "${BASE}.log"

# Now run nsys for all 10 runs as before
# for i in $(seq 1 10); do
#   echo "=== Run $i ==="
#   BASE="${OUTPUT_DIR}/run${i}_${ROWS}x${COLS}"
#   nsys profile --stats=true -o "${BASE}" ./ej1_part1 "${ROWS}" "${COLS}" &> "${BASE}.log"
# done

echo "Done"