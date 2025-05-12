#!/bin/bash
#SBATCH --job-name=ej2_part2_profile
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00 # Increased time slightly for more profiling steps
#SBATCH --gres=gpu:1
#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

set -e # Exit immediately if a command exits with a non-zero status.

export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

# Ensure ej2_part2 is compiled
# If you have a specific target for ej2_part2 in your Makefile, you might want to change 'make' to 'make ej2_part2'
# For now, assuming 'make' compiles both or the relevant one.
make clean && make

# Matrix dims and block dims
ROWS=${1:-1024}
COLS=${2:-1024}
# ej2_part2.cu takes blockX and blockY as 3rd and 4th args
# Defaulting them here as well, similar to ej2_part1.sh
BLOCKX=${3:-32}
BLOCKY=${4:-32}

TAG="${ROWS}x${COLS}_${BLOCKX}x${BLOCKY}"
# Construct the execution command with all arguments
EXEC="./ej2_part2 ${ROWS} ${COLS} ${BLOCKX} ${BLOCKY}"

OUTPUT_DIR="output/ej2_part2_results" # Specific subdir for part2 results
NVPROF_LOG="${OUTPUT_DIR}/NVPROF_LOG"
NVPROF_FULL="${OUTPUT_DIR}/NVPROF_FULL"
NSYS_REPORT="${OUTPUT_DIR}/NSYS_REPORT"
NSYS_LOG="${OUTPUT_DIR}/NSYS_LOG"
mkdir -p "$NVPROF_LOG" "$NVPROF_FULL" "$NSYS_REPORT" "$NSYS_LOG"

echo "Iniciando profiling para ej2_part2 con matriz ${ROWS}x${COLS} y bloque ${BLOCKX}x${BLOCKY}..."

# Run once without profiling to check basic execution (output to /dev/null)
echo "--- Initial execution (output suppressed) ---"
$EXEC > /dev/null
echo "Initial execution completed."

echo "--- nvprof metrics (gld_efficiency, gst_efficiency) ---"
nvprof --metrics gld_efficiency,gst_efficiency --log-file "${NVPROF_LOG}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof metrics log: ${NVPROF_LOG}/run_${TAG}.nvprof.log"

echo "--- nvprof full profile (text log) ---"
nvprof --log-file "${NVPROF_FULL}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof full log: ${NVPROF_FULL}/run_${TAG}.nvprof.log"

echo "--- nvprof normal output (binary .nvprof) ---"
nvprof -o "${OUTPUT_DIR}/nvprof_normal_${TAG}.nvprof" $EXEC
echo "nvprof normal binary output: ${OUTPUT_DIR}/nvprof_normal_${TAG}.nvprof"

echo "--- nvprof ALL METRICS (binary .nvprof and text log) ---"
# This is the one you originally had, now with specific logging
nvprof --metrics all --log-file "${NVPROF_LOG}/run_${TAG}_allmetrics.nvprof.log" -o "${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof" $EXEC
echo "nvprof all metrics log: ${NVPROF_LOG}/run_${TAG}_allmetrics.nvprof.log"
echo "nvprof all metrics binary output: ${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof"

echo "--- nsys report & log ---"
# Note: ej2_part2.cu takes rows, cols, blockX, blockY as args.
# The EXEC variable already includes these.
nsys profile --stats=true \
     -o "${NSYS_REPORT}/run_part2_${TAG}.nsys-rep" \
     --force-overwrite true \
     $EXEC &> "${NSYS_LOG}/run_part2_${TAG}.nsys.log"
echo "nsys report: ${NSYS_REPORT}/run_part2_${TAG}.nsys-rep"
echo "nsys log   : ${NSYS_LOG}/run_part2_${TAG}.nsys.log"

echo "Todas las ejecuciones de profiling para ej2_part2 completadas. Logs en ${OUTPUT_DIR}" 