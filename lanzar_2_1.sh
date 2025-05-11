#!/bin/bash
#SBATCH --job-name=ej2_part1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

set -e

export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

make clean && make

ROWS=${1:-1024}
COLS=${2:-1024}
BLOCKX=32
BLOCKY=32
TAG="${ROWS}x${COLS}_${BLOCKX}x${BLOCKY}"
EXEC="./ej2_part1 ${ROWS} ${COLS} ${BLOCKX} ${BLOCKY}"

OUTPUT_DIR="output"
NVPROF_LOG="${OUTPUT_DIR}/NVPROF_LOG_EJ2"
NVPROF_FULL="${OUTPUT_DIR}/NVPROF_FULL_EJ2"
NSYS_REPORT="${OUTPUT_DIR}/NSYS_REPORT_EJ2"
NSYS_LOG="${OUTPUT_DIR}/NSYS_LOG_EJ2"
mkdir -p "$NVPROF_LOG" "$NVPROF_FULL" "$NSYS_REPORT" "$NSYS_LOG"

echo "Iniciando profiling para matriz de ${ROWS}x${COLS} con bloque 32x32..."

$EXEC > /dev/null

echo "-- nvprof metrics --"
nvprof --metrics gld_efficiency,gst_efficiency --log-file "${NVPROF_LOG}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof metrics log: ${NVPROF_LOG}/run_${TAG}.nvprof.log"

echo "-- nvprof full profile --"
nvprof --log-file "${NVPROF_FULL}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof full log: ${NVPROF_FULL}/run_${TAG}.nvprof.log"

echo "-- nvprof normal output (default .nvprof binary) --"
nvprof -o "${OUTPUT_DIR}/nvprof_normal_${TAG}.nvprof" $EXEC
echo "nvprof normal binary output: ${OUTPUT_DIR}/nvprof_normal_${TAG}.nvprof"

# Optionally, also print to terminal and save as text
# nvprof $EXEC | tee "${OUTPUT_DIR}/nvprof_normal_${TAG}.txt"
# echo "nvprof normal output: ${OUTPUT_DIR}/nvprof_normal_${TAG}.txt"

echo "-- nvprof ALL METRICS (this may take a long time!) --"
nvprof --metrics all --log-file "${NVPROF_LOG}/run_${TAG}_allmetrics.nvprof.log" -o "${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof" $EXEC
echo "nvprof all metrics log: ${NVPROF_LOG}/run_${TAG}_allmetrics.nvprof.log"
echo "nvprof all metrics binary output: ${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof"

echo "-- nsys report & log --"
sys profile --stats=true \
     -o "${NSYS_REPORT}/run_${TAG}.nsys-rep" \
     --force-overwrite true \
     $EXEC &> "${NSYS_LOG}/run_${TAG}.nsys.log"
echo "nsys report: ${NSYS_REPORT}/run_${TAG}.nsys-rep"
echo "nsys log   : ${NSYS_LOG}/run_${TAG}.nsys.log"

echo "Todas las ejecuciones de profiling completadas. Logs en ${OUTPUT_DIR}" 