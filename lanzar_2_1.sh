#!/bin/bash
#SBATCH --job-name=pmppg34
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

OUTPUT_DIR="output/2_1"
NVPROF_LOG="${OUTPUT_DIR}/NVPROF_LOG_EJ2"
NVPROF_FULL="${OUTPUT_DIR}/NVPROF_FULL_EJ2"
NSYS_REPORT="${OUTPUT_DIR}/NSYS_REPORT_EJ2"
NSYS_LOG="${OUTPUT_DIR}/NSYS_LOG_EJ2"
mkdir -p "$NVPROF_LOG" "$NVPROF_FULL" "$NSYS_REPORT" "$NSYS_LOG"

echo "Iniciando profiling para matriz de ${ROWS}x${COLS} con bloque 32x32..."

# Warmup
$EXEC >/dev/null

echo "-- nvprof metrics --"
nvprof --metrics gld_efficiency,gst_efficiency --log-file "${NVPROF_LOG}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof metrics log: ${NVPROF_LOG}/run_${TAG}.nvprof.log"

echo "-- nvprof shared memory metrics --"
nvprof --metrics shared_bank_conflicts,shared_efficiency --log-file "${NVPROF_LOG}/run_${TAG}_shared.nvprof.log" $EXEC
echo "nvprof shared metrics log: ${NVPROF_LOG}/run_${TAG}_shared.nvprof.log"

echo "-- nvprof shared memory events --"
nvprof --events shared_load_bank_conflict,shared_store_bank_conflict --log-file "${NVPROF_LOG}/run_${TAG}_shared_events.nvprof.log" $EXEC
echo "nvprof shared events log: ${NVPROF_LOG}/run_${TAG}_shared_events.nvprof.log"

echo "-- nvprof log file --"
nvprof --log-file "${NVPROF_FULL}/run_${TAG}.nvprof.log" $EXEC
echo "nvprof full log: ${NVPROF_FULL}/run_${TAG}.nvprof.log"

echo "-- nvprof ALL METRICS --"
nvprof --metrics all -o "${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof" $EXEC
echo "nvprof all metrics output: ${OUTPUT_DIR}/nvprof_allmetrics_${TAG}.nvprof"

echo "-- nsys report & log --"
nsys profile --stats=true -o "${NSYS_REPORT}/run_${TAG}.nsys-rep" --force-overwrite true $EXEC &>"${NSYS_LOG}/run_${TAG}.nsys.log"
echo "nsys report: ${NSYS_REPORT}/run_${TAG}.nsys-rep"
echo "nsys log   : ${NSYS_LOG}/run_${TAG}.nsys.log"

echo "Todas las ejecuciones de profiling completadas. Logs en ${OUTPUT_DIR}"
