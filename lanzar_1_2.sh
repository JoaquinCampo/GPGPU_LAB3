#!/bin/bash
#SBATCH --job-name=mitrabajo
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

declare -a BLOCK_SIZES=(
    "32 32"
    "32 16"
    "32 8"
    "32 4"
    "16 16"
    "64 16"
    "128 8"
    "256 4"
    "512 2"
    "1024 1"
)

OUTPUT_DIR="output"
NVPROF_LOG="${OUTPUT_DIR}/NVPROF_LOG"
NSYS_REPORT="${OUTPUT_DIR}/NSYS_REPORT"
NSYS_LOG="${OUTPUT_DIR}/NSYS_LOG"
mkdir -p "${NVPROF_LOG}" "${NSYS_REPORT}" "${NSYS_LOG}"

echo "Iniciando profiling para matriz de ${ROWS}x${COLS}..."

for bs in "${BLOCK_SIZES[@]}"; do
    read -r BX BY <<< "$bs"
    TAG="${ROWS}x${COLS}_${BX}x${BY}"
    EXEC="./ej1_part1 ${ROWS} ${COLS} ${BX} ${BY}"

    echo "=== Probando bloque ${BX}x${BY} ==="
    ${EXEC} > /dev/null

    echo "-- nvprof log --"
    nvprof --metrics gld_efficiency,gst_efficiency --log-file "${NVPROF_LOG}/run_${TAG}.nvprof.log" ${EXEC}
    echo "nvprof log: ${NVPROF_LOG}/run_${TAG}.nvprof.log"

    echo "-- nsys report & log --"
    nsys profile --stats=true \
         -o "${NSYS_REPORT}/run_${TAG}.nsys-rep" \
         --force-overwrite true \
         ${EXEC} &> "${NSYS_LOG}/run_${TAG}.nsys.log"
    echo "nsys report: ${NSYS_REPORT}/run_${TAG}.nsys-rep"
    echo "nsys log   : ${NSYS_LOG}/run_${TAG}.nsys.log"

    echo

done

echo "Todas las ejecuciones de profiling completadas. Logs en ${OUTPUT_DIR}"
