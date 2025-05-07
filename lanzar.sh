#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

set -e # Exit immediately if a command exits with a non-zero status.

# module load cuda   # uncomment if your cluster supports it
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# --- Configuration ---
EXECUTABLE="./ej1_part1" # Assuming the executable is named ej1_part1

# Matrix dims (use arguments or defaults)
# ROWS=${1:-1024} # Removed, fixed in C++ code
# COLS=${2:-1024} # Removed, fixed in C++ code

# Define block sizes to test (pairs of BlockX BlockY)
# Ensure BlockX * BlockY <= 1024
declare -a BLOCK_SIZES=(
    "128 128"
    "64 64"
    "32 32"
    "16 16"
    "8 8"
    "4 4"
)

# Prepare output folder
OUTPUT_DIR="output" # Keep fixed name or adjust as needed
mkdir -p "${OUTPUT_DIR}"

# --- Build Step ---
echo "=== Building Project ==="
# rm -f "${EXECUTABLE}" # Ensure recompilation (already done by direct nvcc overwrite)
# make clean && make ej1_part1 # Explicitly build the target --- Bypassing make for this target

echo "Compiling ${EXECUTABLE} directly..."
nvcc -O3 -arch=compute_50 -o "${EXECUTABLE}" ej1_part1.cu -lnvToolsExt
if [ $? -ne 0 ]; then
    echo "Error: Direct compilation of ${EXECUTABLE} failed."
    exit 1
fi

# We might still want to build other targets if needed, using make:
# echo "Building other make targets..."
# make other_target1 other_target2

# Check if executable exists after build
if [ ! -x "${EXECUTABLE}" ]; then
    echo "Error: Executable '$EXECUTABLE' not found after direct compilation."
    exit 1
fi

# --- Profiling Loop ---
echo "=== Starting Profiling Runs for fixed 1024x1024 ==="

for block_pair in "${BLOCK_SIZES[@]}"; do
    # Split the pair into BLOCK_X and BLOCK_Y
    read -r BLOCK_X BLOCK_Y <<< "$block_pair"

    echo "--------------------------------------------------"
    echo "Profiling with Block Size: ${BLOCK_X} x ${BLOCK_Y}"
    echo "--------------------------------------------------"

    # Define unique base name for output files for this configuration
    BASE_FILENAME="${OUTPUT_DIR}/run_1024x1024_${BLOCK_X}x${BLOCK_Y}" # Keep naming convention

    # Diagnostic: Run directly first to check argument parsing
    echo "Direct execution check: ${EXECUTABLE} ${BLOCK_X} ${BLOCK_Y}"
    ${EXECUTABLE} ${BLOCK_X} ${BLOCK_Y}
    echo "Direct execution finished."

    # Run nsys profile for the current configuration
    echo "Running nsys profile..."
    nsys profile --stats=true -o "${BASE_FILENAME}" --force-overwrite true ${EXECUTABLE} ${BLOCK_X} ${BLOCK_Y} &> "${BASE_FILENAME}.log"
    if [ $? -ne 0 ]; then
        echo "Warning: nsys profiling failed for ${BLOCK_X}x${BLOCK_Y}. Check ${BASE_FILENAME}.log"
    else
        echo "nsys report generated: ${BASE_FILENAME}.nsys-rep"
    fi
    echo ""

done

echo "All profiling runs completed. Results in ${OUTPUT_DIR}"