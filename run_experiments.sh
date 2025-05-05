 #!/bin/bash

# Script to run the naive transpose kernel with different block sizes

# --- Configuration ---
ROWS=1024
COLS=1024
EXECUTABLE="./programa_part1"

# Define block sizes to test (pairs of BlockX BlockY)
# Ensure BlockX * BlockY <= 1024
declare -a BLOCK_SIZES=(
    "32 32"    # Default square
    "16 64"    # Taller block
    "64 16"    # Wider block
    "8 128"   # Even taller
    "128 8"   # Even wider
    "32 8"     # Smaller, power-of-2 sides
    "8 32"
    "16 16"
    "32 16"
    "16 32"
)

# --- Execution ---

# Check if executable exists
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or not executable."
    echo "Please compile ej1_part1.cu first, e.g.:"
    echo "  nvcc ej1_part1.cu -o $EXECUTABLE -lnvToolsExt"
    exit 1
fi

echo "Running experiments for ${ROWS}x${COLS} matrix..."

for block_pair in "${BLOCK_SIZES[@]}"; do
    # Split the pair into BLOCK_X and BLOCK_Y
    read -r BLOCK_X BLOCK_Y <<< "$block_pair"

    echo "--------------------------------------------------"
    echo "Running with Block Size: ${BLOCK_X} x ${BLOCK_Y}"
    echo "--------------------------------------------------"

    # Run the program
    $EXECUTABLE $ROWS $COLS $BLOCK_X $BLOCK_Y

    echo ""
    sleep 1
done

echo "All experiments completed."
