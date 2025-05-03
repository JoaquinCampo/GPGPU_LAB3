#!/bin/bash
#SBATCH --job-name=pmppg34
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

module load cuda
make clean && make

# Default sizes
ROWS=${1:-1024}
COLS=${2:-1024}

# Array to collect GPU-activity times
times=()

for i in $(seq 1 10); do
  echo "=== Run $i ==="
  # Profile this run, write out a small report file
  REPORT="run${i}_${ROWS}x${COLS}.qdrep"
  nsys profile --stats=true -o "${REPORT}" ./programa "${ROWS}" "${COLS}" &> run${i}.log

  # Extract the “GPU activities” time (in ms) from the log
  t=$(grep "GPU activities" run${i}.log | awk '{print $3}')
  echo "GPU activities: ${t} ms"
  times+=("$t")
done

# Compute mean and stddev
python3 - <<EOF
import numpy as np
data = np.array([float(x) for x in ${times[@]}])
print()
print(f"Mean time : {data.mean():.3f} ms")
print(f"Stddev    : {data.std(ddof=0):.3f} ms")
EOF

echo
echo "All profiling runs complete. Reports: run1_*.qdrep … run10_*.qdrep"
