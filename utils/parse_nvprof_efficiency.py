import os
import re
import glob

# Directory containing the nvprof logs
LOG_DIR = 'output/NVPROF_LOG'

# Pattern to extract block size from filename
FILENAME_PATTERN = re.compile(r'run_\d+x\d+_(\d+x\d+)\.nvprof\.log$')

# Pattern to extract efficiency metrics from file
GLD_PATTERN = re.compile(r'gld_efficiency\s+.*?(\d+\.\d+)%')
GST_PATTERN = re.compile(r'gst_efficiency\s+.*?(\d+\.\d+)%')

print('filename,block_size,gld_efficiency,gst_efficiency')

for filepath in sorted(glob.glob(os.path.join(LOG_DIR, '*.nvprof.log'))):
    filename = os.path.basename(filepath)
    block_size_match = FILENAME_PATTERN.search(filename)
    block_size = block_size_match.group(1) if block_size_match else ''
    gld_eff = ''
    gst_eff = ''
    with open(filepath, 'r') as f:
        for line in f:
            if 'gld_efficiency' in line:
                m = GLD_PATTERN.search(line)
                if m:
                    gld_eff = m.group(1)
            if 'gst_efficiency' in line:
                m = GST_PATTERN.search(line)
                if m:
                    gst_eff = m.group(1)
    print(f'{filename},{block_size},{gld_eff},{gst_eff}') 