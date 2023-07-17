import subprocess
import sys

script_name = 'tests/test_evogfuzz.py'
output_prefix = 'out'
n_iter = 30

for i in range(n_iter):
    subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)
