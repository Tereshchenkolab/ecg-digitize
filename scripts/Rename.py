import os
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

import glob
from pathlib import Path

for directory in glob.glob("./data/*/"):
    for index, file in enumerate(sorted(glob.glob(f"{directory}/*.txt"))):
        path = Path(file)
        newPath = path.with_name(f'{index+1}.txt')
        print(path, '->', newPath)
        path.rename(newPath)
