import os
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from pathlib import Path
from time import time
from typing import Iterable

import ecgdigitize
from ecgdigitize.otsu import otsuThreshold
import ecgdigitize.visualization as viz
from ecgdigitize.image import openImage


myImage = openImage(Path('data/images/002.JPG'))

grayscale = myImage.toGrayscale()
print(grayscale)
print(grayscale.normalized())