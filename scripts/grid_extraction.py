import os
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from typing import Optional
import cv2
import matplotlib.pyplot as plt
from cv2 import imread as loadImage
import numpy as np
import scipy.signal

from digitize import Visualization
from digitize import Vision
from digitize.GridExtraction import estimateFrequencyViaAutocorrelation


# image = loadImage("lead-pictures/slighty-noisey-aVL.png")
image = loadImage("lead-pictures/007-cropped.jpeg")
# image = loadImage("lead-pictures/II.png")  # TODO: Work on handling this
# image = loadImage("lead-pictures/fullscan-II.png")  # Human-estimated grid size: 7.8791666667
# image = loadImage("data/SOHSU10121052013140_0003.tif")

assert image is not None
height, width, depth = image.shape

# Processing
greyscale = Vision.greyscale(image)
adjusted = Vision.adjustWhitePoint(greyscale, strength=1.0)
binary = Vision.binarize(adjusted, 230)

gridSpacing = estimateFrequencyViaAutocorrelation(binary)
print(gridSpacing)

columnDensity = np.sum(binary, axis=0)
maxStrength = max(columnDensity)
minStrength = maxStrength / 2
peaks, _ = scipy.signal.find_peaks(columnDensity, height=minStrength)
firstColumn = peaks[0]

output = image.copy()

for column in np.arange(firstColumn, width, gridSpacing):
    cv2.line(output, (round(column), 0), (round(column), height-1), (185, 248, 19), thickness=1)

Visualization.displayColorImage(output)