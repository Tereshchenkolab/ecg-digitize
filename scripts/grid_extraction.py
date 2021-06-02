from ecgdigitize.common import Failure
import os

from numpy.core.fromnumeric import mean, trace

if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from time import time
from typing import Optional
from pathlib import Path

import cv2
from cv2 import imread as loadImage
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import ecgdigitize
from ecgdigitize import visualization
from ecgdigitize.grid import detection as grid_detection
from ecgdigitize.grid import extraction as grid_extraction
from ecgdigitize.image import ColorImage, GrayscaleImage, openImage, saveImage

LEAD_PICTURES = [
    'lead-pictures/002-49853179.png',
    'lead-pictures/002-81476687.png',
    'lead-pictures/002-99471875.png',
    'lead-pictures/007-90692182.png',
    'lead-pictures/007-99788247.png',
    'lead-pictures/007-cropped.jpeg',
    'lead-pictures/029-12833928.png',
    'lead-pictures/029-38848327.png',
    'lead-pictures/029-62602410.png',
    'lead-pictures/040-2822995.png',
    'lead-pictures/040-46000898.png',
    'lead-pictures/040-8749883.png',
    'lead-pictures/1470_Page4-32601695.png',
    'lead-pictures/1470_Page4-38844414.png',
    'lead-pictures/1470_Page4-50738174.png',
    'lead-pictures/1470_Page4-75571614.png',
    'lead-pictures/1470_Page4-76759157.png',
    'lead-pictures/1470_Page4-985647.png',
    'lead-pictures/1475_Page2-11052757.png',
    'lead-pictures/1475_Page2-72414646.png',
    'lead-pictures/1475_Page2-95216965.png',
    'lead-pictures/413035413 B-13160186.png',
    'lead-pictures/413035413 B-39589090.png',
    'lead-pictures/I.png',
    'lead-pictures/II.png',
    'lead-pictures/SOHSU10121052013140_0001-50298946.png',
    'lead-pictures/SOHSU10121052013140_0001-51034552.png',
    'lead-pictures/SOHSU10121052013140_0001-56314289.png',
    'lead-pictures/SOHSU10121052013140_0001-60415631.png',
    'lead-pictures/SOHSU10121052013140_0001-96871515.png',
    'lead-pictures/SOHSU10121052013140_0003-24608258.png',
    'lead-pictures/SOHSU10121052013140_0003-31018692.png',
    'lead-pictures/SOHSU10121052013140_0003-51954989.png',
    'lead-pictures/SOHSU10121052013140_0003-54427551.png',
    'lead-pictures/aVL.png',
    'lead-pictures/fullscan-I.png',
    'lead-pictures/fullscan-II.png',
    'lead-pictures/fullscan-aVL.png',
    'lead-pictures/slighty-noisey-V2.png',
    'lead-pictures/slighty-noisey-aVL-small.png',
    'lead-pictures/slighty-noisey-aVL-tiny.png',
    'lead-pictures/slighty-noisey-aVL.png',
]

# image = loadImage("lead-pictures/slighty-noisey-aVL.png")
# image = loadImage("lead-pictures/007-cropped.jpeg")
# image = loadImage("lead-pictures/II.png")  # TODO: Work on handling this
# image = loadImage("lead-pictures/fullscan-II.png")  # Human-estimated grid size: 7.8791666667

acTimes = []
traceTimes = []

for path in LEAD_PICTURES:
    inputImage = openImage(Path(path))
    assert inputImage is not None

    # For visualization
    binary = grid_detection.allDarkPixels(inputImage)

    # Actual call
    gridSpacing = ecgdigitize.digitizeGrid(inputImage)

    print(gridSpacing, f"({path})")

    if gridSpacing is None:
        continue

    columnDensity = np.sum(binary.data, axis=0)
    maxStrength = max(columnDensity)
    minStrength = maxStrength / 2
    peaks, _ = scipy.signal.find_peaks(columnDensity, height=minStrength)
    firstColumn = peaks[0]

    output = inputImage.toGrayscale().toColor().data

    if not isinstance(gridSpacing, Failure):
        for column in np.arange(firstColumn, inputImage.width, gridSpacing):
            cv2.line(output, (round(column), 0), (round(column), inputImage.height-1), (85, 19, 248), thickness=1)

    saveImage(ColorImage(output), Path(f'validation/grid/{Path(path).name}'))
