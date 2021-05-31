import os
from typing import Iterable
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from pathlib import Path
import matplotlib.pyplot as plt


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


def printLiteralList(elements: Iterable):
    print('[')
    for element in elements:
        print(f"    '{element}',")
    print(']')

def listAvailableLeadPictures(asLiteralList: bool = False):
    leadPicturesDirectory = Path("./lead-pictures")
    paths = sorted(leadPicturesDirectory.iterdir())

    if asLiteralList:
        printLiteralList(paths)
    else:
        for path in paths:
            print(path)


if __name__ == "__main__":
    listAvailableLeadPictures()