#!$HOME/.pyenv/shims/python
# coding: utf-8

import os
from pathlib import Path
from typing import Tuple
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ecgdigitize import otsu, vision, visualization, image, common
from ecgdigitize.image import BinaryImage, ColorImage, openImage, saveImage
from ecgdigitize.signal.extraction import *
from ecgdigitize.grid.extraction import estimateFrequencyViaAutocorrelation


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


def detectSignal(image: ColorImage, otsuHedging: int = 0.6) -> BinaryImage:
    grayscaleImage = image.toGrayscale()
    threshold = otsu.otsuThreshold(grayscaleImage) * otsuHedging
    _, binary = cv2.threshold(image.toGrayscale().data, threshold, 1, cv2.THRESH_BINARY_INV)

    hedging = float(maxHedge)
    binary = detectSignal(image, otsuHedging=hedging, erosions=0, dilations=0)

    while estimateFrequencyViaAutocorrelation(binary.data):
        hedging -= 0.05  # TODO: More intelligent choice of step
        if hedging < minHedge:
            break

        binary = detectSignal(image, otsuHedging=hedging, erosions=0, dilations=0)


    return BinaryImage(binary)


def denoise(image: BinaryImage, kernelSize: int = 3, erosions: int = 1, dilations: int = 1) -> BinaryImage:
    eroded = image

    for _ in range(erosions):
        eroded = cv2.erode(
            eroded,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (kernelSize, kernelSize))
        )

    dilated = eroded

    for _ in range(dilations):
        dilated = cv2.dilate(
            dilated,
            cv2.getStructuringElement(cv2.MORPH_DILATE, (kernelSize, kernelSize))
        )

    return BinaryImage(dilated)


# TODO: Make score multiply, or normalize the score by the length of the path
def score(currentPoint: Tuple[int, int], candidatePoint: Tuple[int, int], candidateAngle: float):
    DISTANCE_WEIGHT = 1

    currentAngle = angleBetweenPoints(candidatePoint, currentPoint)
    angleValue = angleSimilarity(currentAngle, candidateAngle)
    distanceValue = distanceBetweenPoints(currentPoint, candidatePoint)

    return (distanceValue * DISTANCE_WEIGHT) + (angleValue * (1 - DISTANCE_WEIGHT))


def getAdjacent(pointsByColumn, bestPathToPoint, startingColumn, minimumLookBack):
    rightColumnIndex = startingColumn
    leftColumnIndex = common.lowerClamp(startingColumn-minimumLookBack, 0)

    result = common.flatten(pointsByColumn[leftColumnIndex:rightColumnIndex])

    while len(result) == 0 and leftColumnIndex >= 0:
        leftColumnIndex -= 1
        result = common.flatten(pointsByColumn[leftColumnIndex:rightColumnIndex])

    for point in result:
        x, y = point
        pointScore, _, pointAngle = bestPathToPoint[y][x]
        yield pointScore, point, pointAngle


def extractSignal(binary: BinaryImage):
    pointsByColumn = getPointLocations(binary.data)
    points = np.array(common.flatten(pointsByColumn))

    # plt.figure(figsize=(20,40))
    # plt.imshow(binary * -1 + np.full_like(binary, 1), cmap='Greys')
    # plt.scatter(points[:,1], points[:,0], s=1, c='violet')
    # plt.show()

    minimumLookBack = 1

    bestPathToPoint = defaultdict(dict)

    # TODO: Allow some leeway either (1) Initialize the first N columns with 0s or (2) Search until some threshold for seeding is met

    # Initialize the DP table with base cases (far left side)
    for column in pointsByColumn[:1]:
        for x,y in column:
            bestPathToPoint[y][x] = (0, None, 0)

    # Build the table
    for column in pointsByColumn[1:]:
        for x, y in column:
            # Gather all other points in the perview of search for the current point
            adjacent = list(getAdjacent(pointsByColumn, bestPathToPoint, y, minimumLookBack))

            if len(adjacent) == 0:
                bestPathToPoint[y][x] = (float('inf'), None, 0)
            else:
                bestScore, bestPoint = min(
                    [(score((x,y), candidatePoint, candidateAngle) + cadidateScore, candidatePoint)
                    for cadidateScore, candidatePoint, candidateAngle in adjacent]
                )
                bestPathToPoint[y][x] = (bestScore, bestPoint, angleBetweenPoints(bestPoint, (x,y)))

    # TODO: Search backward in some 2D area for the best path ?
    OPTIMAL_ENDING_WIDTH = 20
    optimalCandidates = getAdjacent(pointsByColumn, bestPathToPoint, startingColumn=image.width, minimumLookBack=OPTIMAL_ENDING_WIDTH)
    _, current = min([(totalScore, point) for totalScore, point, _ in optimalCandidates])

    print(current)

    bestPath = []

    while current is not None:
        bestPath.append(current)
        x, y = current
        _, current, _ = bestPathToPoint[y][x]

    signal = np.full(image.width, np.nan)

    for row, column in bestPath:
        signal[column] = row

    return signal


    scores = [bestPathToPoint[y][x][0] ** .5 for x,y in points]

    # # plt.imshow((binary * -1 + np.full_like(binary, 1)) * .1, cmap='Greys')
    # plt.imshow(image.toGrayscale().data, cmap='Greys')
    # plt.scatter(points[:,1], points[:,0], c=scores)
    # plt.plot(signal, c='purple')
    # plt.show()


maxHedge = 1
minHedge = 0.6  # 0.5

for path in LEAD_PICTURES:
    image = openImage(Path(path))


    print(str(path), '\t', hedging)

    # plt.imshow(binary * -1 + np.full_like(binary, 1), cmap='Greys')
    # plt.show()

    signal = extractSignal(binary)
    output = visualization.overlaySignalOnImage(signal, binary.toColor())


    saveImage(output, Path(f"validation/signal/{Path(path).name}"))


# TODO:
# - Treat all pixels as nodes? -- probably too slow
# - Add prior weights that incentivize nodes that are in the middle (horizontall and vertically) of other pixels to draw the line to the center of the signal trace