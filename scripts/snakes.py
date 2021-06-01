#!$HOME/.pyenv/shims/python
# coding: utf-8

import os
if 'PYTHONPATH' not in os.environ:
    print("Error: Run `source env.sh` to be able to run `/scripts/*.py`")
    exit(1)

from collections import defaultdict
from pathlib import Path
from typing import Tuple
from time import time, sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval, polyvalfromroots

from ecgdigitize import otsu, vision, visualization, image, common
from ecgdigitize.image import BinaryImage, ColorImage, GrayscaleImage, openImage


def detectSignal(image: ColorImage, otsuHedging: int = 0.6, kernelSize: int = 3, erosions: int = 1, dilations: int = 1):
    grayscaleImage = image.toGrayscale()
    threshold = otsu.otsuThreshold(grayscaleImage) * otsuHedging
    _, binary = cv2.threshold(image.toGrayscale().data, threshold, 1, cv2.THRESH_BINARY_INV)

    return BinaryImage(denoise(binary, kernelSize, erosions, dilations))


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

    return dilated


def finiteDifferences(x: np.ndarray, order: int = 1, direction: str = 'forward'):
    assert direction in {'forward', 'backward', 'central'}

    deltas = x[1:] - x[:-1]
    outputs = np.zeros_like(x)

    for index in range(len(outputs)):
        if direction == 'forward':
            if index == len(x) - 1:
                continue
            outputs[index] = deltas[index]

        # outputs[index] =

    if order == 1:
        return outputs
    else:
        return derivative(outputs, order-1)


def derivative(x: np.ndarray, order: int = 1, width: int = 1) -> np.ndarray:
    deltas = x[1:] - x[:-1]
    outputs = np.zeros_like(x)

    for index in range(width, len(outputs) - width):
        outputs[index] = np.sum(deltas[(index - width):(index + width + 1)]) / len(deltas)

    # differences = np.diff(x)
    # outputs = np.append(differences, differences[-1:])
    # print(x.shape, outputs.shape)

    if order == 1:
        return outputs
    else:
        return derivative(common.padRight(outputs[1:], 1, fillValue=0), order-1)


def differentiateContour(xs: np.ndarray, width: int = 1) -> np.ndarray:
    firstOrder = np.zeros_like(xs)
    secondOrder = np.zeros_like(xs)
    thirdOrder = np.zeros_like(xs)
    fourthOrder = np.zeros_like(xs)

    for index in range(width, len(xs) - width):

        x = np.arange(-width, width + 1)
        y = xs[(index - width):(index + width + 1)]

        # print(y)

        f = Polynomial.fit(x, y, deg=4)
        f1 = f.deriv()
        f2 = f1.deriv()
        f3 = f2.deriv()
        f4 = f3.deriv()

        # xFine = np.arange(-width, width, step=0.001)

        # plt.plot(x, y)
        # plt.plot(xFine, polyval(xFine, f.coef))
        # plt.plot(xFine, polyval(xFine, f1.coef))
        # plt.show()

        firstOrder[index] = polyval(0, f1.coef)
        secondOrder[index] = polyval(0, f2.coef)
        thirdOrder[index] = polyval(0, f3.coef)
        fourthOrder[index] = polyval(0, f4.coef)

    # plt.plot(xs)
    # plt.plot(firstOrder)
    # plt.plot(secondOrder)
    # plt.plot(thirdOrder)
    # plt.plot(fourthOrder)
    # plt.show()

    return secondOrder, fourthOrder


def generateImageEnergy(image: BinaryImage) -> np.ndarray:
    outputs = np.full_like(image.data, image.height, dtype=float)

    for x in range(image.width):
        column = image.data[:,x]
        lastOnPixel = None
        # Scan up
        for y in common.reversedRange(image.height):
            if image.data[y][x] == 1:
                lastOnPixel = y
            elif lastOnPixel:
                outputs[y][x] = abs(y - lastOnPixel) ** 2 * -1
        lastOnPixel = None
        # Scan down
        for y in range(image.height):
            if image.data[y][x] == 1:
                outputs[y][x] = 0
                lastOnPixel = y
            elif lastOnPixel:
                outputs[y][x] = abs(lastOnPixel - y) ** 2

    return outputs / np.max(outputs)

def snakes(binaryImage: BinaryImage):
    alpha = 10
    beta = 20
    external = 1
    step = 5

    greyscale = binaryImage.toColor().toGrayscale()

    contour = np.array([
        float(binaryImage.height / 2) for _ in range(binaryImage.width)
    ])

    imageEnergy = generateImageEnergy(binaryImage)
    # plt.imshow(imageEnergy)
    # plt.show()

    for iteration in range(2):
        secondDerivate, fourthDerivative = finiteDifferences(contour, 2), finiteDifferences(contour, 4) # differentiateContour(contour, width=2) #

        F_cont = -alpha * secondDerivate  # derivative(contour, 2, width=10)
        F_curv = beta * fourthDerivative  # derivative(contour, 4, width=10)

        F_image = np.array([imageEnergy[int(y)][x] for x, y in enumerate(contour)]) # TODO: Smooth interpolation instead of int

        # Alternatively, the image forces can be normalized for each step such that the image
        # forces only update the snake by one pixel. ... This avoids the problem of dominating
        # internal energies that arise from tuning the time step.[5]
        # F_image_norm = weight.k * F_image ./ norm (F_image)

        F = F_cont + F_curv + (external * F_image)

        progress = greyscale.toColor()

        plt.plot(contour)
        plt.plot(secondDerivate * 20 + image.height // 2)
        plt.plot(fourthDerivative * 20 + image.height // 2)
        plt.show()

        # progress = visualization.overlaySignalOnImage(derivative(contour, 1, width=10) + image.height // 2, progress, lineWidth=1, color=(236,130,80))
        # progress = visualization.overlaySignalOnImage(derivative(contour, 2, width=10) + image.height // 2, progress, lineWidth=1, color=(30,236,220))

        # progress = visualization.overlaySignalOnImage(secondDerivate + image.height // 2, progress, lineWidth=1, color=(236,130,80))
        # progress = visualization.overlaySignalOnImage(fourthDerivative + image.height // 2, progress, lineWidth=1, color=(30,236,220))

        # progress = visualization.overlaySignalOnImage(F_image * 10 + image.height // 2, progress, lineWidth=1, color=(80,250,30))
        # # # progress = visualization.overlaySignalOnImage(F, progress, lineWidth=1, color=(30,236,180))
        # progress = visualization.overlaySignalOnImage(contour, progress, lineWidth=1)
        # visualization.displayImage(progress, title=str(iteration))

        contour = contour - (step * F)
        contour = np.clip(contour, 0, image.height - 1)

        if 'not converged': continue
        else              : break

# path = "lead-pictures/slighty-noisey-aVL-small.png"
# path = "lead-pictures/007-cropped.jpeg"
# path = "lead-pictures/II.png"
# path = "lead-pictures/slighty-noisey-aVL-tiny.png"
path = "lead-pictures/I.png"

image = openImage(Path(path))

binary = detectSignal(image, otsuHedging=0.6, erosions=0, dilations=0)

snakes(binary)

# inputs = np.array([0,5,6,10,8,3,0])
# finiteDifferences(inputs)

# plt.plot(inputs)
# plt.plot(np.diff(inputs))
# plt.plot(finiteDifferences(inputs))
# plt.plot(finiteDifferences(inputs, order=2))
# plt.plot(finiteDifferences(inputs, order=4))
# plt.show()