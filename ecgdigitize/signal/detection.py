"""
detection.py
Created May 23, 2021

Converts a color image to binary mask of the lead's curve.
"""
import cv2

from .. import otsu
from ecgdigitize.image import BinaryImage, ColorImage
from .. import vision


def mallawaarachchi(image: ColorImage, useBlur: bool = False, invert: bool = True) -> BinaryImage:
    """The most straightforward implementation of binarization from Mallawaarachchi et. al., 2014"""

    # "The first [this] method tends to preserve significantly more information than the second does. For traces with minimal
    #  information, the first method will be more suitable. For newer traces, the second method [CIE-LAB color space] gives
    #  better results."
    # TODO: Implement CIE-LAB color space approach
    greyscaleImage = image.toGrayscale()

    # Apply blur to reduce noise (⚠️ not in the paper)
    if useBlur:
        blurredImage = vision.blur(greyscaleImage, kernelSize=3)
    else:
        blurredImage = greyscaleImage

    # Get the threshold using the method from Otsu
    binaryImage = blurredImage.toBinary(inverse=invert)

    return binaryImage


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



def adaptiveSignalIsolation(image: ColorImage) -> BinaryImage:
    maxHedge = 1
    minHedge = 0.6  # 0.5

    grayscaleImage = image.toGrayscale()
    otsuThreshold = otsu.otsuThreshold(grayscaleImage)

    hedging = float(maxHedge)
    binary = grayscaleImage.toBinary(otsuThreshold * hedging)

    while estimateFrequencyViaAutocorrelation(binary.data):
        hedging -= 0.05  # TODO: More intelligent choice of step
        if hedging < minHedge:
            break

        binary = grayscaleImage.toBinary(otsuThreshold * hedging)


    return binary

