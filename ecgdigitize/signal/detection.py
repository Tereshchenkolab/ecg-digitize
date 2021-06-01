"""
detection.py
Created May 23, 2021

Converts a color image to binary mask of the lead's curve.
"""
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
