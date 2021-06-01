from typing import Optional
from ecgdigitize.image import ColorImage
from . import common
from .grid import detection as grid_detection
from . import vision


def estimateRotationAngle(image: ColorImage, houghThresholdFraction: float = 0.25) -> Optional[float]:
    binaryImage = grid_detection.thresholdApproach(image)

    houghThreshold = int(image.width * houghThresholdFraction)
    lines = vision.houghLines(binaryImage, houghThreshold)

    angles = common.mapList(lines, vision.houghLineToAngle)
    offsets = common.mapList(angles, lambda angle: angle % 90)
    candidates = common.filterList(offsets, lambda offset: abs(offset) < 30)

    if len(candidates) > 1:
        estimatedAngle = common.mean(candidates)
        return estimatedAngle
    else:
        return None