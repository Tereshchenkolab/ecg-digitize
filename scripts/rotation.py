

from pathlib import Path
from ecgdigitize.image import openImage
import ecgdigitize


image = openImage(Path('lead-pictures/002-49853179.png'))
print(ecgdigitize.estimateRotationAngle(image))