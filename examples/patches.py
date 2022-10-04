import os
from itertools import count

import cv2

from pixelpipes.resource.loading import ImageDirectory
from pixelpipes.resource.list import RandomResource
from pixelpipes.image.geometry import ViewImage, RandomPatchView
from pixelpipes.utilities import pipeline

@pipeline()
def stream():
    images = ImageDirectory(os.path.join(os.path.dirname(__file__), "images"))
    image = RandomResource(images)["image"]
    return ViewImage(image, RandomPatchView(image, 200, 200), 200, 200)

p = stream()

for i in count(1):
    image, = p.run(i)
    cv2.imshow("Patch", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if cv2.waitKey() != 32:
        break

