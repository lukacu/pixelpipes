import os

import cv2

from pixelpipes.flow import Switch
from pixelpipes.numbers import RandomBoolean, SampleUnform
from pixelpipes.resource.list import RandomResource
from pixelpipes.resource.loading import ImageDirectory
from pixelpipes.image.geometry import Resize, Flip, Scale
from pixelpipes.image.augmentation import ImageBrightness, ImageNoise, ImagePiecewiseAffine

from pixelpipes.utilities import pipeline, collage

@pipeline()
def stream():
    images = ImageDirectory(os.path.join(os.path.dirname(__file__), "images"))
    image = Scale(Resize(RandomResource(images)["image"], 160, 120), 0.4)

    return Switch([
        ImageBrightness(image, SampleUnform(-90, 90)),
        ImageNoise(image, SampleUnform(0, 0.1)),
        ImagePiecewiseAffine(image, 10, 10),
        Flip(image, RandomBoolean(), RandomBoolean()),
        image
    ], [0.3, 0.3, 0.3, 0.2, 0.2]),

samples = collage(stream(), index=0, rows=8, columns=20)

cv2.imshow("Samples", cv2.cvtColor(samples, cv2.COLOR_RGB2BGR))
cv2.waitKey()



