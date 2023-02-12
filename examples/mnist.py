import os
import struct
from itertools import count
import cv2

import numpy as np

from pixelpipes.graph import RandomSeed, Constant
from pixelpipes.list import GetRandom
from pixelpipes.image import ConvertDepth
from pixelpipes.utilities import pipeline

@pipeline()
def mnist():
    root = os.path.dirname(__file__)
    images_file = os.path.join(root, "_data", "mnist", "train-images.idx3-ubyte")
    labels_file = os.path.join(root, "_data", "mnist", "train-labels.idx1-ubyte")

    images = []
    labels = []
    with open(images_file, mode="rb") as h:
        h.read(4) # Magic number
        count, height, width = struct.unpack(">III", h.read(12))
        for _ in range(count):
            images.append(np.frombuffer(h.read(width * height), dtype=np.uint8).reshape(height, width))

    with open(labels_file, mode="rb") as h:
        h.read(4) # Magic number
        count, = struct.unpack(">i", h.read(4))
        for _ in range(count):
            labels.append(struct.unpack("B", h.read(1))[0])

        assert len(images) == len(labels)

    # Pipeline starts here

    l = Constant(labels)
    i = Constant(images)
    s = RandomSeed() # Both label and image should be sampled the same way, we are binding the same random seed

    return ConvertDepth(GetRandom(i, seed=s), "Float"), GetRandom(l, seed=s)

if __name__ == "__main__":

    # Download original train or test files from http://yann.lecun.com/exdb/mnist/
    # unzip them and point the paths below to the final files

    stream = mnist()

    for image, label in stream:
        print(label)
        cv2.imshow("Patch", (image * 255).astype(np.uint8))
        if cv2.waitKey() != 32:
            break

