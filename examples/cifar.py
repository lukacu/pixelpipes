
import os
import pickle
from itertools import count

import numpy as np
from attributee import String

from pixelpipes.resource.list import ResourceListSource, RandomResource

class CIFARDataset(ResourceListSource):

    directory = String(default=".")

    def load(self):

        image_batches = []
        label_batches = []

        for i in range(1, 5):
            file = os.path.join(self.directory, "data_batch_%d" % i)
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                image_batches.append(np.transpose(np.reshape(
                    dict[b"data"], (-1, 3, 32, 32)), (0, 2, 3, 1)))
                label_batches.append(np.array(dict[b"labels"], dtype=np.int32))

        return {"image": np.ascontiguousarray(np.concatenate(image_batches, 0)),
                "label": np.ascontiguousarray(np.concatenate(label_batches, 0))}

from pixelpipes.utilities import pipeline

@pipeline()
def cifar(directory):
    dataset = CIFARDataset(directory)
    sample = RandomResource(dataset)
    return sample["image"], sample["label"]


if __name__ == "__main__":

    import cv2

    # Download original dataset from http://www.cs.toronto.edu/~kriz/cifar.html
    # unzip it and point the paths below to the final files

    root = os.path.dirname(__file__)

    stream = cifar(os.path.join(root, "cifar"))

    for image, _ in stream:
        cv2.imshow("CIFAR", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if cv2.waitKey() != 32:
            break