import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Dataset():

    import tensorflow_datasets as tfds
    dataset = tfds.as_numpy(tfds.load(
            data_dir="./datasets/tensorflow_datasets",
            name="mnist",
            split="train"))

    def to_list(self) -> list:
        print("Converting dataset to List(Numpy)...")
        return [data["image"] for data in self.dataset]

from relative_import import *

from pixelpipes.graph import Output
from pixelpipes.list import RandomListElement
from pixelpipes.graph import GraphBuilder
from pixelpipes.image import ConstantImageList, ImageNoise, Resize
from pixelpipes.sink import PipelineDataLoader

with GraphBuilder() as graph:
    n0 = ConstantImageList(source=Dataset().to_list())
    n1 = RandomListElement(source=n0)
    n2 = ImageNoise(source=n1, amount=0.25)  
    n3 = Resize(source=n2, width=28, height=28) # To avoid PipelineDataLoader error
    Output(outputs=[n3])

ITERATIONS = 10
BATCH_SIZE = 1

print("Creating DataLoader...")
loader = PipelineDataLoader(graph, BATCH_SIZE, 1)

print("Iterating...")
import cv2
for index, batch in enumerate(loader):
    if index == ITERATIONS:
        break
    cv2.imwrite("img/tensorflow/example_{}.png".format(index), batch[0][0])
