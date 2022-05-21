import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pixelpipes.compiler import Compiler
from pixelpipes.list import RandomListElement
from pixelpipes.graph import GraphBuilder, Output
from pixelpipes.image import ConstantImageList, ImageNoise, Resize
from pixelpipes.sink import PipelineDataLoader

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("This demo requires TensorFlow")
    sys.exit(1)

class Dataset():

    dataset = tfds.as_numpy(tfds.load(
            data_dir="./datasets/mnist",
            name="mnist",
            split="train"))

    def to_list(self) -> list:
        return [data["image"] for data in self.dataset]

if __name__ == "__main__":

    with GraphBuilder() as graph:
        n0 = ConstantImageList(source=Dataset().to_list())
        n1 = RandomListElement(source=n0)
        n2 = ImageNoise(source=n1, amount=0.25)  
        n3 = Resize(source=n2, width=28, height=28) # To avoid PipelineDataLoader error
        Output(n3)

    ITERATIONS = 10
    BATCH_SIZE = 1

    print("Creating DataLoader...")
    loader = PipelineDataLoader(graph.pipeline(), BATCH_SIZE, 1)

    print("Iterating...")
    import cv2
    for index, batch in enumerate(loader):
        if index == ITERATIONS:
            break
        cv2.imwrite("img/tensorflow/example_{}.png".format(index), batch[0][0])
