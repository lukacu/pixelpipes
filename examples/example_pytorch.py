import sys
import warnings
warnings.simplefilter("ignore", UserWarning)

from pixelpipes.list import RandomListElement
from pixelpipes.graph import GraphBuilder, Output
from pixelpipes.image import ConstantImageList, ImageNoise, Resize
from pixelpipes.sink import PipelineDataLoader

try:
    import torch
    import torchvision
except ImportError:
    print("This demo requires PyTorch")
    sys.exit(1)

class Dataset():
    dataset = torchvision.datasets.MNIST(
        root='./datasets/data',
        train=True,
        download=True).data.numpy()

    def to_list(self) -> list:
        print("Converting dataset to List(Numpy)...")
        return [self.dataset[i,:,:] for i in range(self.dataset.shape[0])]

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
        cv2.imwrite("img/pytorch/example_{}.png".format(index), batch[0][0])
