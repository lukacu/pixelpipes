
from concurrent.futures import ProcessPoolExecutor
import os
import random
import tempfile
from threading import Thread
import time
import numpy as np
import cv2 as cv

from pixelpipes.graph import Graph
from pixelpipes.compiler import Compiler, Conditional
from pixelpipes.core import Output
from pixelpipes.core.numbers import NormalDistribution, UniformDistribution
from pixelpipes.core.flow import Switch
from pixelpipes.sink import AbstractDataLoader, PipelineDataLoader
from pixelpipes.resource import ExtractField, GetRandomResource, ImageDirectory
from pixelpipes.image.geometry import ViewImage
from pixelpipes.geometry.view import AffineView
from pixelpipes.image.augmentation import ImageBrightness

def prepare_dataset():

    dataset_dir = os.path.join(tempfile.gettempdir(), "pp_benchmark")
    os.makedirs(dataset_dir, exist_ok=True)

    for i in range(100):
        filename = os.path.join(dataset_dir, "%08d.jpg" % i)

        if os.path.isfile(filename):
            continue
        
        img = np.random.rand(600, 400, 3)

        cv.imwrite(filename, img)

    return dataset_dir

def compile_graph(dataset_dir):

    with Graph() as builder:
 
        images = ImageDirectory(dataset_dir)

        x = UniformDistribution(0, 100)

        y = Conditional(true=x, false=x+2, condition=x > 0.5)

        image = ExtractField(GetRandomResource(images), "image")

        view = AffineView(x = NormalDistribution(0, 10), y = y)

        image = ImageBrightness(ViewImage(image, view, width=100, height=100), UniformDistribution(-10, 30) )

        Output(outputs=[image])

        graph = builder.build()

    return graph
    
dataset_dir = prepare_dataset()

def python_plain(_):

    i = random.randint(1, 99)

    image = cv.imread(os.path.join(dataset_dir, "%08d.jpg" % i))

    M = np.eye(2, 3, dtype=float)

    M[0, 2] = random.randint(-10, 20)
    M[1, 2] = random.randint(-10, 20)

    image = cv.warpAffine(image, M, (100, 100))

    image = image + random.randint(-10, 20)

    x = random.random()
    y = x + 2

    if x > 0.5:
        y = x

    return np.array([image]),

class PythonDataLoader(AbstractDataLoader):

    def _commit(self, index):
        return self._workers.submit(python_plain, index)



def wrap_worker(fn, *args):

    stop = False

    busy = 0.99

    ns = 1000 * 1000 * 1000

    def dummy_compute():

        i = 0
        while not stop:
            start = time.perf_counter_ns()

            while True:
                n = 0
                for x in range(10000):
                    n = n + x
                elapsed = time.perf_counter_ns() - start

                if elapsed / ns > busy:
                    break

            time.sleep(1 - elapsed / ns)

            i += elapsed / ns

    compute_worker = Thread(target=dummy_compute)
    compute_worker.start()

    res = fn(*args)

    stop = True
    compute_worker.join()

    return res


graph = compile_graph(dataset_dir)

B = 10

for N in [1, 10, 30]:

    print("N =", N)

    loader = PythonDataLoader(B, N)
    print("pythread =", wrap_worker(loader.benchmark, 10))

    processes = ProcessPoolExecutor(N)
    loader = PythonDataLoader(B, processes)
    print("pyproc =", wrap_worker(loader.benchmark, 10))

    loader = PipelineDataLoader(graph, B, N)
    print("pixpip =", wrap_worker(loader.benchmark, 10))


