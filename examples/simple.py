from itertools import count

from pixelpipes import Engine
from pixelpipes.graph import Constant
from pixelpipes.list import GetRandom
from pixelpipes.utilities import pipeline

@pipeline()
def simple():
    a = Constant([1, 2, 3, 7, 8, 9, 4, 5, 6])
    return GetRandom(a)

p = simple()
engine = Engine(p, batch_size=1, workers=4)

for i in count(1):
    print(engine.next_batch()[0][0])
