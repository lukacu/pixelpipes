from itertools import count

from pixelpipes.graph import Constant
from pixelpipes.list import RandomElement
from pixelpipes.utilities import pipeline

@pipeline()
def simple():
    a = Constant([1, 2, 3, 7, 8, 9, 4, 5, 6])
    return RandomElement(a)

p = simple()

for i in count(1):
    print(p.run(i)[0])