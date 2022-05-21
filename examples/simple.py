from itertools import count

from pixelpipes.list import ConstantList, RandomListElement
from pixelpipes.utilities import pipeline

@pipeline
def simple():
    a = ConstantList([1, 2, 3, 7, 8, 9, 4, 5, 6])
    return RandomListElement(a)

p = simple()

for i in count(1):
    print(p.run(i)[0])