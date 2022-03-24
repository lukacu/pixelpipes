from pixelpipes.graph import GraphBuilder
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Output
from pixelpipes.numbers import UniformDistribution
from pixelpipes.flow import Switch

def compile_graph():

    width = 5

    def layer(depth):

        weights = [1] * width
        if depth == 0:
            inputs = [UniformDistribution(0, 1) for _ in range(width)]
        else:
            inputs = [layer(depth-1) for _ in range(width)]

        return Switch(inputs=inputs, weights=weights)

    with GraphBuilder() as builder:
 
        Output(outputs=[layer(4)])

        graph = builder.build()

    print("Graph created, compiling")

    return Compiler.compile_graph(graph)
    

import cProfile

cProfile.run('compile_graph()', '.profile.dat')


import pstats
from pstats import SortKey
p = pstats.Stats('.profile.dat')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()