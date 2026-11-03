from pixelpipes import load_module, types
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Graph, Input, Operation, SampleIndex, outputs


load_module("example")


class Triple(Operation):
    value = Input(types.Integer())

    def operation(self):
        return "example:example_triple",


with Graph() as graph:
    outputs(Triple(SampleIndex()))

print(Compiler().build(graph).run(7)[0])  # 21
