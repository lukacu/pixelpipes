from pixelpipes import Engine, load_module, types
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Graph, Input, Operation, SampleIndex, outputs


load_module("example")


class Triple(Operation):
    value = Input(types.Integer())

    def operation(self):
        return "example:example_triple",


with Graph() as graph:
    outputs(Triple(SampleIndex()))

pipeline = Compiler().build(graph)
print(Engine(pipeline, batch_size=1, first_index=7).next_batch()[0][0])  # 21
