from pixelpipes import Engine, load_module, types
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Graph, Input, Operation, SampleIndex, outputs


class Triple(Operation):
    value = Input(types.Integer())

    def operation(self):
        return "example:example_triple",


def main():
    load_module("example")

    with Graph() as graph:
        outputs(Triple(SampleIndex()))

    pipeline = Compiler().build(graph)
    value = Engine(pipeline, batch_size=1, first_index=7).next_batch()[0][0]
    print("triple(7):", value)  # 21


if __name__ == "__main__":
    main()
