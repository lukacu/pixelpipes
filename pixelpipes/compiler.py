
import logging
import typing
import numbers

from functools import reduce as _reduce
from functools import partial

from pixelpipes import Graph, GraphBuilder, Reference, Output, Node, Macro, Copy
from pixelpipes.nodes import Constant, Variable
import pixelpipes.types as types
import pixelpipes.engine as engine

_logger = logging.getLogger(__name__)

def _toposort(data):
    """Dependencies are expressed as a dictionary whose keys are items
and whose values are a set of dependent items. Output is a list of
sets in topological order. The first set consists of items with no
dependences, each subsequent set consists of items that depend upon
items in the preceeding sets.
"""
    # Special case empty input.
    if len(data) == 0:
        return

    # Copy the input so as to leave it unmodified.
    data = data.copy()

    # Ignore self dependencies.
    for k, v in data.items():
        v.discard(k)
    # Find all items that don't depend on anything.
    extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
    # Add empty dependences where needed.
    data.update({item : set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}
    if len(data) != 0:
        raise ValueError('Cyclic dependency detected among nodes: {}'.format(', '.join(repr(x) for x in data.items())))


def _infer_type(node, nodes, type_cache=None, debug=False) -> types.Type:

    if not isinstance(node, Reference):
        if isinstance(node, int):
            return types.Integer(node)
        if isinstance(node, float):
            return types.Float(node)

    name = node.name

    if name not in nodes:
        raise ValueError("Node reference {} not found".format(node))
    if type_cache is not None and name in type_cache:
        return type_cache[name]

    try:
        input_types = {k: _infer_type(i, nodes, type_cache, debug) for k, i in zip(nodes[name].input_names(), nodes[name].input_values())}
        output_type = nodes[name].validate(**input_types)
        if debug:
            print("Validating node {}, inferred type: {}".format(name, output_type))
        if type_cache is not None:
            type_cache[name] = output_type
        return output_type
    except ValueError as ve:
        raise ValueError(node + ": " + str(ve)) from ve

class Compiler(object):
    """Compiler object contains utilities to validate a graph and compiles it to a pipeline
     (a sequence of operations, written in native code) that can be executed to obtain output
     variables.
    """

    def __init__(self, fixedout=False, debug=False):
        """[summary]

        Args:
            fixedout (bool, optional): Dimensions of all outputs should be fixed, making their concatenation possible.
                This is important when creating a batch dataset. Defaults to False.
            debug (bool, optional): Defaults to False.
        """
        self._fixedout = fixedout
        self._debug = debug

    def validate(self, graph: typing.Union[Graph, typing.Mapping[str, Node]]):
        """Validates graph by interring input and output types for all nodes. An exception will be
        thrown if dependencies cannot be resolved or if output of a node is not compatible with
        an input specification of a dependant node.

        Args:
            graph (typing.Mapping or Graph): Graph representation

        Raises:
            ValueError: Different validation errors share this exception type

        Returns:
            dict: resolved types of all nodes
        """
        if isinstance(graph, Graph):
            nodes = graph.nodes
        elif isinstance(graph, typing.Mapping):
            nodes = graph
        else:
            raise ValueError("Illegal graph specification")

        type_cache = {}

        outputs = []

        for k in nodes.keys():
            _infer_type(Reference(k), nodes, type_cache, self._debug)
            if isinstance(nodes[k], Output):
                outputs.append(k)

        output_types = None

        for output in outputs:
            inputs = [_infer_type(i, nodes, type_cache, self._debug) for i in nodes[output].input_values()]

            for i, output_type in enumerate(inputs):
                if self._fixedout and not output_type.fixed():
                    raise ValueError("Output {} not fixed for output node {}".format(i, output))
                if isinstance(output_type, types.Complex):
                    raise ValueError("Output {} is an complex type for output node {}".format(i, output))

            if output_types is None:
                output_types = inputs
            else:
                if len(output_types) != len(inputs):
                    raise ValueError("Unexpected number of inputs for output node {}".format(output))
                for out_a, out_b in zip(output_types, inputs):
                    if not out_a.castable(out_b):
                        raise ValueError("Incompatible inputs for output node {}".format(output))

        return type_cache

    def compile(self, graph: typing.Union[Graph, typing.Mapping[str, Node]],
            variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None) -> engine.Pipeline:
        """Compile a graph into a pipeline of native operations.

        Args:
            graph (typing.Union[Graph, typing.Dict]): Graph representation

        Raises:
            ValueError: raised if graph is not valid

        Returns:
            engine.Pipeline: resulting pipeline
        """

        if isinstance(graph, Graph):
            nodes = graph.nodes
        elif isinstance(graph, GraphBuilder):
            nodes = graph.nodes()
        elif isinstance(graph, typing.Mapping):
            nodes = graph
        else:
            raise ValueError("Illegal graph specification")

        # Copy the nodes mapping structure
        nodes = dict(**nodes)

        if self._debug:
            print("Compiling %d source nodes" % len(nodes))

        for name, node in nodes.items():
            if isinstance(node, Variable):
                if variables is not None and node.name in variables:
                    nodes[name] = Constant(value=variables[node.name])
                else:
                    nodes[name] = Constant(value=node.default)

        inferred_types = self.validate(nodes)

        builder = GraphBuilder()
        constants = GraphBuilder(prefix="_const")

        def normalize_input(node):
            inputs = {}
            for input_name, input_value in zip(node.input_names(), node.input_values()):
                if not isinstance(input_value, Reference):
                    constant_node = Constant(value=input_value, _auto=False)
                    inputs[input_name] = constants.add(constant_node)
                else:
                    inputs[input_name] = input_value

            return inputs

        def expand_macro(name, node):
            inputs = {}
            for input_name, input_value in zip(node.input_names(), node.input_values()):
                if not isinstance(input_value, Reference):
                    inputs[input_name] = (input_value, Constant.resolve_type(input_value))
                else:
                    inputs[input_name] = (input_value, inferred_types[input_value.name])

            subgraph = node.expand(inputs, name)

            if self._debug:
                print("Expanding macro node {} to {} nodes".format(name, len(subgraph)))

            for subname, subnode in subgraph.items():                
                if subname != name and not subname.startswith(name + "."):
                    raise ValueError("Expanded node has illegal name {}, must start with {}.".format(subname, name))
                if isinstance(subnode, Macro):
                    expand_macro(subname, subnode)
                else:
                    inputs = normalize_input(subnode)
                    builder.add(subnode.duplicate(**inputs), subname)

        for name, node in nodes.items():
            if isinstance(node, Macro):
                expand_macro(name, node)
            else:
                inputs = normalize_input(node)
                builder.add(node.duplicate(**inputs), name)

        builder += constants

        expanded = builder.nodes()

        if self._debug:
            print("Expanded graph to %d nodes" % len(expanded))

        self.validate(expanded)

        main_output = None

        values = dict()
        aliases = dict()

        for name, node in expanded.items():
            if isinstance(node, Output):
                if main_output is not None:
                    raise ValueError("Only one output node required")
                main_output = name
            elif isinstance(node, Copy):
                aliases[name] = node.source.name
            elif isinstance(node, Constant):
                if node.key() in values:
                    aliases[name] = values[node.key()]
                else:
                    values[node.key()] = name

        builder = GraphBuilder()

        for name, node in expanded.items():
            if name not in aliases:
                inputs = {}
                for input_name, input_value in zip(node.input_names(), node.input_values()):
                    inputs[input_name] = Reference(aliases.get(input_value.name, input_value))
                builder.add(node.duplicate(**inputs), name)

        optimized = builder.nodes()

        if self._debug:
            print("Optimized graph to %d nodes" % len(optimized))


        visited = set()

        def visit(node):
            for i in optimized[node].input_values():
                visit(i.name)
            visited.add(node)

        visit(main_output)

        dependencies = {}
        operations = {}

        for node in visited:
            dependencies[node] = set([i.name for i in optimized[node].input_values()])
            operations[node] = optimized[node].operation()

        if self._debug:
            print("Generated %d operations" % len(operations))
            print("Resolving dependencies")

        ordered = [(level, item) for level, sublist in enumerate(_toposort(dependencies)) for item in sublist]
        ordered = sorted(ordered)

        pipeline = engine.Pipeline()

        indices = {}

        for _, ref in ordered:
            inputs = [indices[r.name] for r in optimized[ref].input_values()]
            indices[ref] = pipeline.append(operations[ref], inputs)
            if self._debug:
                print("{}: {} ({})".format(indices[ref], operations[ref].__class__.__name__, ",".join([str(i) for i in inputs])))


        pipeline.finalize()
        return pipeline