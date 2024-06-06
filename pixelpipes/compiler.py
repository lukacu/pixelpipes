

from functools import reduce as _reduce
from collections.abc import Container
import typing
import numbers

from .graph import Graph, Graph, InferredReference, Macro, Node, NodeException, Operation, Reference, ValidationException, Constant, Copy, Debug, Output
from . import types
from .numbers import Constant
from .graph import RandomSeed, SampleIndex, Node, Operation
from .flow import Conditional
from pixelpipes import Pipeline, PipelineOperation

from attributee import String, Number


class CompilerException(Exception):
    pass


class Variable(Node):
    """Variable placeholder that can be overriden later"""

    name = String()
    default = Number()


def toposort(data):
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
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {item: (dep - ordered)
                for item, dep in data.items() if item not in ordered}
    if len(data) != 0:
        raise CompilerException('Cyclic dependency detected among nodes: {}'.format(
            ', '.join(repr(x) for x in data.keys())))


def infer_type(node: typing.Union[Reference, str], graph: Graph = None, type_cache: typing.Mapping[str, types.Data] = None) -> types.Data:
    """Computes output type for a given node by recursively computing types of its dependencies and
    calling validate method of a node with the information about their computed output types.

    Args:
        node (typing.Union[Reference, typing.Type[Node]]): Reference of the node or raw value
        graph (Graph): Mapping of all nodes in the graph
        type_cache (typing.Mapping[str, types.Type], optional): Optional cache for already computed types. Makes repetititve calls much faster. Defaults to None.

    Raises:
        ValidationException: Contains information about the error during node validation process.

    Returns:
        types.Type: Computed type for the given node.
    """

    if graph is None:
        graph = Graph.default()

    if isinstance(node, Node):
        name = graph.reference(node)
    elif isinstance(node, Reference):
        name = node
    else:
        return None

    if type_cache is not None and name in type_cache:
        return type_cache[name]

    if name not in graph:
        raise ValidationException("Node reference {} not found".format(name))

    node = graph[name]

    if not hasattr(node, "evaluate"):
        raise ValidationException(
            "Node {} does not implement evaluate method".format(node))
        #return None

    input_types = {}
    for k, i in zip(node.input_names(), node.input_values()):
        typ = infer_type(i, graph, type_cache)
        if typ is None:
            return None
        input_types[k] = typ

    output_type = node.evaluate(**input_types)

    if type_cache is not None and output_type is not None:
        type_cache[name] = output_type
    return output_type


class Compiler(object):
    """Compiler object contains utilities to validate a graph and compiles it to a pipeline
     (a sequence of operations, written in native code) that can be executed to obtain output
     variables.
    """

    @staticmethod
    def build_graph(graph: typing.Union[Graph, typing.Mapping[str, Node]],
                    variables: typing.Optional[typing.Mapping[str,
                                                              numbers.Number]] = None,
                    output: typing.Optional[str] = None) -> Pipeline:
        compiler = Compiler()
        return compiler.build(graph, variables, output)

    def __init__(self, debug=False):
        """[summary]

        Args:
            debug (bool, optional): Print a lot of debug messages. Defaults to False.
        """
        self._debug_enabled = debug

    def validate(self, graph: typing.Union[Graph, typing.Mapping[str, Node]]):
        """Validates graph by interring input and output types for all nodes. An exception will be
        thrown if dependencies cannot be resolved or if output of a node is not compatible with
        an input specification of a dependant node.

        Args:
            graph (typing.Mapping or Graph): Graph representation

        Raises:
            ValidationException: Different validation errors share this exception type

        Returns:
            dict: resolved types of all nodes
        """

        if isinstance(graph, Graph):
            nodes = graph.nodes()
        elif isinstance(graph, typing.Mapping):
            nodes = graph
        else:
            raise ValidationException("Illegal graph specification")

        type_cache = {}

        outputs = []

        for k in nodes.keys():
            infer_type(Reference(k), nodes, type_cache)
            if isinstance(nodes[k], Output):
                outputs.append(k)

        if self._debug_enabled:
            for name, node in nodes.items():
                self._debug("Node {} ({}), inferred type: {}", name,
                            node.__class__.__name__, type_cache[name])

        for output in outputs:
            inputs = [infer_type(i, nodes, type_cache)
                      for i in nodes[output].input_values()]

            for i, output_type in enumerate(inputs):
                if not isinstance(output_type, types.Token):
                    raise ValidationException("Output {} is a non-primitive type for output node {}: {}".format(
                        i, output, output_type), output)

        return type_cache

    def build(self, graph: Graph,
              variables: typing.Optional[typing.Mapping[str,
                                                        numbers.Number]] = None,
              output: typing.Optional[typing.Union[Container, typing.Callable]] = None, optimize: bool = None) -> Pipeline:
        """Compiles the graph and builds a pipeline from it in one function.

        Args:
            graph (Graph): _description_
            variables (typing.Optional[typing.Mapping[str, numbers.Number]], optional): _description_. Defaults to None.
            output (typing.Optional[typing.Union[Container, typing.Callable]], optional): _description_. Defaults to None.
            optimize (bool, optional): Optimize conditional operations by inserting jumps into the pipeline and using cache.

        Returns:
            Pipeline: Pipeline object
        """
        import datetime

        pipeline = Pipeline(self.compile(
            graph, variables, output), optimize=optimize)
        pipeline.metadata["timestamp"] = datetime.datetime.now().isoformat()
        return pipeline

    def compile(self, graph: Graph,
                variables: typing.Optional[typing.Mapping[str,
                                                          numbers.Number]] = None,
                output: typing.Optional[typing.Union[Container, typing.Callable]] = None) -> typing.Iterable[PipelineOperation]:
        """Compile a graph into a pipeline of native operations.

        Args:
            graph (Graph): Graph representation

        Raises:
            CompilerException: raised if graph is not valid

        Returns:
            engine.Pipeline: resulting pipeline
        """

        if Graph.default() is not None:
            raise CompilerException(
                "Cannot compile within graph scope")

        if not isinstance(graph, Graph):
            raise CompilerException("Illegal graph specification")

        # Copy the nodes mapping structure
        graph = graph.copy()

        self._debug("Compiling {} source nodes", len(graph))

        for name, node in graph:
            if isinstance(node, Variable):
                if variables is not None and node.name in variables:
                    value = variables[node.name]
                else:
                    value = node.default
                graph.replace(name, Constant(value=value, _auto=False))

        inferred_types = {}

        def normalize_input(input_value):
            if not isinstance(input_value, Reference):
                input_value = Constant(value=input_value)
            elif input_value.name == "[random]":
                input_value = RandomSeed()
            elif input_value.name == "[sample]":
                input_value = SampleIndex()
            else:
                return input_value, infer_type(input_value, graph, inferred_types)
            return input_value, infer_type(input_value, None, inferred_types)

        def normalize_inputs(node):
            inputs = {}
            change = False
            with graph.subgraph() as subgraph:
                for input_name, input_value in zip(node.input_names(), node.input_values()):
                    input_value, input_type = normalize_input(input_value)
                    if input_type is None:
                        return None, False
                    
                    if isinstance(input_value, Node):
                        change = True
                        input_value = subgraph.reference(input_value)
                        self._debug("Generating new node {}", input_value)
                    inputs[input_name] = InferredReference(
                        input_value, input_type)
                subgraph.commit()
            return inputs, change

        def expand_macro(name, node):
            self._debug("Expanding macro {} ({})", name, node)

            

            inputs, _ = normalize_inputs(node)
            if inputs is None:
                return False

            # Evaluate in macros does nothing, but calling it in unit tests allows us to monitor which macros are
            node.evaluate(**inputs)

            try:
                #node.evaluate(**{k: v.type for k, v in inputs.items()})
                # Expand the macro subgraph
                with graph.subgraph(prefix=Reference(name)) as macro_builder:
                    with node.context:
                        output = node.expand(**inputs)
                        subgraph = macro_builder.nodes()

            except Exception as ee:
                if isinstance(ee, NodeException):
                    ee.print_nodestack()
                raise CompilerException(
                    "Exception during macro {} ({}) expansion".format(name, node)) from ee

            self._debug("Expanded macro {} to {} nodes",
                        name, len(subgraph))

            output_reference = macro_builder.reference(output)

            if Reference(name) in macro_builder:
                if not name == output_reference.name:
                    raise NodeException(
                        "Node naming convention violation for output node in macro {}.".format(name), node=name)

            # delete original macro node, replace it with the subgraph
            graph.remove(node)

            for subname, subnode in subgraph.items():
                if subname == output_reference.name:
                    subname = name
                elif not subname.startswith(name.name + "."):
                    raise NodeException("Expanded node has illegal name {}, must start with {}.".format(
                        subname, name), node=name)
                subnode._origin = node
                graph.add(subnode, subname)
            return True

        while True:
            expanded = True
            changes = 0
            node_pairs = list(iter(graph))
            for name, node in node_pairs:
                if isinstance(node, Macro):
                    expanded = False
                    if expand_macro(name, node):
                        changes += 1
                else:
                    inputs, change = normalize_inputs(node)
                    if change:
                        graph.replace(name, node.duplicate(**inputs))
                        changes += 1
                    infer_type(name, graph, inferred_types)

            if not expanded and changes == 0:
                raise CompilerException(
                    "Unable to expand graph, probably due to some misbehaving nodes")
                    
            if expanded and changes == 0:
                break

        expanded = graph.nodes()

        self._debug("Expanded graph to {} nodes", len(expanded))

        output_nodes = []

        values = dict()
        aliases = dict()

        if output is None:
            def include_output(x): return True
        elif isinstance(output, Container):
            def include_output(x): return x in output
        elif isinstance(output, str):
            def include_output(x): return x == output
        else:
            include_output = output

        for name, node in expanded.items():
            if isinstance(node, Output):
                if include_output(node.label):
                    output_nodes.append(name)
            elif isinstance(node, Copy):
                aliases[aliases.get(name, name)] = aliases.get(
                    node.source.name, node.source.name)
            elif isinstance(node, Debug) and not self._debug_enabled:
                aliases[aliases.get(name, name)] = aliases.get(
                    node.source.name, node.source.name)
            elif isinstance(node, Constant):
                key = node.key()
                if key is not None:
                    if key in values:
                        aliases[aliases.get(name, name)] = values[key]
                    else:
                        values[key] = aliases.get(name, name)

        if not output_nodes:
            raise CompilerException("No output selected or available")

        builder = Graph()

        for name, node in expanded.items():
            if name not in aliases:
                inputs = {}
                for input_name, input_value in zip(node.input_names(), node.input_values()):
                    inputs[input_name] = Reference(
                        aliases.get(input_value.name, input_value))
                builder.add(node.duplicate(**inputs), name)

        optimized = builder.nodes()

        self._debug("Optimized graph to {} nodes", len(optimized))

        def traverse(nodes, start, depthfirst=True, stack=None):
            if not start in nodes:
                return
            if depthfirst:
                if stack is None:
                    yield start
                else:
                    yield start, stack + [start]
                    stack = stack + [start]
                for node in nodes[start].input_values():
                    yield from traverse(nodes, node.name, True, stack)
            else:
                if stack is not None:
                    stack = stack + [start]
                for node in nodes[start].input_values():
                    yield from traverse(nodes, node.name, False, stack)
                if stack is None:
                    yield start
                else:
                    yield start, stack

        dependencies = {}
        conditions = []

        for output_node in output_nodes:
            for name in traverse(optimized, output_node):
                node = optimized[name]
                if not isinstance(node, Operation):
                    raise ValidationException(
                        "Only operations allowed here, got {}".format(node), node=node)
                if isinstance(node, Conditional):
                    # In case of a conditional node we can determine which nodes will be executed only
                    # in certain conditions and insert jumps into the pipeline to speed up execution.

                    # Only register conditon once, do not use sets to preserve order
                    if not node.condition.name in conditions:
                        conditions.append(node.condition.name)
                    # List of nodes required by branch true
                    tree_true = set(traverse(optimized, node.true.name))
                    # List of nodes required by branch false
                    tree_false = set(traverse(optimized, node.false.name))
                    # List of nodes required to process condition
                    tree_condition = set(
                        traverse(optimized, node.condition.name))
                    # Required by both branches (A - B) + C
                    common = tree_true.intersection(
                        tree_false).union(tree_condition)

                    if node.condition.name not in common:
                        for sub_node in tree_true.difference(common):
                            dependencies.setdefault(
                                sub_node, set()).add(node.condition.name)
                        for sub_node in tree_false.difference(common):
                            dependencies.setdefault(
                                sub_node, set()).add(node.condition.name)

                dependencies.setdefault(name, set()).update(
                    [node.name for node in optimized[name].input_values()])

                for node in optimized[name].input_values():
                   if isinstance(optimized[node.name], Output):
                        raise ValidationException(
                            "Output node cannot be used as an input", node=optimized[node.name])

        self._debug("Resolving dependencies")

        # Transform conditions to name-index map
        conditions = {v: i for i, v in enumerate(conditions)}

        def operation_data(node):
            meta = node.operation()
            if not isinstance(meta, (tuple, list)):
                raise CompilerException(
                    "Illegal operation data for node: %s" % node)
            return meta

        # Retain correct order of output nodes
        for i, name in enumerate(output_nodes):
            dependencies.setdefault(name, set()).update(output_nodes[0:i])

        # Do not process jumps, just sort operations according to their dependencies
        ordered = [(level, item) for level, sublist in enumerate(
            toposort(dependencies)) for item in sublist]
        ordered = sorted(ordered)
        operations = [(name, operation_data(optimized[name]), [
            r.name for r in optimized[name].input_values()]) for _, name in ordered]

        pipeline_operations = []

        # Assemble a list of pipeline operations
        for i, (name, data, inputs) in enumerate(operations):
            self._debug("#{}: {} ({}), args: {}, inputs: {}",
                        i, data[0], inferred_types[name], data[1:], inputs)
            pipeline_operations.append(PipelineOperation(
                name, data[0], list(data[1:]), inputs))

        return pipeline_operations

    def _debug(self, message: str, *args, **kwargs):
        if self._debug_enabled:
            print(message.format(*args, **kwargs))
