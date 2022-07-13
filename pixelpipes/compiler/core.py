

from collections.abc import Container
import typing
import numbers

from ..graph import Graph, Graph, InferredReference, Macro, Node, NodeException, Operation, Reference, ValidationException, Constant, Copy, Debug, Output
from .. import Pipeline, PipelineOperation, types
from ..numbers import Constant
from .utilities import BranchSet, toposort
from . import CompilerException, Variable
from ..graph import RandomSeed, SampleIndex
from ..flow import Conditional

def infer_type(node: typing.Union[Reference, str], graph: Graph = None, type_cache: typing.Mapping[str, types.Data] = None) -> types.Data:
    """Computes output type for a given node by recursively computing types of its dependencies and
    calling validate method of a node with the information about their computed output types.

    Args:
        node (typing.Union[Reference, typing.Type[Node]]): Reference of the node or raw value
        graph (Graph): Mapping of all nodes in the graph
        type_cache (typing.Mapping[str, types.Type], optional): Optional cache for already computed types.
            Makes repetititve calls much faster. Defaults to None.

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

    if not hasattr(node, "validate"):
        return None

    input_types = {}
    for k, i in zip(node.input_names(), node.input_values()):
        typ = infer_type(i, graph, type_cache) 
        if typ is None:
            return None
        input_types[k] = typ

    output_type = node.validate(**input_types)

    if type_cache is not None and output_type is not None:
        type_cache[name] = output_type
    return output_type


def print_graph(graph: Graph):
    print("========== START", len(graph), " nodes =========" )
    for ref, _ in graph:
        print(ref)
    print("========== END", len(graph), " nodes =========" )

class Compiler(object):
    """Compiler object contains utilities to validate a graph and compiles it to a pipeline
     (a sequence of operations, written in native code) that can be executed to obtain output
     variables.
    """

    @staticmethod
    def build_graph(graph: typing.Union[Graph, typing.Mapping[str, Node]], 
                      variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
                      output: typing.Optional[str] = None, fixedout: bool = False) -> Pipeline:
        compiler = Compiler(fixedout)
        return compiler.build(graph, variables, output)

    def __init__(self, fixedout=False, predictive=True, debug=False):
        """[summary]

        Args:
            fixedout (bool, optional): Dimensions of all outputs should be fixed, making their concatenation possible.
                This is important when creating a batch dataset. Defaults to False.
            predictive (bool, optional): Optimize conditional operations by inserting jumps into the pipeline.
            debug (bool, optional): Print a lot of debug messages. Defaults to False.
        """
        self._fixedout = fixedout
        self._debug_enabled = debug
        self._predictive = predictive

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
                if isinstance(output_type, types.Complex):
                    raise ValidationException("Output {} is a non-primitive type for output node {}: {}".format(
                        i, output, output_type), output)

        return type_cache

    def build(self, graph: Graph,
                variables: typing.Optional[typing.Mapping[str,
                                                          numbers.Number]] = None,
                output: typing.Optional[typing.Union[Container, typing.Callable]] = None) -> Pipeline:

        return Pipeline(self.compile(graph, variables, output))

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
                    value=variables[node.name]
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
                    inputs[input_name] = InferredReference(input_value, input_type)
                subgraph.commit()
            return inputs, change

        def expand_macro(name, node):
            self._debug("Expanding macro {} ({})", name, node)

            #print_graph(graph)

            inputs, _ = normalize_inputs(node)
            if inputs is None:
                return False

            try:
                node.validate(**{k: v.type for k, v in inputs.items()})
                # Expand the macro subgraph
                with graph.subgraph(prefix=Reference(name)) as macro_builder:
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
                    raise NodeException("Node naming convention violation for output node in macro {}.".format(name), node=name)

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
                raise CompilerException("Unable to expand graph, probably due to some misbehaving nodes")

            if expanded:
                break

        expanded = graph.nodes()

        self._debug("Expanded graph to {} nodes", len(expanded))

        output_nodes = []

        values = dict()
        aliases = dict()

        if output is None:
            include_output = lambda x: True
        elif isinstance(output, Container):
            include_output = lambda x: x in output
        elif isinstance(output, str):
            include_output = lambda x: x == output
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
                    raise ValidationException("Only atomic operations allowed, got {}".format(node), node=node)
                if isinstance(node, Conditional):
                    # In case of a conditional node we can determine which nodes will be executed only
                    # in certain conditions and insert jumps into the pipeline to speed up execution.
                    #
                    # We add this constraints also in case we do not use predicitive jumps to maintain
                    # the order of operations for comparison.

                    # Only register conditon once, do not use sets to preserve order
                    if not node.condition.name in conditions:
                        conditions.append(node.condition.name)
                    # List of nodes required by branch true
                    tree_true = set(traverse(optimized, node.true.name))
                    # List of nodes required by branch false
                    tree_false = set(traverse(optimized, node.false.name))
                    # List of nodes required to process condition
                    tree_condition = set(traverse(optimized, node.condition.name))
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

        if not self._predictive:
            self._debug("Jump optimization not disabled, skipping")
            # Do not process jumps, just sort operations according to their dependencies
            ordered = [(level, item) for level, sublist in enumerate(
                toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)
            operations = [(name, operation_data(optimized[name]), [
                           r.name for r in optimized[name].input_values()]) for _, name in ordered]
        else:

            def merge_branch(branch, condition, value=None):
                if condition in branch:
                    del branch[condition]
                elif value is not None:
                    branch[condition] = value

            self._debug("Calculating branch sets")
            branches = {}
            for output_node in output_nodes:
                for name, stack in traverse(optimized, output_node, stack=[]):
                    branches.setdefault(name, BranchSet(conditions))
                    branch = {}
                    for i, node in enumerate(stack[:-1]):
                        if isinstance(optimized[node], Conditional):
                            condition = optimized[node]
                            ancestor = stack[i+1]
                            if condition.condition == ancestor:
                                merge_branch(branch, condition.condition.name)
                                break
                            elif condition.true == ancestor:
                                merge_branch(
                                    branch, condition.condition.name, True)
                            elif condition.false == ancestor:
                                merge_branch(
                                    branch, condition.condition.name, False)

                branches[name].add(**branch)

            #print("digraph pixelpipes {")
            # for k, deps in dependencies.items():
            #    k = k.replace(".", "_")
            #    for v in deps:
            #        v = v.replace(".", "_")
            #        print("   %s -> %s; " % (k, v))
            # print("}")

            converter = BranchSet(conditions)

            # There are cases where a node without a direct dependency is considered
            # redundant in certain branch, we have to add these condition nodes to its
            # dependencies to maintain a valid order
            for name, node_branches in branches.items():
                dependencies.setdefault(name, set()).update(
                    node_branches.used())

            # Insert partiton information into the sorting criteria, group instructions within levels by their
            # partition sets
            ordered = [(level, branches[item].condition(), item) for level,
                       sublist in enumerate(toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)

            operations = []

            pending = None
            state = []

            for level, condition, name in ordered:
                if self._debug_enabled:
                    self._debug("{}, {}: {}", level, name,
                                converter.text(condition))
                if condition != state:
                    if not pending is None:
                        current_position = len(operations)
                        pending_position, pending_negate, pending_inputs = pending
                        cjump = ("_cjump", pending_negate,
                                 current_position - 1 - pending_position)
                        operations[pending_position] = (
                            "GOTO: %d" % current_position, cjump, pending_inputs)
                        pending = None

                    equation, inputs = converter.function(condition)
                    if equation is not None:
                        operations.append(None)
                        pending = (len(operations) - 1, equation, inputs)

                    state = condition

                operations.append((name, operation_data(optimized[name]), [
                                  r.name for r in optimized[name].input_values()]))

        pipeline_operations = []
    
        # Assemble a list of pipeline operations
        for i, (name, data, inputs) in enumerate(operations):
            self._debug("{}: {}", i, data[0])
            pipeline_operations.append(PipelineOperation(name, data[0], list(data[1:]), inputs))

        return pipeline_operations

    def _debug(self, message: str, *args, **kwargs):
        if self._debug_enabled:
            print(message.format(*args, **kwargs))
