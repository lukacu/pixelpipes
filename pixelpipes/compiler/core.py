

from pixelpipes.node import InferredReference, Macro, Node, NodeException, Reference, ValidationException
from pixelpipes import Pipeline
from pixelpipes.core.numbers import Constant
import typing
import numbers

from ..graph import Graph, GraphBuilder

from pixelpipes.core import Copy, DebugOutput, Output, create_pipeline
import pixelpipes.types as types

from .utilities import BranchSet, Counter, infer_type, merge, toposort

from . import CompilerException, Conditional, Variable
from ..core import make_operation, create_pipeline

class Compiler(object):
    """Compiler object contains utilities to validate a graph and compiles it to a pipeline
     (a sequence of operations, written in native code) that can be executed to obtain output
     variables.
    """

    @staticmethod
    def compile_graph(graph, variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
            output: typing.Optional[str] = None, fixedout: bool = False) -> Pipeline:
        compiler = Compiler(fixedout)
        return compiler.compile(graph, variables, output)

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
        output_groups = []

        if isinstance(graph, Graph):
            nodes = graph.nodes
            output_groups = graph.groups
        elif isinstance(graph, GraphBuilder):
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
                self._debug("Node {} ({}), inferred type: {}", name, node.__class__.__name__, type_cache[name])

        output_types = None

        for output in outputs:
            inputs = [infer_type(i, nodes, type_cache) for i in nodes[output].input_values()]

            for i, output_type in enumerate(inputs):
                if self._fixedout and not output_type.fixed():
                    raise ValidationException("Output {} not fixed for output node {}: {}".format(i, output, output_type), output)
                if isinstance(output_type, types.Complex):
                    raise ValidationException("Output {} is an complex type for output node {}: {}".format(i, output, output_type), output)

            if output_types is None:
                output_types = inputs
            else:
                if len(output_types) != len(inputs):
                    raise ValidationException("Unexpected number of inputs for output node {}".format(output), output)
                for out_a, out_b in zip(output_types, inputs):
                    if not out_a.castable(out_b):
                        raise ValidationException("Incompatible inputs for output node {}".format(output), output)

        if output_groups and len(outputs) != len(output_groups):
            raise ValidationException("Provided output group assignment does not match the size of the output")

        return type_cache

    def compile(self, graph: typing.Union[Graph, typing.Mapping[str, Node]],
            variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
            output: typing.Optional[str] = None) -> Pipeline:
        """Compile a graph into a pipeline of native operations.

        Args:
            graph (typing.Union[Graph, typing.Dict]): Graph representation

        Raises:
            CompilerException: raised if graph is not valid

        Returns:
            engine.Pipeline: resulting pipeline
        """

        if GraphBuilder.default() is not None:
            raise CompilerException("Cannot compile within graph builder scope")

        output_groups = []
        if isinstance(graph, Graph):
            nodes = graph.nodes
            output_groups = graph.groups
        elif isinstance(graph, GraphBuilder):
            nodes = graph.nodes()
        elif isinstance(graph, typing.Mapping):
            nodes = graph
        else:
            raise CompilerException("Illegal graph specification")

        # Copy the nodes mapping structure
        nodes = dict(**nodes)

        self._debug("Compiling {} source nodes", len(nodes))

        for name, node in nodes.items():
            if isinstance(node, Variable):
                if variables is not None and node.name in variables:
                    nodes[name] = Constant(value=variables[node.name])
                else:
                    nodes[name] = Constant(value=node.default)

        inferred_types = self.validate(nodes)

        builder = GraphBuilder()
        constants = Counter()

        def normalize_input(node):
            inputs = {}
            for input_name, input_value in zip(node.input_names(), node.input_values()):
                if not isinstance(input_value, Reference):
                    constant_node = Constant(value=input_value, _auto=False)
                    constant_name = "constant%d" % constants()
                    inputs[input_name] = builder.add(constant_node, constant_name)
                else:
                    inputs[input_name] = input_value

            return inputs

        def expand_macro(name, node):
            inputs = {}
            for input_name, input_value in zip(node.input_names(), node.input_values()):
                if not isinstance(input_value, Reference):
                    constant_name = "constant%d" % constants()
                    constant_node = builder.add(Constant(value=input_value, _auto=False), constant_name)
                    inputs[input_name] = InferredReference(constant_node, Constant.resolve_type(input_value))
                    inferred_types[constant_name] = Constant.resolve_type(input_value)
                else:
                    inputs[input_name] = InferredReference(input_value, inferred_types[input_value.name])

            try:
                subgraph = node.expand(inputs, name)

            except Exception as ee:
                raise CompilerException("Exception during macro {} expansion".format(name)) from ee

            # Special case where only a single node is returned
            if isinstance(subgraph, Node):
                subgraph = {name : subgraph}

            self._debug("Expanding macro node {} to {} nodes", name, len(subgraph))

            for subname, _ in subgraph.items():
                infer_type(Reference(subname), subgraph, inferred_types)

            # TODO: verify that declared type is same as the one actually returned by generated subgraph

            for subname, subnode in subgraph.items():
                if subname != name and not subname.startswith(name + "."):
                    raise NodeException("Expanded node has illegal name {}, must start with {}.".format(subname, name), node=name)
                if isinstance(subnode, Macro):
                    expand_macro(subname, subnode)
                else:
                    inputs = normalize_input(subnode)
                    builder.add(subnode.duplicate(_origin=node, **inputs), subname)


        for name, node in nodes.items():
            if isinstance(node, Macro):
                expand_macro(name, node)
            else:
                inputs = normalize_input(node)
                builder.add(node.duplicate(**inputs), name)

        expanded = builder.nodes()

        self._debug("Expanded graph to {} nodes", len(expanded))

        self.validate(expanded)

        output_node = None

        values = dict()
        aliases = dict()

        for name, node in expanded.items():
            if isinstance(node, Output):
                if (output is not None and node.identifier == output) or output_node is None:
                    output_node = name
                elif output is None and output_node is not None:
                    raise CompilerException("Only one output node required")
            elif isinstance(node, Copy):
                aliases[aliases.get(name, name)] = aliases.get(node.source.name, node.source.name)
            elif isinstance(node, DebugOutput) and not self._debug_enabled:
                aliases[aliases.get(name, name)] = aliases.get(node.source.name, node.source.name)
            elif isinstance(node, Constant):
                if node.key() in values:
                    aliases[aliases.get(name, name)] = values[node.key()]
                else:
                    values[node.key()] = aliases.get(name, name)

        if output_node is None:
            if output is not None:
                raise CompilerException("Output node {} not found".format(output))
            else:
                raise CompilerException("No output node found")

        builder = GraphBuilder()

        for name, node in expanded.items():
            if name not in aliases:
                inputs = {}
                for input_name, input_value in zip(node.input_names(), node.input_values()):
                    inputs[input_name] = Reference(aliases.get(input_value.name, input_value))
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

        for name in traverse(optimized, output_node):
            node = optimized[name]
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
                common = tree_true.intersection(tree_false).union(tree_condition)

                if node.condition.name not in common:
                    for sub_node in tree_true.difference(common):
                        dependencies.setdefault(sub_node, set()).add(node.condition.name)
                    for sub_node in tree_false.difference(common):
                        dependencies.setdefault(sub_node, set()).add(node.condition.name)

            dependencies.setdefault(name, set()).update([node.name for node in optimized[name].input_values()])

        self._debug("Resolving dependencies")

        # Transform conditions to name-index map
        conditions = {v: i for i, v in enumerate(conditions)}

        def operation_data(node):
            meta = node.operation()
            if not isinstance(meta, (tuple, list)):
                raise CompilerException("Illegal operation data for node: %s" % node)
            return meta

        if not self._predictive:
            self._debug("Jump optimization not disabled, skipping")
            # Do not process jumps, just sort operations according to their dependencies
            ordered = [(level, item) for level, sublist in enumerate(toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)
            operations = [(name, operation_data(optimized[name]), [r.name for r in optimized[name].input_values()]) for _, name in ordered]
        else:

            def merge_branch(branch, condition, value = None):
                if condition in branch:
                    del branch[condition]
                elif value is not None:
                    branch[condition] = value

            self._debug("Calculating branch sets")
            branches = {}
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
                            merge_branch(branch, condition.condition.name, True)
                        elif condition.false == ancestor:
                            merge_branch(branch, condition.condition.name, False)

                branches[name].add(**branch)

            #print("digraph pixelpipes {")
            #for k, deps in dependencies.items():
            #    k = k.replace(".", "_")
            #    for v in deps:
            #        v = v.replace(".", "_")
            #        print("   %s -> %s; " % (k, v))
            #print("}")

            converter = BranchSet(conditions)

            # There are cases where a node without a direct dependency is considered
            # redundant in certain branch, we have to add these condition nodes to its
            # dependencies to maintain a valid order
            for name, node_branches in branches.items():
                dependencies.setdefault(name, set()).update(node_branches.used())
                
            # Insert partiton information into the sorting criteria, group instructions within levels by their
            # partition sets
            ordered = [(level, branches[item].condition(), item) for level, sublist in enumerate(toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)

            operations = []

            pending = None
            state = []

            for level, condition, name in ordered:
                if self._debug_enabled:
                    self._debug("{}, {}: {}", level, name, converter.text(condition))
                if condition != state:
                    if not pending is None:
                        current_position = len(operations)
                        pending_position, pending_negate, pending_inputs = pending
                        cjump = ("_cjump", pending_negate, current_position - 1 - pending_position)
                        operations[pending_position] = ("GOTO: %d" % current_position, cjump, pending_inputs)
                        pending = None

                    equation, inputs = converter.function(condition)
                    if equation is not None:
                        operations.append(None)
                        pending = (len(operations) - 1, equation, inputs)

                    state = condition

                operations.append((name, operation_data(optimized[name]), [r.name for r in optimized[name].input_values()]))

        pipeline = create_pipeline()
        indices = {}

        # Assemble the low-level pipeline
        for name, data, inputs in operations:
            operation = make_operation(*data)
            input_indices = [indices[name] for name in inputs]
            indices[name] = pipeline.append(operation, input_indices)
            assert indices[name] >= 0
            self._debug("{} ({}): {} ({})", indices[name], name,
                    data[0], ", ".join(["{} ({})".format(i, n) for i, n in zip(input_indices, inputs)]))

        output_types = [inferred_types[ref.name] for ref in optimized[output_node].input_values()]

        if not output_groups:
            output_groups = ["default" for _ in output_types]

        pipeline.finalize()
        return Pipeline(pipeline, output_groups, output_types, [data[0] for _, data, _ in operations])

    def _debug(self, message: str, *args, **kwargs):
        if self._debug_enabled:
            print(message.format(*args, **kwargs))