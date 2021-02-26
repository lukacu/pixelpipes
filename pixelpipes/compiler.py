
import logging
import typing
import numbers
from functools import reduce as _reduce
from itertools import combinations

from intbitset import intbitset

from pixelpipes import Graph, GraphBuilder, Reference, \
    Output, Node, Macro, Copy, DebugOutput, Input, \
    NodeException, ValidationException, Pipeline
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
        raise CompilerException('Cyclic dependency detected among nodes: {}'.format(', '.join(repr(x) for x in data.items())))

def infer_type(node: str, nodes: typing.Mapping[str, Node], type_cache: typing.Mapping[str, types.Type] = None) -> types.Type:
    """Computes output type for a given node by recursively computing types of its dependencies and
    calling validate method of a node with the information about their computed output types.

    Args:
        node (str): Name of the node
        nodes (typing.Mapping[str, Node]): Mapping of all nodes in the graph
        type_cache (typing.Mapping[str, types.Type], optional): Optional cache for already computed types.
            Makes repetititve calls much faster. Defaults to None.

    Raises:
        ValidationException: Contains information about the error during node validation process.

    Returns:
        types.Type: Computed type for the given node.
    """

    if not isinstance(node, Reference):
        if isinstance(node, int):
            return types.Integer(node)
        if isinstance(node, float):
            return types.Float(node)

    name = node.name

    if type_cache is not None and name in type_cache:
        return type_cache[name]

    if name not in nodes:
        raise ValidationException("Node reference {} not found".format(node), node=node)

    try:

        input_types = {k: infer_type(i, nodes, type_cache) for k, i in zip(nodes[name].input_names(), nodes[name].input_values())}
        output_type = nodes[name].validate(**input_types)

        if type_cache is not None:
            type_cache[name] = output_type
        return output_type

    except ValidationException as e:
        raise ValidationException("Node {}: {}".format(name, str(e)), node=node)

def merge(i, j):
    """ Combine two minterms. """
    if i[1] != j[1]:
        return None
    y = i[0] ^ j[0]
    if not (y & (~y + 1)) == y:
        return None
    return (i[0] & j[0], i[1] | y)


def _insert_minterm(minterms, new):
    (pos2, neg2) = tuple(new)

    for i, (pos1, neg1) in enumerate(minterms):
        used1 = pos1 | neg1 # Terms used in clause 1
        used2 = pos2 | neg2 # Terms used in clause 2
        common = used1 & used2 # All terms used in both clauses

        if used1 == common and used2 == common: # Same variables used
            if pos1 == pos2: # Same clause
                return minterms
            change = pos1 ^ pos2
            if len(change) == 1: # We can remove a single independent variable
                del minterms[i]
                new = (pos1 - change, neg1 - change)
                return _insert_minterm(minterms, new)
        elif pos1 == (pos2 & pos1) and neg1 == (neg2 & neg1):
            return minterms # Reduce to clause 1, already merged
        elif pos2 == (pos2 & pos1) and neg2 == (neg2 & neg1):
            del minterms[i]
            return _insert_minterm(minterms, (pos2, neg2)) # Reduce to clause 2
        # Clause not the same, move to next one

    # Not merged, add to list
    return minterms + [new]

class BranchSet(object):

    def __init__(self, variables):
        self._variables = variables
        self._branches = []

    def add(self, **variables):
        pos = intbitset([self._variables.index(x) for x, v in variables.items() if x in self._variables and v])
        neg = intbitset([self._variables.index(x) for x, v in variables.items() if x in self._variables and not v])

        self._branches = _insert_minterm(self._branches, (pos, neg))

    def condition(self):
        return self._branches

    def text(self, minterms):
        def parentheses(glue, array):
            if not array:
                return "True"
            if len(array) > 1:
                return ''.join(['(', glue.join(array), ')'])
            else:
                return glue.join(array)

        or_terms = []
        for (pos, neg) in minterms:
            and_terms = []
            for j in range(len(self._variables)):
                if j in pos:
                    and_terms.append(self._variables[j])
                elif j in neg:
                    and_terms.append('(NOT %s)' % self._variables[j])
            or_terms.append(parentheses(' AND ', and_terms))
        return parentheses(' OR ', or_terms)


    def function(self, minterms):

        variables = []
        structure = []
        for (pos, neg) in minterms:
            and_terms = []
            for j in range(len(self._variables)):
                if j in pos:
                    and_terms.append(False)
                    variables.append(self._variables[j])
                elif j in neg:
                    and_terms.append(True)
                    variables.append(self._variables[j])
            if and_terms:
                structure.append(and_terms)
        if structure:
            return structure, variables
        else:
            return None, []

class CompilerException(Exception):
    pass

class Conditional(Node):
    """Conditional selection
    
    Node that executes conditional selection, output of branch "true" will be selected if
    the "condition" is not zero, otherwise output of branch "false" will be selected.

    Inputs:
     * true (Primitve): Use this data if condition is true
     * false (Primitve): Use this data if condition is false
     * condition (Integer): Condition to test

    Parameters

    Category: flow
    """

    true = Input(types.Primitive())
    false = Input(types.Primitive())
    condition = Input(types.Integer())

    def operation(self) -> engine.Operation:
        return engine.Conditional([[False]])

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["true"].common(inputs["false"])

class Compiler(object):
    """Compiler object contains utilities to validate a graph and compiles it to a pipeline
     (a sequence of operations, written in native code) that can be executed to obtain output
     variables.
    """

    @staticmethod
    def compile_graph(graph, variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
            output: typing.Optional[str] = None, fixedout: bool = False):
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

        output_types = None

        for output in outputs:
            inputs = [infer_type(i, nodes, type_cache) for i in nodes[output].input_values()]

            for i, output_type in enumerate(inputs):
                if self._fixedout and not output_type.fixed():
                    raise ValidationException("Output {} not fixed for output node {}: {}".format(i, output, output_type), output)
                if isinstance(output_type, types.Complex):
                    raise ValidationException("Output {} is an complex type for output node {}: {}".format(i, output, output_type))

            if output_types is None:
                output_types = inputs
            else:
                if len(output_types) != len(inputs):
                    raise ValidationException("Unexpected number of inputs for output node {}".format(output))
                for out_a, out_b in zip(output_types, inputs):
                    if not out_a.castable(out_b):
                        raise ValidationException("Incompatible inputs for output node {}".format(output))

        if output_groups and len(outputs) != len(output_groups):
            raise ValidationException("Provided output group assignment does not match the size of the output")

        #for name, node in nodes.items():
        #    self._debug("Node {} ({}), inferred type: {}", name, node.__class__.__name__, type_cache[name])

        return type_cache

    def compile(self, graph: typing.Union[Graph, typing.Mapping[str, Node]],
            variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
            output: typing.Optional[str] = None) -> engine.Pipeline:
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

            # Special case where only a single node is returned
            if isinstance(subgraph, Node):
                subgraph = {name : subgraph}

            self._debug("Expanding macro node {} to {} nodes", name, len(subgraph))

            for subname, _ in subgraph.items():
                infer_type(Reference(subname), subgraph, inferred_types)

            for subname, subnode in subgraph.items():
                if subname != name and not subname.startswith(name + "."):
                    raise NodeException("Expanded node has illegal name {}, must start with {}.".format(subname, name), node=name)
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

                for sub_node in tree_true.difference(common):
                    dependencies.setdefault(sub_node, set()).add(node.condition.name)
                for sub_node in tree_false.difference(common):
                    dependencies.setdefault(sub_node, set()).add(node.condition.name)

            dependencies.setdefault(name, set()).update([node.name for node in optimized[name].input_values()])

        self._debug("Resolving dependencies")

        if not self._predictive:
            self._debug("Jump optimization not disabled, skipping")
            # Do not process jumps, just sort operations according to their dependencies
            ordered = [(level, item) for level, sublist in enumerate(_toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)
            operations = [(name, optimized[name].operation(), [r.name for r in optimized[name].input_values()]) for _, name in ordered]
        else:
            self._debug("Calculating branch sets")
            branches = {}
            for name, stack in traverse(optimized, output_node, stack=[]):
                branches.setdefault(name, BranchSet(conditions))
                branch = {}
                for i, node in enumerate(stack[:-1]):
                    if isinstance(optimized[node], Conditional):
                        condition = optimized[node]
                        ancestor = stack[i+1]
                        if condition.true == ancestor:
                            branch[condition.condition.name] = True
                        elif condition.false == ancestor:
                            branch[condition.condition.name] = False
                branches[name].add(**branch)


            # Insert partiton information into the sorting criteria, group instructions within levels by their
            # partition sets
            ordered = [(level, branches[item].condition(), item) for level, sublist in enumerate(_toposort(dependencies)) for item in sublist]
            ordered = sorted(ordered)

            converter = BranchSet(conditions)
            operations = []

            pending = None
            state = []

            for level, condition, name in ordered:
                self._debug("{}, {}: {}", level, name, converter.text(condition))
                if condition != state:
                    if not pending is None:
                        current_position = len(operations)
                        pending_position, pending_negate, pending_inputs = pending
                        operations[pending_position] = ("GOTO: %d" % current_position, engine.ConditionalJump(pending_negate, current_position - 1 - pending_position), pending_inputs)
                        pending = None

                    equation, inputs = converter.function(condition)
                    if equation is not None:
                        operations.append(None)
                        pending = (len(operations) - 1, equation, inputs)

                    state = condition

                operations.append((name, optimized[name].operation(), [r.name for r in optimized[name].input_values()]))

        pipeline = engine.Pipeline()

        indices = {}

        # Assemble the low-level pipeline
        for name, operation, inputs in operations:
            input_indices = [indices[name] for name in inputs]
            indices[name] = pipeline.append(operation, input_indices)
            assert indices[name] >= 0
            self._debug("{} ({}): {} ({})", indices[name], name,
                    operation.__class__.__name__, ", ".join(["{} ({})".format(i, n) for i, n in zip(input_indices, inputs)]))

        output_types = [inferred_types[ref.name] for ref in optimized[output_node].input_values()]

        if not output_groups:
            output_groups = ["default" for _ in output_types]

        pipeline.finalize()
        return Pipeline(pipeline, output_groups, output_types)

    def _debug(self, message: str, *args, **kwargs):
        if self._debug_enabled:
            print(message.format(*args, **kwargs))