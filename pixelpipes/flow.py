
from attributee import List, Float

from . import types
from .graph import Macro, Input, NodeException, SeedInput, Copy, Operation

class Conditional(Operation):
    """Node that executes conditional selection, output of branch "true" will be selected if
    the "condition" is not zero, otherwise output of branch "false" will be selected. Note that the inferred type of these 
    two branches should match as much as possible, otherwise the inferred type of this node will cause problems with
    dependent nodes.
    """

    true = Input(types.Wildcard(), description="Use this data if condition is true")
    false = Input(types.Wildcard(), description="Use this data if condition is false")
    condition = Input(types.Integer(), description="Condition to test") # Should be boolean, but at the moment testing throws an error

    def operation(self):
        return "condition",

class Toggle(Macro):
    """Node that executes conditional selection, output of branch "true" will be selected in "p" cases, otherwise output of branch "false" will be selected.
    """

    true = Input(types.Wildcard(), description="Use this data if condition is true")
    false = Input(types.Wildcard(), description="Use this data if condition is false")
    p = Float(val_min=0, val_max=1, description="Probability of selecting true branch")
    seed = SeedInput()

    def expand(self, **inputs):
        from .numbers import SampleUnform
        from .resource import Resource, ConditionalResource

        resource_type = Resource()

        is_resource = resource_type.castable(inputs["true"].type) and resource_type.castable(inputs["false"].type)

        random = SampleUnform(min=0, max=1, seed=inputs["seed"])

        comparison = random < self.p

        if is_resource:
            return ConditionalResource(condition=comparison, true=inputs["true"], false=inputs["false"])
        else:
            return Conditional(condition=comparison, true=inputs["true"], false=inputs["false"])

class Switch(Macro):
    """Random switch between multiple branches, a macro that generates a tree of binary choices based on a random
    variable. The probability of choosing a defined branch """

    inputs = List(Input(types.Anything()), description="Two or more input branches")
    weights = List(Float(val_min=0), description="Corresponing branch weights")
    seed = SeedInput()

    def _init(self):
        if len(self.inputs) == 0:
            raise NodeException("No inputs provided", node=self)

        if len(self.inputs) != len(self.weights):
            raise NodeException("Number of inputs and weights does not match", node=self)

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in enumerate(self.inputs)] + [self.seed]

    def get_inputs(self):
        return [(str(k), types.Anything()) for k, _ in enumerate(self.inputs)] + [("seed", types.Integer())]

    # TODO: sort probabilites to minimize average number of binary comparisons
    def expand(self, **inputs):
        """Decomposes switch statement into a series of internal conditional
        nodes that are recognized by the graph compiler. Adds a uniform
        distribition sampler as a source of the switch.
        """

        from .numbers import SampleUnform
        from .resource import Resource, ConditionalResource, CopyResource

        resource_type = Resource()

        is_resource = all([resource_type.castable(inputs[str(i)].type) for i in range(len(self.weights))])

        total_weight = sum(self.weights)

        random = SampleUnform(min=0, max=total_weight, seed=inputs["seed"])

        threshold = 0
        tree = None

        for i, weight in enumerate(self.weights):
            branch = inputs[str(i)]
            if weight == 0:
                continue

            if tree is None:
                tree = branch
                threshold += weight
                continue

            comparison = random < threshold

            if is_resource:
                tree = ConditionalResource(condition=comparison, true=tree, false=branch)
            else:
                tree = Conditional(condition=comparison, true=tree, false=branch)
            threshold += weight

        if is_resource:
            return CopyResource(source=tree)
        else:
            return Copy(source=tree)
