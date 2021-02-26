
from attributee import List, Float

from pixelpipes import Macro, Input, NodeException, GraphBuilder, Copy, hidden
from pixelpipes import types

from pixelpipes.nodes.resources import Resource, ResourceList

from pixelpipes.compiler import Conditional


class ConditionalResource(Macro):
    """Conditional selection
    
    Node that executes conditional selection, output of branch "true" will be selected if
    the "condition" is not zero, otherwise output of branch "false" will be selected.

    Inputs:
     * true (Complex): Use this data if condition is true
     * false (Primitve): Use this data if condition is false
     * condition (Integer): Condition to test

    Category: flow, resource
    """

    true = Input(Resource())
    false = Input(Resource())
    condition = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["true"].common(inputs["false"])

    def expand(self, inputs, parent: "Reference"):

        true_reference, true_type = inputs["true"]
        false_reference, false_type = inputs["false"]
        condtion_reference, _ = inputs["condition"]

        common_type = true_type.common(false_type)

        assert not isinstance(common_type, types.Any)

        with GraphBuilder(prefix=parent) as builder:

            if isinstance(common_type, ResourceList):
                for field in common_type.meta():
                    Conditional(true=true_type.access(field, true_reference),
                        false=false_type.access(field, false_reference),
                        condition=condtion_reference, _name="." + field)

            for field in common_type.fields():
                true = true_type.access(field, true_reference)
                false = false_type.access(field, false_reference)

                Conditional(true=true, false=false,
                    condition=condtion_reference, _name="." + field)
            
            return builder.nodes()

@hidden
class CopyResource(Macro):

    source = Input(types.Union(ResourceList(), Resource()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: "Reference"):

        source_reference, source_type = inputs["source"]

        with GraphBuilder(prefix=parent) as builder:

            if isinstance(source_type, ResourceList):
                for field in source_type.meta():
                    Copy(source=source_type.access(field, source_reference), _name="." + field)

            for field in source_type.fields():
                if isinstance(source_type, ResourceList) and source_type.virtual(field):
                    continue
                Copy(source=source_type.access(field, source_reference), _name="." + field)
            
            return builder.nodes()

class Switch(Macro):
    """Random switch between multiple branches

    Inputs:
        inputs: Input branches
        weights: Corresponing branch probabilities

    Category: core
    Tags: random, switch
    """

    inputs = List(Input(types.Any()))
    weights = List(Float(val_min=0))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if len(self.inputs) == 0:
            raise NodeException("No inputs provided", node=self)

        if len(self.inputs) != len(self.weights):
            raise NodeException("Number of inputs and weights does not match", node=self)

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Any()) for k, _ in enumerate(self.inputs)]

    def validate(self, **inputs):
        super().validate(**inputs)

        output = None

        for i in inputs.values():
            if output is None:
                output = i
                continue
            output = output.common(i)

        return output

    # TODO: sort probabilites to minimize average number of binary comparisons
    def expand(self, inputs, parent: str):
        """Decomposes switch statement into a series of internal conditional
        nodes that are recognized by the graph compiler. Adds a uniform
        distribition sampler as a source of the switch.
        """

        from pixelpipes.nodes.numeric import UniformDistribution, Threshold

        resource_type = Resource()

        is_resource = all([resource_type.castable(inputs[str(i)][1]) for i in range(len(self.weights))])

        total_weight = sum(self.weights)

        with GraphBuilder(prefix=parent) as builder:
            
            random = UniformDistribution(min=0, max=total_weight)

            threshold = 0
            tree = None

            for i, weight in enumerate(self.weights):
                branch = inputs[str(i)][0]
                if weight == 0:
                    continue

                if tree is None:
                    tree = branch
                    threshold += weight
                    continue

                comparison = Threshold(threshold=threshold, comparison="LOWER", source=random)

                if is_resource:
                    tree = ConditionalResource(condition=comparison, true=tree, false=branch)
                else:
                    tree = Conditional(condition=comparison, true=tree, false=branch)
                threshold += weight

            if is_resource:
                CopyResource(source=tree, _name=parent)
            else:
                Copy(source=tree, _name=parent)

            return builder.nodes()
        