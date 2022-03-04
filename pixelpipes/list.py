from attributee.primitives import Boolean
from attributee import List, Primitive

from . import ComparisonOperations, LogicalOperations
from .graph import Node, Input, SeedInput, ValidationException, hidden
import pixelpipes.types as types

class ConstantList(Node):

    source = List(Primitive())

    def _init(self):
        if all(isinstance(x, int) for x in self.source):
            self._type = types.Integer()
        elif all(isinstance(x, (int, float)) for x in self.source):
            self._type = types.Float()
        else:
            raise types.TypeException("Unsupported element")

    def _output(self):
        return types.List(self._type, len(self.source))

    def operation(self):
        return "_constant", list(self.source)

class ConstantTable(Node):

    source = List(List(Primitive()), separator=";")

    def _init(self):
        elements = [item for sublist in self.source for item in sublist]

        if all(isinstance(x, int) for x in elements):
            self._type = types.Integer()
        elif all(isinstance(x, (int, float)) for x in elements):
            self._type = types.Float()
        else:
            raise types.TypeException("Unsupported element")

    def _output(self):
        return types.List(types.List(self._type), len(self.source))

    def operation(self):
        return "_constant", list([list(x) for x in self.source])

class SublistSelect(Node):
    """Sublist

    Selects a range from the source list as a new list.

    Inputs:
     - parent: Source list
     - begin: Start of sublist
     - end: End of sublist

    Category: list
    """

    parent = Input(types.List(types.Primitive()))
    begin = Input(types.Integer())
    end = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["begin"].value is not None and inputs["end"].value is not None:
            return types.List(inputs["parent"].element, inputs["end"].value - inputs["begin"].value)

        return types.List(inputs["parent"].element)

    def operation(self):
        return "list_sublist",


class ListConcatenate(Node):

    inputs = List(Input(types.List(types.Primitive())))

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Number()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)

    def validate(self, **inputs):
        super().validate(**inputs)

        length = 0
        common = inputs["0"].type

        for l in inputs.values():
            common = common.common(l)
            if l.length is None:
                length = None
                return
            else:
                length += l.length

        if isinstance(common, types.Any):
            raise ValidationException("Incompatible sublists")

        return types.List(common, length)

    def operation(self):
        return "list_concatenate",
class FilterSelect(Node):

    node_name = "List filter"
    node_description = "Select elements from source list based on indices in another list"
    node_category = "list"

    parent = Input(types.List(types.Primitive()))
    filter = Input(types.List(types.Integer()))

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["filter"].length is not None:
            return types.List(inputs["parent"].element, inputs["filter"].length)

        return types.List(inputs["parent"].element)

    def operation(self):
        return "list_filter",

class ListRemap(Node):
    """List remap
    
    Maps elements from source list to a result list using indices from indices list.

    """

    source = Input(types.List(types.Primitive()))
    indices = Input(types.List(types.Integer()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(inputs["parent"].element)

    def operation(self):
        return "list_remap",

class ListRange(Node):
    """List range

    Generates a list of numbers from start to end of a given length

    """

    start = Input(types.Float())
    end = Input(types.Float())
    length = Input(types.Integer())
    round = Boolean(default=False)

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer() if self.round else types.Float(), inputs["length"].value)

    def operation(self):
        return "list_range", self.round


class ListPermute(Node):
    """List permute

    Randomly permutes an input list

    """

    source = Input(types.List(types.Primitive()))
    seed = SeedInput()

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(inputs["source"].element, inputs["source"].length)

    def operation(self):
        return "list_permute",

class ListPermutation(Node):
    """List permutation

    Generates a list of numbers from 0 to length in random order.

    """

    lenght = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["length"].value)

    def operation(self):
        return "list_permutation",
class ListElement(Node):
    """Retrieve element

    Returns an element from a list for a given index

    Inputs:
        parent: Source list
        index: Position of the element

    Category: list
    """

    parent = Input(types.List(types.Primitive()))
    index = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        index = inputs["index"].value

        if index is not None:
            return inputs["parent"][index]

        return inputs["parent"].element

    def operation(self):
        return "list_element",

class ListLength(Node):

    parent = Input(types.List(types.Primitive()))

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Integer(inputs["parent"].length)

    def operation(self):
        return "list_length",

class ListBuild(Node):
    """Build list

    Builds list from inputs. All inputs should be of the same type as the first input, it determines
    the type of a list.

    Inputs:
      - inputs: Inputs to put in a list

    Category: list
    """

    inputs = List(Input(types.Number()))

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Number()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(inputs["0"], len(inputs))

    def operation(self):
        return "list_build",

class RepeatElement(Node):
    """Repeat list element a number of times

    Inputs:
     - source: Element to replicate
     - length: how many times to repeat

    Output: List

    Category: list
    """

    source = Input(types.Primitive())
    length = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.List(inputs["source"], inputs["length"].value)

    def operation(self):
        return "list_repeat",

@hidden
class _ListCompare(Node):

    a = Input(types.List(types.Number()))
    b = Input(types.List(types.Number()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["a"].length)

    def operation(self):
        raise NotImplementedError()

class ListCompareEqual(_ListCompare):

    def operation(self):
        return "list_compare", ComparisonOperations["EQUAL"]

class ListCompareLower(_ListCompare):

    def operation(self):
        return "list_compare", ComparisonOperations["LOWER"]

class ListCompareLowerEqual(_ListCompare):

    def operation(self):
        return "list_compare", ComparisonOperations["LOWER_EQUAL"]

class ListCompareGreater(_ListCompare):

    def operation(self):
        return "list_compare", ComparisonOperations["GREATER"]

class ListCompareGreaterEqual(_ListCompare):

    def operation(self):
        return "list_compare", ComparisonOperations["GREATER_EQUAL"]

class _ListLogical(Node):

    a = Input(types.List(types.Number()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["a"].length)

    def operation(self):
        raise NotImplementedError()

class ListLogicalNot(_ListLogical):

    def operation(self):
        return "list_compare", LogicalOperations["NOT"]

class ListLogicalAnd(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        return "list_compare", LogicalOperations["AND"]

class ListLogicalOr(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        return "list_compare", LogicalOperations["OR"]

# TODO: register artithmetic operations