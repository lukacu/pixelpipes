from attributee.primitives import Boolean, String
from attributee import List, Primitive

from . import types
from .graph import Node, Input, SeedInput, ValidationException, hidden, Macro, GraphBuilder, Reference
from .numbers import Round, UniformDistribution

class ConstantList(Node):
    """Constant List

    Inputs:
        - source: List type

    Category: list
    """

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

class ConstantTable(Macro):
    """Constant Table

    Inputs:
        - source: Table type

    Category: list
    """

    source = List(List(Primitive()), separator=";")

    def _init(self):
        rows = [len(sublist) for sublist in self.source]

        if not all(x == rows[0] for x in rows):
            raise types.TypeException("All rows should be equal")

        items = [item for sublist in self.source for item in sublist]
        if all(isinstance(x, int) for x in items):
            self._type = types.Integer()
        elif all(isinstance(x, (int, float)) for x in items):
            self._type = types.Float()
        else:
            raise types.TypeException("Unsupported element")

        self._row = rows[0]

    def _output(self):
        return types.List(types.List(self._type, self._row), len(self.source))

    def expand(self, _, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:
            data = ConstantList([item for sublist in self.source for item in sublist])
            ListAsTable(data, row=self._row, _name=parent)

            return builder.nodes()


class FileList(Node):
    """ String list of file patchs. Use this operation to inject file dependencies into the 
    pipeline.
    """

    list = List(String())

    def _output(self):
        # TODO: fix this, depth should not be hardcoded
        return types.List(types.String(), length=len(self.list))

    def operation(self):
        return "file_list", list(self.list)

class SublistSelect(Node):

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

class ListAsTable(Node):
    """ListAsTable

    View list as table

    Inputs:
     - parent: Source list
     - row: Row length

    Category: list
    """

    parent = Input(types.List())
    row = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        length = None

        if inputs["row"].value is not None and inputs["parent"].length is not None:
            length = inputs["parent"].length / inputs["row"].value

        return types.List(types.List(inputs["parent"].element, inputs["row"].value), length)

    def operation(self):
        return "list_table",

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
    """List filter

    Select an interval from a given list.

    Inputs:
        - parent: A list type
        - being: Sublist starting index
        - end: Sublist last index

    Category: list
    """

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

    Inputs:
        - source: A list type
        - indicies: A list of integers

    Category: list
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

    Inputs:
        - start: Starting number
        - end: Ending number
        - length: List length
        - round: Boolean

    Category: list
    """

    start = Input(types.Float())
    end = Input(types.Float())
    length = Input(types.Integer())
    round = Input(types.Boolean(), default=False)

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Float(), inputs["length"].value)

    def operation(self):
        return "list_range",


class ListPermute(Node):
    """List permute

    Randomly permutes an input list

    Category: list
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

    Category: list
    """

    length = Input(types.Integer())
    seed = SeedInput()

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["length"].value)

    def operation(self):
        return "list_permutation",

class ListElement(Node):
    """Retrieve element

    Returns an element from a list for a given index

    Inputs:
        - parent: Source list
        - index: Position of the element

    Category: list
    """

    parent = Input(types.List())
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
    """List Length

    Returns a list length

    Inputs:
        - parent: Source list

    Category: list
    """

    parent = Input(types.List())

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


class RandomListElement(Macro):

    source = Input(types.List(types.Primitive()))
    seed = SeedInput()

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["source"].element

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            length = inputs["source"].type.length
            generator = UniformDistribution(min=0, max=length-1, seed=inputs["seed"])
            index = Round(generator)
            ListElement(parent=inputs["source"], index=index, _name=parent)

            return builder.nodes()

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
        return "list_compare_equal",

class ListCompareNotEqual(_ListCompare):

    def operation(self):
        return "list_compare_not_equal",

class ListCompareLower(_ListCompare):

    def operation(self):
        return "list_compare_less",

class ListCompareLowerEqual(_ListCompare):

    def operation(self):
        return "list_compare_less_equal",

class ListCompareGreater(_ListCompare):

    def operation(self):
        return "list_compare_greater",

class ListCompareGreaterEqual(_ListCompare):

    def operation(self):
        return "list_compare_grater_equal",

class _ListLogical(Node):

    a = Input(types.List(types.Number()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["a"].length)

    def operation(self):
        raise NotImplementedError()

class ListLogicalNot(_ListLogical):

    def operation(self):
        return "list_logical_not",

class ListLogicalAnd(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        return "list_logical_and",

class ListLogicalOr(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        return "list_logical_or",

# TODO: register artithmetic operations