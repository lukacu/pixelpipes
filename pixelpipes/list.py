
from attributee.primitives import String
from attributee import List, Primitive

from . import types
from .graph import Constant, Operation, Input, SeedInput, ValidationException, hidden, Macro, Node, NodeOperation
from .numbers import Round, SampleUnform

def Wildlist(element=None):
    return types.Wildcard(element=element, mindim=1)

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

    def expand(self):
        data = Constant([item for sublist in self.source for item in sublist])
        return ListAsTable(data, row=self._row)


class FileList(Operation):
    """ String list of file patchs. Use this operation to inject file dependencies into the 
    pipeline.
    """

    list = List(String())

    def infer(self):
        return types.Token("char", len(self.list), None)

    def operation(self):
        return "file_list", list(self.list)


class SublistSelect(Operation):
    """
    Selects a range from the source list as a new list.
    """

    parent = Input(Wildlist(), description="Source list")
    begin = Input(types.Integer(), description="Start index")
    end = Input(types.Integer(), description="End index")

    def infer(self, parent, begin, end):
        return types.List(parent.element)

    def operation(self):
        return "list_sublist",


class ListAsTable(Operation):
    """Transform list to table
    """

    parent = Input(types.List(), description="Source list")
    row = Input(types.Integer(),
                description="Row size, total length of list must be its multiple")

    def infer(self, parent, row):
        return types.List(parent.element)

    def operation(self):
        return "list_table",


class ListConcatenate(Operation):

    inputs = List(Input(Wildlist()), description="Two or more input lists")

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), Wildlist()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)

    def infer(self, **inputs):
        length = 0
        common = inputs["0"].type

        for l in inputs.values():
            common = common.common(l)
            if l.length is None:
                length = None
                return
            else:
                length += l.length

        if isinstance(common, types.Anything):
            raise ValidationException("Incompatible elements")

        return common.push(length)

    def operation(self):
        return "list_concatenate",


class FilterSelect(Operation):
    """Generate a sublist based on values from a filter list
    """

    parent = Input(Wildlist())
    filter = Input(types.IntegerList())

    def infer(self, parent, filter):
        return parent

    def operation(self):
        return "list_filter",


class ListRemap(Operation):
    """
    Maps elements from source list to a result list using indices from indices list.
    """

    source = Input(Wildlist())
    indices = Input(types.IntegerList())

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(inputs["parent"].element)

    def operation(self):
        return "list_remap",


class ListRange(Operation):
    """
    Generates a list of numbers from start to end of a given length
    """

    start = Input(types.Float())
    end = Input(types.Float())
    length = Input(types.Integer())
    round = Input(types.Boolean(), default=False)

    def validate(self, start, end, length, round):
        return types.FloatList()

    def operation(self):
        return "list_range",


class ListPermute(Operation):
    """
    Randomly permutes an input list
    """

    source = Input(Wildlist(), description="Input list")
    seed = SeedInput()

    def infer(self, source, seed):
        return source

    def operation(self):
        return "list_permute",


class ListPermutation(Operation):
    """Generates a list of numbers from 0 to length in random order."""

    length = Input(types.Integer())
    seed = SeedInput()

    def infer(self, length):
        return types.List("int")

    def operation(self):
        return "list_permutation",


class ListElement(Operation):
    """
    Returns an element from a list for a given index
    """

    parent = Input(Wildlist())
    index = Input(types.Integer())

    def infer(self, parent, index):
        return parent.pop()

    def operation(self):
        return "list_element",

Node.register_operation(NodeOperation.INDEX, ListElement, Wildlist(), types.Integer())

class ListLength(Operation):
    """
    Returns a list length
    """

    parent = Input(Wildlist())

    def infer(self, parent):
        return types.Integer()

    def operation(self):
        return "list_length",

Node.register_operation(NodeOperation.LENGTH, ListLength, Wildlist())

class ListBuild(Operation):
    """
    Builds list from inputs. All inputs should be of the same type as the first input, it determines
    the type of a list.
    """

    inputs = List(Input(types.Wildcard()))

    def input_values(self):
        return [self.inputs[int(name)] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(str(k), types.Wildcard()) for k, _ in enumerate(self.inputs)]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(_origin=_origin, **config)

    def infer(self, **inputs):
        common = types.Anything()
        for _, v in inputs.items():
            common = common.common(v)
        return common.push()

    def operation(self):
        return "list_build",


class RepeatElement(Operation):
    """Repeat list element a number of times
    """

    source = Input(types.Wildcard(), description="Element to repeat")
    length = Input(types.Integer(), description="Number of repetitions")

    def infer(self, source, length):
        return source.push()

    def operation(self):
        return "list_repeat",


class RandomElement(Macro):

    source = Input(Wildlist())
    seed = SeedInput()

    def expand(self, source, seed):
        generator = SampleUnform(0, ListLength(source)-1, seed=seed)
        index = Round(generator)
        return ListElement(source, index)


@hidden
class _ListArithmetic(Operation):

    a = Input(types.List("float"))
    b = Input(types.List("float"))

    def infer(self, **inputs):
        return types.FloatList()

    def operation(self):
        raise NotImplementedError()


class ListSum(_ListArithmetic):

    def operation(self):
        return "list_sum",


class ListMinus(_ListArithmetic):

    def operation(self):
        return "list_minus",


class ListMultiply(_ListArithmetic):

    def operation(self):
        return "list_multiply",


class ListDivide(_ListArithmetic):

    def operation(self):
        return "list_divide",


class ListModulo(_ListArithmetic):

    a = Input(types.List("int"))
    b = Input(types.List("int"))

    def infer(self, **inputs):
        return types.IntegerList()

    def operation(self):
        return "list_modulus",


Node.register_operation(NodeOperation.ADD, ListSum,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.SUBTRACT, ListMinus,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.MULIPLY, ListMultiply,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.DIVIDE, ListDivide,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.MODULO, ListModulo,
                        types.FloatList(), types.FloatList())


@hidden
class _ListCompare(Operation):

    a = Input(types.FloatList())
    b = Input(types.FloatList())

    def infer(self, **inputs):
        return types.BooleanList()

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
        return "list_compare_greater_equal",

Node.register_operation(NodeOperation.EQUAL, ListCompareEqual,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.NOT_EQUAL, ListCompareNotEqual,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.LOWER, ListCompareLower,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.LOWER_EQUAL, ListCompareLowerEqual,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.GREATER, ListCompareGreater,
                        types.FloatList(), types.FloatList())
Node.register_operation(NodeOperation.GREATER_EQUAL, ListCompareGreaterEqual,
                        types.FloatList(), types.FloatList())


class _ListLogical(Operation):

    a = Input(types.BooleanList())

    def infer(self, **inputs):
        return types.BooleanList()

    def operation(self):
        raise NotImplementedError()


class ListLogicalNot(_ListLogical):

    def operation(self):
        return "list_logical_not",


class ListLogicalAnd(_ListLogical):

    b = Input(types.BooleanList())

    def operation(self):
        return "list_logical_and",


class ListLogicalOr(_ListLogical):

    b = Input(types.BooleanList())

    def operation(self):
        return "list_logical_or",

# TODO: register artithmetic operations

