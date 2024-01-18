
from attributee.primitives import String
from attributee import List, Primitive

from . import types
from .graph import Constant, Operation, Input, SeedInput, ValidationException, hidden, Macro, Node, NodeOperation
from .numbers import Round, SampleUnform

def Wildlist(element=None):
    return types.Wildcard(element=element, mindim=1)

class Table(Macro):
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


class Concatenate(Operation):

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


class Remap(Operation):
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


class Range(Operation):
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


class Permute(Operation):
    """
    Randomly permutes an input list
    """

    source = Input(Wildlist(), description="Input list")
    seed = SeedInput()

    def infer(self, source, seed):
        return source

    def operation(self):
        return "list_permute",


class Permutation(Operation):
    """Generates a list of numbers from 0 to length in random order."""

    length = Input(types.Integer())
    seed = SeedInput()

    def infer(self, length):
        return types.List("int")

    def operation(self):
        return "list_permutation",


class GetElement(Operation):
    """
    Returns an element from a list for a given index
    """

    parent = Input(Wildlist())
    index = Input(types.Integer())

    def infer(self, parent, index):
        return parent.pop()

    def operation(self):
        return "list_element",

Node.register_operation(NodeOperation.INDEX, GetElement, Wildlist(), types.Integer())

class Length(Operation):
    """
    Returns a list length
    """

    parent = Input(Wildlist())

    def infer(self, parent):
        return types.Integer()

    def operation(self):
        return "list_length",

Node.register_operation(NodeOperation.LENGTH, Length, Wildlist())

class MakeList(Operation):
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


class Repeat(Operation):
    """Repeat list element a number of times
    """

    source = Input(types.Wildcard(), description="Element to repeat")
    length = Input(types.Integer(), description="Number of repetitions")

    def infer(self, source, length):
        return source.push()

    def operation(self):
        return "list_repeat",


class GetRandom(Macro):

    source = Input(Wildlist())
    seed = SeedInput()

    def expand(self, source, seed):
        generator = SampleUnform(0, Length(source)-1, seed=seed)
        index = Round(generator)
        return GetElement(source, index)




