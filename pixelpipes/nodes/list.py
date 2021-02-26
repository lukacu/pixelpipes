from attributee import AttributeException, List

from pixelpipes import Node, Input, hidden
import pixelpipes.engine as engine
import pixelpipes.types as types

@hidden
class ListSource(Node):

    def __init__(self, source, *args, element_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = source
        self._type = element_type

        if isinstance(self._source, engine.IntegerList):
            base_typ = types.Integer()
        elif isinstance(self._source, engine.FloatList):
            base_typ = types.Float()
        elif isinstance(self._source, engine.ImageFileList):
            base_typ =  types.Image()
        elif isinstance(self._source, engine.ImageList):
            base_typ = types.Image()
        elif isinstance(self._source, engine.PointList):
            base_typ = types.Points()
        elif isinstance(self._source, engine.TableList):
            base_typ = types.List(types.Number())
        else:
            raise types.TypeException("Cannot determine output type for list")

        if self._type is not None:
            assert base_typ.castable(element_type)
        else:
            self._type = base_typ

    def dump(self, ignore=None):
        # TODO: can we make list serializable?
        raise AttributeException("Node is not serializable")

    def duplicate(self, **inputs):
        return self

    def _output(self):
        return types.List(self._type)

    def operation(self):
        return engine.ListSource(self._source)

class SublistSelect(Node):

    node_name = "List interval"
    node_description = "Select an interval from a given list"
    node_category = "list"

    parent = Input(types.List(types.Primitive()))
    begin = Input(types.Integer())
    end = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["begin"].value is not None and inputs["end"].value is not None:
            return types.List(inputs["parent"].element, inputs["end"].value - inputs["begin"].value)

        return types.List(inputs["parent"].element)

    def operation(self):
        return engine.SublistSelect()

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
        return engine.FilterSelect()

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

        return inputs["parent"].element

    def operation(self):
        return engine.ListElement()

class ListLength(Node):

    parent = Input(types.List(types.Primitive()))

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Integer(inputs["parent"].length)

    def operation(self):
        return engine.ListLength()

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

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            i = int(k)
            assert i >= 0 and i < len(config["inputs"])
            config["inputs"][i] = v
        return self.__class__(**config)

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(inputs["0"], len(inputs))

    def operation(self):
        return engine.ListBuild()

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
        return engine.RepeatElement()

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
        raise engine.ListCompare(engine.Compare.EQUAL)

class ListCompareLower(_ListCompare):

    def operation(self):
        raise engine.ListCompare(engine.Compare.LOWER)

class ListCompareLowerEqual(_ListCompare):

    def operation(self):
        raise engine.ListCompare(engine.Compare.LOWER_EQUAL)

class ListCompareGreater(_ListCompare):

    def operation(self):
        raise engine.ListCompare(engine.Compare.GREATER)

class ListCompareGreaterEqual(_ListCompare):

    def operation(self):
        raise engine.ListCompare(engine.Compare.GREATER_EQUAL)

class _ListLogical(Node):

    a = Input(types.List(types.Number()))

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.List(types.Integer(), inputs["a"].length)

    def operation(self):
        raise NotImplementedError()

class ListLogicalNot(_ListLogical):

    def operation(self):
        raise engine.ListLogical(engine.Logical.NOT)

class ListLogicalAnd(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        raise engine.ListLogical(engine.Logical.AND)

class ListLogicalOr(_ListLogical):

    b = Input(types.List(types.Number()))

    def operation(self):
        raise engine.ListLogical(engine.Logical.OR)

