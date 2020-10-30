from attributee import String, Float, Integer, Map, List, Boolean, Enumeration

from pixelpipes import Node, Input, wrap_pybind_enum
import pixelpipes.engine as engine
import pixelpipes.types as types

class ListSource(Node):

    def __init__(self, source, element_type, **kwargs):
        super().__init__(**kwargs)
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
        elif isinstance(self._source, engine.PointsList):
            base_typ = types.Points()
        else:
            raise types.TypeException("Cannot determine output type for list")

        assert base_typ.castable(element_type)

    def duplicate(self, **inputs):
        return self

    def _output(self):
        return types.List(self._type)

    def operation(self):
        return engine.ListSource(self._source)

class SublistSelect(Node):

    parent = Input(types.List(types.Any()))
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

    parent = Input(types.List(types.Any()))
    filter = Input(types.List(types.Integer()))

    def validate(self, **inputs):
        super().validate(**inputs)

        if inputs["filter"].length is not None:
            return types.List(inputs["parent"].element, inputs["filter"].length)

        return types.List(inputs["parent"].element)

    def operation(self):
        return engine.FilterSelect()

class ListElement(Node):

    parent = Input(types.List(types.Any()))
    index = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["parent"].element

    def operation(self):
        return engine.ListElement()

class ListLength(Node):

    parent = Input(types.List(types.Any()))

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Integer(inputs["parent"].length)

    def operation(self):
        return engine.ListLength()


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

