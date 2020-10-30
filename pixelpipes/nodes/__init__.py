
from attributee import String, Float, Integer, Map, List, Boolean, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types

class Constant(Node):

    value = Number()

    def operation(self):
        return engine.Constant(self.value)

    def _output(self) -> types.Type:
        return Constant.resolve_type(self.value)

    @staticmethod
    def resolve_type(value) -> types.Type:
        if isinstance(value, int):
            return types.Integer(value)
        else:
            return types.Float(value)

    def key(self):
        typ = Constant.resolve_type(self.value)
        if isinstance(typ, types.Integer):
            return ("int", self.value)
        else:
            return ("float", self.value)

class Variable(Node):

    name = String()
    default = Number()

from ._numeric import *
from ._view import *
from ._list import *
from ._image import *
from ._points import *
