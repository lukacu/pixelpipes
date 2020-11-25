
from attributee import String, Float, Integer, Map, List, Boolean, Number

from pixelpipes import Node, Input
import pixelpipes.engine as engine
import pixelpipes.types as types

from ._numeric import *
from ._view import *
from ._list import *
from ._image import *
from ._points import *

class Variable(NumericNode):

    name = String()
    default = Number()