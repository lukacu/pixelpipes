

from attributee import String, Number

from .. import types
from ..graph import Node, Input, Operation

class CompilerException(Exception):
    pass

class Variable(Node):
    """Variable placeholder that can be overriden later"""

    name = String()
    default = Number()

from .core import Compiler