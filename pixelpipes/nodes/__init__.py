
from attributee import String, Float, Integer, Map, List, Boolean, Number

import pixelpipes.engine as engine
import pixelpipes.types as types

from .numeric import *
from .numeric import _NumericNode
from .view import *
from .list import *
from .image import *
from .points import *

class Variable(_NumericNode):

    node_description = "Variable placeholder that can be overriden later"

    name = String()
    default = Number()

def find_nodes(module=None):

    from pixelpipes import Node
    import inspect

    if module is None:
        import pixelpipes
        module = pixelpipes

    nodes = []

    for name in dir(module):
        if name.startswith("_"):
            continue
        member = getattr(module, name)
        if inspect.isclass(member) and issubclass(member, Node) and not member.hiddend():
            nodes.append(member)

    return nodes