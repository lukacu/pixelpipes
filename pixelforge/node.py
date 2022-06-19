
from attributee import List, Number, Integer, Boolean, String, Enumeration, AttributeException

from NodeGraphQt import NodeGraph, BaseNode

from pixelpipes import Input, GraphBuilder, NodeException


# create a example node object with a input/output port.
class Node(BaseNode):
    """example test node."""

    # unique node identifier domain. ("com.chantasticvfx.MyNode")
    __identifier__ = 'pixelforge'

    # initial default node name.
    NODE_NAME = 'Node'
    NODE_CLASS = None

    def __init__(self):
        super().__init__()

        self._dynamic = None
        self._exception = None

        for name, attr in self.node_class.list_attributes():
            if isinstance(attr, Input):
                self.add_input(name)
            elif isinstance(attr, Integer):
                self.add_int_input("pp_" + name, label=name, range=(attr.min, attr.max), value=attr.default)
            elif isinstance(attr, Number):
                self.add_float_input("pp_" + name, label=name, range=(attr.min, attr.max), value=attr.default)
            elif isinstance(attr, String):
                self.add_text_input("pp_" + name, label=name, text=attr.default)
            elif isinstance(attr, Boolean):
                self.add_checkbox("pp_" + name, label=name, state=attr.default)
            if isinstance(attr, List) and isinstance(attr.contains, Input):
                assert self._dynamic is None
                self._dynamic = name
                self.add_input("0")
            elif isinstance(attr, Enumeration):
                self.add_combo_menu("pp_" + name, label=name, items=list(attr.options.keys()))

        self.add_output(display_name=False)
        self.set_color(0, 100, 100)

    @classmethod
    def category(cls):
        return cls.NODE_CLASS.category()

    @classmethod
    def description(cls):
        return cls.NODE_CLASS.description()

    @classmethod
    def name(cls):
        return cls.NODE_CLASS.name()

    @property
    def node_class(self):
        return self.__class__.NODE_CLASS

    def set_error(self, error=None):
        self._error = error
        if self._error is not None:
            self.set_color(100, 0, 0)
        else:
            self.set_color(0, 100, 100)

    def generate(self):

        inputs = self.inputs()

        arguments = {}

        for name, attr in self.node_class.list_attributes():
            if isinstance(attr, Input):
                ports = inputs[name].connected_ports()
                if len(ports) == 1:
                    arguments[name] = "@" + ports[0].node().id
                else:
                    arguments[name] = "@_"
            elif isinstance(attr, List) and isinstance(attr.contains, Input):
                assert self._dynamic is not None
                inlist = []
                while True:
                    inname = str(len(inlist))
                    if not inname in inputs:
                        self.add_input(inname)
                        break
                    ports = inputs[inname].connected_ports()
                    if len(ports) == 1:
                        inlist.append("@" + ports[0].node().id)
                    else:
                        break
                arguments[name] = inlist
   
            else:
                arguments[name] = self.get_property("pp_" + name)

        try:
            return self.node_class(**arguments)

        except (NodeException, AttributeException) as e:
            self.set_error(e)
            return None
