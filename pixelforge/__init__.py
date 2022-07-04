import sys

from typing import ClassVar

from attributee import List, Number, Integer, Boolean, String, Enumeration, AttributeException

from NodeGraphQt import QtCore, QtWidgets
from NodeGraphQt import NodeGraph, setup_context_menu

from pixelpipes.graph import Graph, NodeException
from pixelpipes.utilities import find_nodes
import pixelpipes.resource

from pixelforge.list import NodeListWidget
from pixelpipes.graph import Node

def generate_class(node_class):

    attributes = {'NODE_NAME': node_class.name(), 'NODE_CLASS' : node_class}

    if node_class.category():
        attributes["__identifier__"] = "pixelpipes." + node_class.category()

    return type(node_class.__name__ + 'Proxy', (Node,), attributes)


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(None)

        self.setWindowTitle("PixelForge")

        # create the node graph controller.
        self._graph = NodeGraph()
        
        # set up default menu and commands.
        setup_context_menu(self._graph)
    
        for node in find_nodes():
            self._graph.register_node(generate_class(node))

        for node in find_nodes(pixelpipes.nodes.resources):
            self._graph.register_node(generate_class(node))

        for node in find_nodes(pixelpipes.datasets):
            self._graph.register_node(generate_class(node))


        self.setCentralWidget(self._graph.widget)

        dock = QtWidgets.QDockWidget(self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        dock.setFloating(False)
        dock.setMinimumSize(QtCore.QSize(200,800))
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        nodes_list = NodeListWidget(parent=dock, node_graph=self._graph)
        dock.setWidget(nodes_list)

        self._graph.nodes_deleted.connect(self._change)
        self._graph.node_created.connect(self._change)
        self._graph.port_connected.connect(self._change)
        self._graph.port_disconnected.connect(self._change)
        self._graph.property_changed.connect(self._change)


    def _change(self):
        graph = self.generate()
        if graph is None:
            return
        try:
            graph.validate()
        except NodeException as e:
            nid = None
            if e.node is not None:
                if not isinstance(e.node, str):
                    for name, node in graph.nodes.items():
                        if node == e.node:
                            nid = name
                            break
                else:
                    nid = e.node

            if nid is not None:
                self._graph.get_node_by_id(nid).set_error(e)

    def generate(self):

        builder = Graph()

        for gnode in self._graph.all_nodes():
            node = gnode.generate()
            if node is None:
                continue
            builder.add(node, name=gnode.id)

        return builder.build()

def main():
    app = QtWidgets.QApplication(sys.argv)

    window = Window()
    window.show()

    app.exec_()

