#!/usr/bin/python
# -*- coding: utf-8 -*-
import Qt

from NodeGraphQt import QtWidgets, QtCore, QtGui
from NodeGraphQt.constants import DRAG_DROP_ID


TYPE_NODE = QtWidgets.QListWidgetItem.UserType + 1

ROLE_NODE_ID = QtCore.Qt.UserRole + 1
ROLE_NODE_CATEGORY = QtCore.Qt.UserRole + 2

class NodeListItem(QtWidgets.QListWidgetItem):

    def __eq__(self, other):
        """
        Workaround fix for QTreeWidgetItem "operator not implemented error".
        see link: https://bugreports.qt.io/browse/PYSIDE-74
        """
        return id(self) == id(other)


class NodeListWidget(QtWidgets.QListWidget):
    """
    Node tree for displaying node types.

    Args:
        parent (QtWidgets.QWidget): parent of the new widget.
        node_graph (NodeGraphQt.NodeGraph): node graph.
    """

    def __init__(self, parent=None, node_graph=None):
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        #self.setHeaderHidden(True)
        self._icon_cache = {}
        self._icon_cache["numeric"] = QtGui.QIcon.fromTheme("applications-science")
        self._icon_cache["list"] = QtGui.QIcon.fromTheme("list-add")
        self._icon_cache["default"] = QtGui.QIcon.fromTheme("folder-new")
        self._build_tree(node_graph._node_factory)

    def __repr__(self):
        return '<{} object at {}>'.format(self.__class__.__name__, hex(id(self)))

    def mimeData(self, items):
        node_ids = ','.join(i.data(ROLE_NODE_ID) for i in items)
        mime_data = super().mimeData(items)
        mime_data.setText('<${}>:{}'.format(DRAG_DROP_ID, node_ids))
        return mime_data

    def _build_tree(self, factory):
        """
        Populate the node tree.
        """
        for _, node in factory.nodes.items():
            item = NodeListItem(node.name(), self, type=TYPE_NODE)
            item.setData(ROLE_NODE_ID, node.type_)
            item.setData(ROLE_NODE_CATEGORY, node.category())
            icon = self._icon_cache.get(node.category(), self._icon_cache["default"])
            item.setIcon(icon)
            self.addItem(item)
