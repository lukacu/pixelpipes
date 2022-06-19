#!/usr/bin/python
from distutils.version import LooseVersion

from NodeGraphQt import QtGui, QtCore

# menu
def setup_context_menu(graph):
    """
    populate the specified graph's context menu with essential menus commands.

    example code:

    .. code-block:: python
        :linenos:

        from NodeGraphQt import NodeGraph, setup_context_menu

        graph = NodeGraph()
        setup_context_menu(graph)

    result:

    .. image:: _images/menu_hotkeys.png
        :width: 300px

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    root_menu = graph.get_context_menu('graph')

    file_menu = root_menu.add_menu('&File')
    edit_menu = root_menu.add_menu('&Edit')

    # create "File" menu.
    file_menu.add_command('Open...', _open_session, QtGui.QKeySequence.Open)
    file_menu.add_command('Import...', _import_session)
    file_menu.add_command('Save...', _save_session, QtGui.QKeySequence.Save)
    file_menu.add_command('Save As...', _save_session_as, 'Ctrl+Shift+S')
    file_menu.add_command('New Graph', _new_session)

    # create "Edit" menu.
    undo_actn = graph.undo_stack().createUndoAction(graph.viewer(), '&Undo')
    if LooseVersion(QtCore.qVersion()) >= LooseVersion('5.10'):
        undo_actn.setShortcutVisibleInContextMenu(True)
    undo_actn.setShortcuts(QtGui.QKeySequence.Undo)
    edit_menu.qmenu.addAction(undo_actn)

    redo_actn = graph.undo_stack().createRedoAction(graph.viewer(), '&Redo')
    if LooseVersion(QtCore.qVersion()) >= LooseVersion('5.10'):
        redo_actn.setShortcutVisibleInContextMenu(True)
    redo_actn.setShortcuts(QtGui.QKeySequence.Redo)
    edit_menu.qmenu.addAction(redo_actn)

    edit_menu.add_separator()
    edit_menu.add_command('Clear Undo History', _clear_undo)
    edit_menu.add_command('Show Undo View', _show_undo_view)
    edit_menu.add_separator()

    edit_menu.add_command('Copy', _copy_nodes, QtGui.QKeySequence.Copy)
    edit_menu.add_command('Cut', _cut_nodes, QtGui.QKeySequence.Cut)
    edit_menu.add_command('Paste', _paste_nodes, QtGui.QKeySequence.Paste)
    edit_menu.add_command('Delete', _delete_items, QtGui.QKeySequence.Delete)

    edit_menu.add_separator()

    edit_menu.add_command('Select all', _select_all_nodes, 'Ctrl+A')
    edit_menu.add_command('Deselect all', _clear_node_selection, 'Ctrl+Shift+A')
    edit_menu.add_command('Enable/Disable', _disable_nodes, 'D')

    edit_menu.add_command('Duplicate', _duplicate_nodes, 'Alt+C')
    edit_menu.add_command('Center Selection', _fit_to_selection, 'F')

# --- menu command functions. ---


def _zoom_in(graph):
    """
    Set the node graph to zoom in by 0.1

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    zoom = graph.get_zoom() + 0.1
    graph.set_zoom(zoom)


def _zoom_out(graph):
    """
    Set the node graph to zoom in by 0.1

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    zoom = graph.get_zoom() - 0.2
    graph.set_zoom(zoom)


def _reset_zoom(graph):
    graph.reset_zoom()


def _open_session(graph):
    """
    Prompts a file open dialog to load a session.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    current = graph.current_session()
    viewer = graph.viewer()
    file_path = viewer.load_dialog(current)
    if file_path:
        graph.load_session(file_path)


def _import_session(graph):
    """
    Prompts a file open dialog to load a session.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    current = graph.current_session()
    viewer = graph.viewer()
    file_path = viewer.load_dialog(current)
    if file_path:
        graph.import_session(file_path)


def _save_session(graph):
    """
    Prompts a file save dialog to serialize a session if required.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    current = graph.current_session()
    if current:
        graph.save_session(current)
        msg = 'Session layout saved:\n{}'.format(current)
        viewer = graph.viewer()
        viewer.message_dialog(msg, title='Session Saved')
    else:
        _save_session_as(graph)


def _save_session_as(graph):
    """
    Prompts a file save dialog to serialize a session.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    current = graph.current_session()
    viewer = graph.viewer()
    file_path = viewer.save_dialog(current)
    if file_path:
        graph.save_session(file_path)


def _new_session(graph):
    """
    Prompts a warning dialog to new a node graph session.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    viewer = graph.viewer()
    if viewer.question_dialog('Clear Current Session?', 'Clear Session'):
        graph.clear_session()


def _clear_undo(graph):
    """
    Prompts a warning dialog to clear undo.

    Args:
        graph (NodeGraphQt.NodeGraph): node graph.
    """
    viewer = graph.viewer()
    msg = 'Clear all undo history, Are you sure?'
    if viewer.question_dialog('Clear Undo History', msg):
        graph.clear_undo_stack()


def _copy_nodes(graph):
    graph.copy_nodes()


def _cut_nodes(graph):
    graph.cut_nodes()


def _paste_nodes(graph):
    graph.paste_nodes()


def _delete_items(graph):
    graph.delete_nodes(graph.selected_nodes())
    graph.delete_pipes(graph._viewer.selected_pipes())


def _select_all_nodes(graph):
    graph.select_all()


def _clear_node_selection(graph):
    graph.clear_selection()


def _disable_nodes(graph):
    graph.disable_nodes(graph.selected_nodes())


def _duplicate_nodes(graph):
    graph.duplicate_nodes(graph.selected_nodes())


def _fit_to_selection(graph):
    graph.fit_to_selection()


def _jump_in(graph):
    nodes = graph.selected_nodes()
    if nodes:
        graph.set_node_space(nodes[0])


def _jump_out(graph):
    node = graph.get_node_space()
    if node:
        if node.parent() is not None:
            graph.set_node_space(node.parent())


def _show_undo_view(graph):
    graph.undo_view.show()


def _bg_grid_none(graph):
    graph.set_grid_mode(0)


def _bg_grid_dots(graph):
    graph.set_grid_mode(1)


def _bg_grid_lines(graph):
    graph.set_grid_mode(2)

def get_input_nodes(node):
    """
    Get input nodes of node.

    Args:
        node (NodeGraphQt.BaseNode).
    Returns:
        list[NodeGraphQt.BaseNode].
    """

    nodes = {}
    for p in node.input_ports():
        for cp in p.connected_ports():
            n = cp.node()
            nodes[n.id] = n
    return list(nodes.values())


def get_output_nodes(node, cook=True):
    """
    Get output nodes of node.

    Args:
        node (NodeGraphQt.BaseNode).
        cook (bool): call this function for cook node.
    Returns:
        list[NodeGraphQt.BaseNode].
    """

    nodes = {}
    for p in node.output_ports():
        for cp in p.connected_ports():
            n = cp.node()
            if cook and n.has_property('graph_rect'):
                n.mark_node_to_be_cooked(cp)
            nodes[n.id] = n
    return list(nodes.values())


def _has_input_node(node):
    """
    Returns whether the node has input node.

    Args:
        node (NodeGraphQt.BaseNode).
    Returns:
        bool.
    """

    for p in node.input_ports():
        if p.view.connected_pipes:
            return True
    return False


def _has_output_node(node):
    """
    Returns whether the node has output node.

    Args:
        node (NodeGraphQt.BaseNode).
    Returns:
        bool.
    """

    for p in node.output_ports():
        if p.view.connected_pipes:
            return True
    return False


def _build_down_stream_graph(start_nodes):
    """
    Build a graph by down stream nodes.

    Args:
        start_nodes (list[NodeGraphQt.BaseNode]).
    Returns:
        dict {node0: [output nodes of node0], ...}.
    """

    graph = {}
    for node in start_nodes:
        output_nodes = get_output_nodes(node)
        graph[node] = output_nodes
        while output_nodes:
            _output_nodes = []
            for n in output_nodes:
                if n not in graph:
                    nodes = get_output_nodes(n)
                    graph[n] = nodes
                    _output_nodes.extend(nodes)
            output_nodes = _output_nodes
    return graph


def _build_up_stream_graph(start_nodes):
    """
    Build a graph by up stream nodes.

    Args:
        start_nodes (list[NodeGraphQt.BaseNode]).
    Returns:
        dict {node0: [input nodes of node0], ...}.
    """

    graph = {}
    for node in start_nodes:
        input_nodes = get_input_nodes(node)
        graph[node] = input_nodes
        while input_nodes:
            _input_nodes = []
            for n in input_nodes:
                if n not in graph:
                    nodes = get_input_nodes(n)
                    graph[n] = nodes
                    _input_nodes.extend(nodes)
            input_nodes = _input_nodes
    return graph


def _sort_nodes(graph, start_nodes, reverse=True):
    """
    Sort nodes in graph.

    Args:
        graph (dict): generate from '_build_up_stream_graph' or '_build_down_stream_graph'.
        start_nodes (list[NodeGraphQt.BaseNode]): graph start nodes.
        reverse (bool): reverse the result.
    Returns:
        list[NodeGraphQt.BaseNode]: sorted nodes.
    """

    if not graph:
        return []

    visit = dict((node, False) for node in graph.keys())

    sorted_nodes = []

    def dfs(start_node):
        for end_node in graph[start_node]:
            if not visit[end_node]:
                visit[end_node] = True
                dfs(end_node)
        sorted_nodes.append(start_node)

    for start_node in start_nodes:
        if not visit[start_node]:
            visit[start_node] = True
            dfs(start_node)

    if reverse:
        sorted_nodes.reverse()

    return sorted_nodes


def _update_nodes(nodes):
    """
    Run nodes.

    Args:
        nodes (list[NodeGraphQt.BaseNode]): nodes to be run.
    """
    for node in nodes:
        if node.disabled():
            continue
        try:
            node.run()
        except Exception as error:
            print("Error Update Node : {}\n{}" .format(node, str(error)))
            break



def _update_node_rank_down(node, nodes_rank):
    rank = nodes_rank[node] + 1
    for n in get_output_nodes(node, False):
        if n in nodes_rank:
            nodes_rank[n] = max(nodes_rank[n], rank)
        else:
            nodes_rank[n] = rank
        _update_node_rank_down(n, nodes_rank)


def _compute_rank_down(start_nodes):
    """
    Compute the rank of the down stream nodes.

    Args:
        start_nodes (list[NodeGraphQt.BaseNode]):
            (Optional) the start nodes of the graph.

    Returns:
        dict{NodeGraphQt.BaseNode: node_rank, ...}
    """
    nodes_rank = {}
    for node in start_nodes:
        nodes_rank[node] = 0
        _update_node_rank_down(node, nodes_rank)
    return nodes_rank


def _update_node_rank_up(node, nodes_rank):
    rank = nodes_rank[node] + 1
    for n in get_input_nodes(node):
        if n in nodes_rank:
            nodes_rank[n] = max(nodes_rank[n], rank)
        else:
            nodes_rank[n] = rank
        _update_node_rank_up(n, nodes_rank)


def _compute_rank_up(start_nodes):
    """
    Compute the rank of the up stream nodes.

    Args:
        start_nodes (list[NodeGraphQt.BaseNode]):
            (Optional) the end nodes of the graph.

    Returns:
        dict{NodeGraphQt.BaseNode: node_rank, ...}
    """

    nodes_rank = {}
    for node in start_nodes:
        nodes_rank[node] = 0
        _update_node_rank_up(node, nodes_rank)
    return nodes_rank
