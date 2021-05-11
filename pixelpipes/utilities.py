

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