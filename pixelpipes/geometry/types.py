




from pixelpipes.types import List, Type, Float


class View(Type):
    """Represents a 3x3 linear transformation matrix.
    """

    def castable(self, typ: Type):
        return isinstance(typ, View)

class Point(Type):
    """Represents a two-dimensional point.
    """

    def castable(self, typ: Type):
        return isinstance(typ, Point)

def BoundingBox():
    return List(Float(), 4)

def Rectangle():
    return List(Float(), 4)

def Points(length=None):
    return List(Point(), length)