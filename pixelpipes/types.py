
import typing

class TypeException(Exception):
    pass

_cast_hierarchy = {
    "bool" : ["uchar", "char", "ushort", "short", "int", "float"],
    "char" : ["uchar", "ushort", "short", "int", "float"],
    "uchar" : ["char", "ushort", "short", "int", "float"],
    "short" : ["int", "ushort", "float"],
    "ushort" : ["int", "short", "float"],
    "int" : ["float"],
    "float": []
}

def cast_element(source: str, destination: str):
    if destination is None:
        return source
    if source == destination:
        return source
    if source in _cast_hierarchy:
        if destination in _cast_hierarchy[source]:
            return source
        return None
    else:
        return destination if source == destination else None

def convert_element(element):
    import numpy as np
    if element is None:
        return None
    if isinstance(element, str):
        if element == "":
            return None
        return element
    if isinstance(element, np.dtype):
        if element == np.uint8:
            return "uchar"
        if element == np.int8:
            return "char"
        if element == np.uint16:
            return "ushort"
        if element == np.uint16:
            return "ushort"
        if element == np.int16:
            return "short"
        if element == np.int32:
            return "int"
        if element == np.float32:
            return "float"
        raise TypeException("Unsupported NumPy type: " + element)

class Data(object):
    """Abstract type base, represents description of token types accepted or returned by nodes.
    """

    def castable(self, typ: "Data") -> bool:
        """Can object of given input type description be casted to this type.
        """
        raise NotImplementedError()

    def common(self, typ: "Data") -> "Data":
        """Merge two types by finding their common type. By default this just looks
        if one type is castable into the other.
        """
        if self.castable(typ):
            return self
        elif typ.castable(self):
            return typ
        else:
            return Anything()

    def __str__(self):
        return "Data"


class Anything(Data):
    """Denotes type that accepts all inputs.
    """

    def castable(self, typ: Data):
        return True

    def __str__(self):
        return "Anything"

class Union(Data):
    """Denotes type that accepts any of the given inputs. Do not nest unions.
    """

    def __init__(self, *args: Data):
        self._union = args

    def castable(self, typ: Data) -> bool:
        for subtype in self._union:
            if subtype.castable(typ):
                return True
        return False

    def common(self, typ: "Data") -> "Data":
        for subtype in self._union:
            common = subtype.common(typ)
            if not isinstance(common, Anything):
                return common
        return Anything()

    def __str__(self):
        return "[" + "; ".join([str(x) for x in self._union]) + "]"

class Token(Data):

    def __init__(self, element = None, *shape):
        self._element = convert_element(element)
        self._shape = shape

    @property
    def element(self):
        return self._element

    def dimension(self, i):
        if i < len(self._shape):
            return self._shape[i]
        else:
            return 1

    def __getitem__(self, i):
        return self.dimension(i)

    @property
    def dimensions(self):
        return len(self._shape)

    def castable(self, typ: "Data") -> bool:
        if isinstance(typ, Token):
            a = self
            b = typ.squeeze()
            # Test if element castable
            if cast_element(b.element, a.element) is None and b.element is not None:
                return False
            if b.dimensions > a.dimensions:
                return False
            for d in range(a.dimensions):
                da = a.dimension(d)
                db = b.dimension(d)
                if da is not None and db is not None and da != db:
                    return False
            return True
        return False
    
    def common(self, typ: "Data") -> "Data":
        if isinstance(typ, Token):
            a = self
            b = typ.squeeze()
            shape = []
            for d in range(a.dimensions):
                da = a.dimension(d)
                db = b.dimension(d)
                shape.append(da if da == db else None)
            return Token(cast_element(b.element, a.element), *shape)
        return Anything()

    def __str__(self):
        if len(self._shape) > 0:
            return "Token [%s, %s]" % (self._element, " x ".join([str(x) for x in self._shape]))
        else:
            return "Token [%s]" % self._element

    def pop(self):
        if len(self._shape) > 0:
            return Token(self._element, *self._shape[1:]) 
        else:
            return Token(self._element) 

    def push(self, length=None):
        return Token(self._element, length, *self._shape) 

    def squeeze(self):

        if self.dimensions == 0:
            return self

        nonzero = [i for i, e in enumerate(self._shape) if e != 1]

        if not nonzero:
            return Token(self._element)

        return Token(self._element, *self._shape[:nonzero[-1]+1])

class Wildcard(Token):

    def __init__(self, element = None, mindim = None, maxdim = None):
        self._element = convert_element(element)
        self._shape = []
        self._maxdim = maxdim
        self._mindim = mindim

    def castable(self, typ: "Data") -> bool:
        if isinstance(typ, Token):
            if cast_element(typ.element, self.element) is None and typ.element is not None:
                return False
            if self._mindim is not None and typ.dimensions < self._mindim:
                return False
            if self._maxdim is not None and typ.dimensions > self._maxdim:
                return False
            return True
        return False

    def common(self, typ: "Data") -> "Data":
        if self.castable(typ):
            return typ
        return Anything()

def Integer():
    return Token("int")

def Float():
    return Token("float")

def Boolean():
    return Token("bool")

def Char():
    return Token("char")

def Short():
    return Token("short")

def UnsignedChar():
    return Token("uchar")

def UnsignedShort():
    return Token("ushort")

def Image(width: typing.Optional[int] = None, height: typing.Optional[int] = None,
        channels: typing.Optional[int] = None, depth: typing.Optional[str] = None):
    """Represents an image type. This type can be specialized with image width, height, number of channels as well
    as bit-depth.
    """
    return Token(depth, height, width, channels)

def List(element = None, length = None):
    """Type that represents a list of elements.
    """
    return Token(element, length)

def FloatList(length = None):
    return Token("float", length)

def IntegerList(length = None):
    return Token("int", length)        

def BooleanList(length = None):
    return Token("bool", length)

def String(length = None):
    return Token("char", length)

def View():
    return Token("float", 3, 3)

def Point():
    return Token("float", 2)

def Rectangle():
    return List("float", 4)

def Points(length=None):
    return Token("float", length, 2)
