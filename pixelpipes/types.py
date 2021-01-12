
import typing
from enum import Enum

class TypeException(Exception):

    pass

class Type(object):
    """Abstract type base, represents description of variable types accepted or returned by nodes.
    """

    def castable(self, typ: "Type") -> bool:
        """Can object of given input type description be casted to this type.
        """
        raise NotImplementedError()

    def common(self, typ: "Type") -> "Type":
        """Merge two types by finding their common archi-type. By default this just looks
        if one type is castable into the other.
        """
        if self.castable(typ):
            return self
        elif typ.castable(self):
            return typ
        else:
            return Any()

    def fixed(self) -> bool:
        """ Are all parameters of the type fixed. If a variable is fixed, then its values can be
        stacked together in a tensor.
        """
        return True

    def constant(self) -> bool:
        """ Is a returned value constant or will it change for different samples?
        """
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__class__.__name__

class Any(Type):
    """Denotes type that accepts all inputs.
    """

    def castable(self, typ: Type):
        return True

class Union(Type):
    """Denotes type that accepts any of the given inputs.
    """

    def __init__(self, *args: Type):
        self._union = args

    def castable(self, typ: Type) -> bool:
        for subtype in self._union:
            if subtype.castable(typ):
                return True
        return False

    def fixed(self) -> bool:
        return False

    def common(self, typ: "Type") -> "Type":
        for subtype in self._union:
            common = subtype.common(typ)
            if not isinstance(common, Any):
                return common
        return Any()

class ParametricType(Type):

    def __init__(self, *parameters):
        super().__setattr__("_parameters", {parameter: None for parameter in parameters})

    def _set(self, key, value):
        self._parameters[key] = value

    def __setattr__(self, key, value):
        if key in self._parameters:
            raise TypeException("Parameter value {} is readonly".format(key))
        super().__setattr__(key, value)

    def __getattr__(self, key):
        if key in self._parameters:
            return self._parameters[key]
        return object.__getattribute__(self, key)

    def _common_parameters(self, typ: "ParametricType"):
        common = {}

        for k in self._parameters:
            pa = getattr(self, k)
            pb = getattr(typ, k)

            common[k] = pa if pa == pb else None
        return common

    def _compare_parameters(self, typ: "ParametricType"):
        for k in self._parameters:
            if (getattr(self, k) is not None
                and getattr(self, k) != getattr(typ, k)):
                return False

        return True

    def fixed(self):

        for k in self._parameters:
            if getattr(self, k) is None:
                return False

        return True

class Number(ParametricType):
    """Base class for integer or float values.
    """

    def __init__(self):
        super().__init__("value")

    def castable(self, typ: Type):
        return isinstance(typ, Number) and self._common_parameters(typ)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Number):
            return Number()
        else:
            return Any()

    def constant(self):
        """ A value is constant if its value is known at compile time.
        """
        return self.fixed()

    def __str__(self):
        return super().__str__() + " ({})".format(self.value)

class Integer(Number):

    def __init__(self, value: int = None):
        super().__init__()
        self._set("value", int(value) if value is not None else None)

    def castable(self, typ: Type):
        return isinstance(typ, Integer) and self._common_parameters(typ)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Integer):
            return Integer(**self._common_parameters(typ))
        else:
            return super().common(typ)

class Float(Number):

    def __init__(self, value: float = None):
        super().__init__()
        self._set("value", float(value) if value is not None else None)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Float):
            return Float(**self._common_parameters(typ))
        else:
            return super().common(typ)

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

class ImagePurpose(Enum):
    VISUAL = 1
    MASK = 2
    HEATMAP = 3

class Image(ParametricType):
    """Represents an image type. This type can be specialized with image width, height, number of channels as well
    as bit-depth.
    """

    def __init__(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None,
        channels: typing.Optional[int] = None, depth: typing.Optional[int] = None,
        purpose: typing.Optional[ImagePurpose] = None):

        super().__init__("width", "height", "channels", "depth", "purpose")

        self._set("width", width)
        self._set("height", height)
        self._set("channels", channels)
        self._set("depth", depth)
        self._set("purpose", purpose)

    def __str__(self):
        return super().__str__() + " ({} x {} x {}, {} bit for {})".format(self.width, self.height, self.channels, self.depth, self.purpose)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Image):
            return Image(**self._common_parameters(typ))
        else:
            return Any()

    def castable(self, typ: Type):
        return isinstance(typ, Image) and self._common_parameters(typ)

class List(ParametricType):
    """Type that represents a list of elements.
    """

    def __init__(self, element_type: typing.Optional[Type] = None, length: typing.Optional[int] = None):
        """[summary]

        Args:
            element_type (typing.Optional[Type], optional): Type of elements in a list. Elements cannot be complex. Defaults to None.
            length (typing.Optional[int], optional): Length of list if known in advance. Defaults to None.

        Raises:
            TypeException: in case of incorrect element type.
        """
        super().__init__("length")
        if element_type is not None:
            if not isinstance(element_type, Type):
                raise TypeException("Incorrect object type for element type: {}".format(element_type))
            if isinstance(element_type, Complex):
                raise TypeException("Complex type cannot be an element of a list")
        self._type = element_type if element_type is not None else Any()
        self._set("length", length)

    def fixed(self):
        return super().fixed() and self._type is not None

    @property
    def element(self):
        return self._type

    def castable(self, typ: Type):
        return isinstance(typ, List) and self._common_parameters(typ) and self._type.castable(typ.element)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, List):
            return List(element_type=self._type.common(typ.element), **self._common_parameters(typ))
        else:
            return Any()

def BoundingBox():
    return List(Float(), 4)

def Rectangle():
    return List(Float(), 4)

def Points(length=None):
    return List(Point(), length)

def Primitive():
    return Union(Number(), Image(), View(), Point(), List())

class Complex(Type):
    """Base class for all non-primitive types. Complex type is essentially a flat structure key-value type.
    """

    def __init__(self, elements: typing.Optional[typing.Dict[str, Type]] = None):
        """[summary]

        Args:
            elements (typing.Optional[typing.Dict[str, Type]], optional): Type structure. Defaults to None.

        Raises:
            TypeException: [description]
        """
        if elements is not None:
            if any([isinstance(x, Complex) for x in elements.values()]):
                raise TypeException("Complex types cannot be nested")
            assert all([isinstance(x, Type) for x in elements.values()])
        self._elements = elements

    def castable(self, typ: "Type"):
        if not isinstance(typ, Complex):
            return False

        if self._elements is None:
            return True

        for k, v in self._elements.items():
            if not k in typ:
                return False
            if not v.castable(typ[k]):
                return False

        return True

    def _merge_elements(self, typ: "Complex"):
        return {k: v.common(typ[k]) for k, v in self._elements.items() if k in typ}

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Complex):
            return Complex(self._merge_elements(typ))
        else:
            return Any()

    def fixed(self):
        if self._elements is None:
            return False
        return all([e.fixed() for e in self._elements.values()])

    def constant(self):
        if self._elements is None:
            return False
        return all([e.constant() for e in self._elements.values()])

    def __contains__(self, key):
        if self._elements is None:
            return False
        return key in self._elements

    def __getitem__(self, key):
        if self._elements is None:
            return None
        return self._elements[key]

    def elements(self):
        return self._elements.items()

    def access(self, element: str, parent: "Reference"):
        from pixelpipes import Reference
        if not element in self:
            raise TypeException("Element {} not found in complex resource".format(element))
        return Reference(parent + "." + element)