
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

    def fixed(self) -> bool:
        """ Are all parameters of the type fixed. If a variable is fixed, then its values can be
        stacked together in a tensor.
        """
        return True

    def constant(self) -> bool:
        """ Is a returned value constant or will it change for different samples.
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

class Number(Type):
    """Base class for integer or float values.
    """

    def __init__(self, value=None):
        self._value = value

    @property
    def value(self):
        """Returns value if known, or None.

        """
        return self._value

    def castable(self, typ: Type):
        if not isinstance(typ, Number):
            return False

        if self._value is not None:
            return self._value == typ.value

        return True

    def constant(self):
        """ A value is constant if its value is known at compile time.
        """
        return self._value is not None

    def __str__(self):
        return super().__str__() + " ({})".format(self._value)

class Integer(Number):

    def castable(self, typ: Type):
        if not isinstance(typ, Integer):
            return False

        if self._value is not None:
            return self._value == typ.value

        return True

class Float(Number):

    pass

class View(Type):
    """Represents a 3x3 linear transformation matrix.
    """

    def castable(self, typ: Type):
        return isinstance(typ, View)

class Points(Type):
    """Represents a tuple of points.
    """

    def __init__(self, length=None):
        self._length = length

    @property
    def length(self):
        return self._length

    def castable(self, typ: Type):
        if not isinstance(typ, Points):
            return False

        return self._length is None or self._length == typ.length

    def fixed(self) -> bool:
        return self._length is not None

class ImagePurpose(Enum):
    VISUAL = 1
    MASK = 2
    HEATMAP = 3

class Image(Type):
    """Represents an image type. This type can be specialized with image width, height, number of channels as well
    as bit-depth.
    """

    def __init__(self, width: typing.Optional[int] = None, height: typing.Optional[int] = None,
        channels: typing.Optional[int] = None, depth: typing.Optional[int] = None, purpose: ImagePurpose = ImagePurpose.VISUAL):

        self._width = width
        self._height = height
        self._channels = channels
        self._depth = depth
        self._purpose = purpose

    def fixed(self):
        return self._width is not None and self._height is not None \
            and self._channels is not None and self._depth is not None

    @property
    def width(self) -> typing.Union[int, None]:
        return self._width

    @property
    def height(self) -> typing.Union[int, None]:
        return self._height

    @property
    def channels(self) -> typing.Union[int, None]:
        return self._channels

    @property
    def depth(self) -> typing.Union[int, None]:
        return self._depth

    @property
    def purpose(self) -> ImagePurpose:
        return self._purpose

    def castable(self, typ: Type):
        if not isinstance(typ, Image):
            return False

        if self._width is not None:
            if self._width != typ.width:
                return False

        if self._height is not None:
            if self._height != typ.height:
                return False

        if self._channels is not None:
            if self._channels != typ.channels:
                return False

        if self._depth is not None:
            if self._depth != typ.depth:
                return False

        return True

    def __str__(self):
        return super().__str__() + " ({} x {} x {}, {} bit)".format(self._width, self._height, self._channels, self._depth)


class List(Type):
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
        if element_type is not None:
            if not isinstance(element_type, Type):
                raise TypeException("Incorrect object type for element type: {}".format(element_type))
            if isinstance(element_type, Complex):
                raise TypeException("Complex type cannot be an element of a list")
        self._type = element_type
        self._length = length

    def fixed(self):
        return self._type is not None and self._length is not None

    @property
    def element(self):
        return self._type

    @property
    def length(self):
        return self._length

    def castable(self, typ: Type):
        if not isinstance(typ, List):
            return False

        if self._type is not None:
            if not self._type.castable(typ.element):
                return False

        if self._length is not None:
            if self._length != typ.length:
                return False

        return True

def BoundingBox():
    return List(Float(), 4)

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