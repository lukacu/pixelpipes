
import typing

from attributee import String, Unclaimed, Collector

from ..types import Integer, Token, Data, Anything, Wildcard
from ..graph import InferredReference, NodeOperation, Node, Macro, Input, Reference, Copy, ValidationException, hidden
from ..types import TypeException

IMAGE_FIELD = "image"
MASK_FIELD = "mask"
POINTS_FIELD = "points"

class InputCollector(Unclaimed):

    def __init__(self, field, description=""):
        super().__init__(description=description)
        self._field = field

    def filter(self, object, **kwargs):
        attributes = object.attributes()
        claimed = set()
        for aname, afield in attributes.items():
            if isinstance(afield, Collector) and not isinstance(afield, Unclaimed):
                claimed.update(afield.filter(object, **kwargs).keys())
            elif aname in kwargs:
                claimed.add(aname)
        return {k: self._field.coerce(v, None) for k, v in kwargs.items() if k not in claimed}

class ResourceField:
    def __init__(self, typ: Data, purpose: typing.Optional[str] = None):
        self._type = typ
        self._purpose = purpose

    def access(self, parent: "InferredReference"):
        return None

    def reference(self, parent: "InferredReference"):
        return None

    @property
    def type(self):
        return self._type

    @property
    def purpose(self):
        return self._purpose

def real_field(field):
    return field.reference(Reference("test")) is not None

class Resource(Data):
    """Support for virtual resource types. Resource is essentially a flat structure key-value type that simplifies handling
    several inputs in parallel. Their purpose is to structure dataflow at the graph level but get dissolved once the graph is compiled.
    """

    def __init__(self, **fields: typing.Dict[str, ResourceField]):
        """Creates a new resource type by defining all its fields.

        Args:
            fields (typing.Dict[str, TensorType]): Type structure.

        Raises:
            TypeException: [description]
        """
        if len(fields) > 0:
            for x in fields.values():
                if not isinstance(x, ResourceField):
                    raise TypeException("Must be a resouce field: {}".format(x))
        self._fields = fields

    def castable(self, typ: "Data") -> bool:
        """Checks if one resource type can be cast to another, this means that the parameter
        type has all the fields of this type and that all field types can be casted to their
        corresponding field types.

        Args:
            typ (Type): Type to test for compatibility

        Returns:
            bool: True if type is compatible
        """
        if not isinstance(typ, Resource):
            return False

        if self._fields is None:
            return True

        for k, _ in self._fields.items():
            if not k in typ:
                return False

        return True

    def common(self, typ: "Data") -> "Data":
        if isinstance(typ, Resource):
            fields = {k: v for k, v in self._fields.items() if k in typ}
            return Resource(**fields)
        else:
            return Anything()

    def __contains__(self, key):
        if self._fields is None:
            return False
        return key in self._fields

    def __getitem__(self, key):
        if not key in self:
            raise KeyError("Field {} does not exist in the resource".format(key))
        return self._fields[key]

    def __iter__(self):
        for k, v in self._fields.items():
            yield (k, v)

    def fields(self):
        return dict(**self._fields)

    def access(self, field: str, parent: "Reference"):
        if not field in self:
            raise TypeException("Field {} not found in resource".format(field))
        
        aa = self._fields[field].access(parent)
        return aa

    def typehint(self, field: str):
        return Token()

    def __str__(self):
        return "Resource (" + ", ".join(self._fields.keys()) + ")"


class TokenField(ResourceField):

    def __init__(self, field: str):
        self._field = field

    def access(self, parent: "InferredReference"):
        if not isinstance(parent.type, Resource) and not self._field in parent.type:
            raise ValidationException("Not a resource or nonexistent field")
        return Reference(parent.name + "." + self._field)

    def reference(self, parent: "InferredReference"):
        return Reference(parent.name + "." + self._field)

class AliasField(ResourceField):

    def __init__(self, field: str):
        self._field = field

    def access(self, parent: "InferredReference"):
        if not isinstance(parent.type, Resource) and not self._field in parent.type:
            raise ValidationException("Not a resource or nonexistent field")
        return Reference(parent.name + "." + self._field)

class ConditionalField(ResourceField):

    def __init__(self, true_field: ResourceField, false_field: ResourceField, true_filter: str, false_filter: str, condition: Reference):
        self._condition = condition
        self._true_field = true_field
        self._false_field = false_field
        self._true_filter = true_filter
        self._false_filter = false_filter

    def access(self, parent: "InferredReference"):
        from pixelpipes.flow import Conditional

        true_aliases = dict(**parent.type.fields())

        for name, _ in parent.type:
            if name.startswith(self._true_filter):
                true_aliases[name[len(self._true_filter):]] = AliasField(name)
                
        false_aliases = dict(**parent.type.fields())

        for name, _ in parent.type:
            if name.startswith(self._false_filter):
                false_aliases[name[len(self._false_filter):]] = AliasField(name)

        true_parent = InferredReference(parent.name, Resource(**true_aliases))
        false_parent = InferredReference(parent.name, Resource(**false_aliases))

        return Conditional(self._true_field.access(true_parent), self._false_field.access(false_parent), self._condition)

@hidden
class ResourceProxy(Node):

    inputs = InputCollector(Input(Wildcard()), description="A map of inputs that are provide data to the resource")

    def __init__(self, *args, _fields=None, **kwargs):
        super().__init__(*args, **kwargs)
        if _fields is None:
            _fields = {}
        self._fields = _fields
        for key in self.inputs.keys():
            if key not in self._fields:
                self._fields[key] = TokenField(key)

    def input_values(self):
        return [self.inputs[name] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(k, Wildcard()) for k in self.inputs.keys()]

    def evaluate(self, **inputs):
    #    for name, ref in inputs.items():
    #        if name in self._fields
        return Resource(**self._fields)

class MakeResource(Macro):
    """Macro that generates a resource from given inputs
    """

    inputs = InputCollector(Input(Wildcard()), description="A map of inputs that are inserted into the expression")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def input_values(self):
        return [self.inputs[name] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(k, Wildcard()) for k in self.inputs.keys()]

    def expand(self, **inputs):
  
        forward = {}
        fields = {}
        
        # Only copy real fields
        for name, reference in inputs.items():
            fields[name] = TokenField(name)
            forward[name] = reference
            Copy(source=reference, _name="." + name)
        return ResourceProxy(_fields=fields, **forward)

class AppendField(Macro):
    """Produce a resource from an input resource and another field. Essentially just node renaming."""

    source = Input(Resource(), description="Original resource")
    name = String(description="Name of new field")
    value = Input(Wildcard(), description="Value for new field")

    def expand(self, source, value):
        
        resource_type = source.type

        forward = {}
        fields = resource_type.fields()
        
        # Only copy real fields
        for name, field in resource_type:
            reference = field.reference(source)
            if reference is not None:
                forward[name] = reference
                Copy(source=reference, _name="." + name)
            
        Copy(value, _name="." + self.name)
        forward[self.name] = value
        fields[self.name] = TokenField(self.name)

        return ResourceProxy(_fields=fields, **forward)

class ConditionalResource(Macro):
    """Node that executes conditional selection, output of branch "true" will be selected if
    the "condition" is not zero, otherwise output of branch "false" will be selected.
    """

    true = Input(Resource(), description="Use this data if condition is true")
    false = Input(Resource(), description="Use this data if condition is false")
    condition = Input(Integer(), description="Condition to test")

    def expand(self, true, false, condition):

        from pixelpipes.flow import Conditional

        true_type = true.type
        false_type = false.type

        common_type = true_type.common(false_type)

        forward = {}
        fields = {}

        true_filter = "__pass_true_" + condition.name+ "_" + true.name + "_"
        false_filter = "__pass_false_" + condition.name+ "_" + false.name + "_"

        for name, _ in common_type:
            if real_field(true_type[name]) and real_field(false_type[name]):
                true_ref = true_type[name].access(true)
                false_ref = false_type[name].access(false)
                forward[name] = Conditional(true=true_ref, false=false_ref, condition=condition, _name="." + name)
            else:
                if real_field(true_type[name]):
                    hidden = "__cond_true_" + condition.name + "_" + true.name
                    forward[hidden] = Copy(true_type[name].reference(true), _name = "." + hidden)
                    true_field = AliasField(hidden)
                else:
                    true_field = true_type[name]

                if real_field(false_type[name]):
                    hidden = "__cond_false_" + condition.name + "_" + false.name
                    forward[hidden] = Copy(false_type[name].reference(false), _name = "." + hidden)
                    false_field = AliasField(hidden)
                else:
                    false_field = false_type[name]

                fields[name] = ConditionalField(true_field, false_field, true_filter, false_filter, condition)

        for name, _ in true_type:
            if not name in common_type:
                if name.startswith("__"):
                    hidden = name
                else:
                    hidden = true_filter + name
                forward[hidden] = Copy(true_type[name].reference(true), _name = "." + hidden)

        for name, _ in false_type:
            if not name in common_type:
                if name.startswith("__"):
                    hidden = name
                else:
                    hidden = false_filter + name
                forward[hidden] = Copy(false_type[name].reference(false), _name = "." + hidden)

        return ResourceProxy(_fields=fields, **forward)

@hidden
class CopyResource(Macro):

    source = Input(Resource())

    def expand(self, source):

        resource_type = source.type

        forward = {}
        fields = resource_type.fields()
        
        # Only copy real fields
        for name, field in resource_type:
            reference = field.reference(source)
            if reference is not None:
                forward[name] = reference
                Copy(source=reference, _name="." + name)
    
        return ResourceProxy(_name=".", _fields=fields, **forward)


class GetField(Macro):
    """This macro exposes only selected field of an input structure as an output, enabling processing of that data.
    """

    source = Input(Resource(), description="Input resource")
    element = String(description="Name of the structure field")

    def expand(self, source):
        from .list import is_resource_list

        if is_resource_list(source.type) and not real_field(source.type[self.element]):
            raise TypeException("Fields can only be accessed on a single resource, not a list")

        return Copy(source=source.type[self.element].access(source))

from .. import types

Node.register_operation(NodeOperation.INDEX, GetField, Resource(), types.String())
