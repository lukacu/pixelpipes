
import os
from pathlib import Path

import typing
import json
import hashlib

from attributee import String
from attributee.containers import Map
from attributee.object import class_fullname


from pixelpipes.utilities import PersistentDict

from ..types import Type
from ..graph import GraphBuilder, SeedInput, ValidationException, hidden, Macro, Input, Reference, Constant, Copy
from pixelpipes.types import List, Image, Complex, Integer, TypeException
import pixelpipes.types as types
from ..list import ListElement, ListLength, ListPermutation, ListRemap, RepeatElement, SublistSelect, ConstantList
from ..numbers import Add, Round, UniformDistribution

_RESOURCE_CACHE = PersistentDict(os.environ.get("PIXELPIPES_CACHE", os.path.join(Path.home(), ".cache/pixelpipes")))

def make_hash(o):
    """
    makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
    """

    sha1 = hashlib.sha1()

    sha1.update(json.dumps(o, sort_keys=True).encode("utf-8"))

    return sha1.hexdigest()

class ResourceField:

    def __init__(self, typ: types.Type):
        self._typ = typ

    @property
    def type(self) -> types.Type:
        return self._typ
class VirtualField(ResourceField):

    def __init__(self, typ: types.Type):
        super().__init__(typ)

    def generate(self, parent, resource):
        raise NotImplementedError()

class Resource(Complex):
    """A resource is a complex type composed from different fields.
    """

    def __init__(self, fields: typing.Mapping[str, types.Type] = None):
        elements = {}
        if fields is not None:
            for k, t in fields.items():
                if k in elements:
                    raise TypeException("Field name {} already used".format(k))
                assert not isinstance(t, VirtualField)
                elements[k] = t
        super().__init__(elements)
        self._fields = fields

    def fields(self) -> typing.Mapping[str, types.Type]:
        return self._fields

    def field(self, name) -> typing.Union[types.Type, VirtualField]:
        return self._fields[name]

    def has(self, name):
        return name in self._fields

    def type(self, field: str) -> types.Type:
        if field not in self._fields:
            raise ValueError("Field does not exist")
        return self._fields[field]

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, Resource):
            return Resource({k: v.common(typ[k]) for k, v in self.fields().items() if typ.has(k)})
        else:
            return super().common(typ)

class ResourceList(Complex):

    def __init__(self, fields: typing.Mapping[str, typing.Union[types.Type, VirtualField]] = None,
            size: typing.Optional[int] = None, meta=None):
        elements = {} if meta is None else dict(**meta)
        if fields is not None:
            for k, t in fields.items():
                if k in elements:
                    raise TypeException("Name {} already used".format(k))
                if isinstance(t, VirtualField):
                    continue
                elements[k] = List(t, size)
        super().__init__(elements)
        self._fields = fields
        self._size = size

    def castable(self, typ: "Type") -> bool:
        if not isinstance(typ, ResourceList):
            return False

        return super().castable(typ)

    @property
    def size(self):
        return self._size

    def element(self):
        return Resource({k : (f if not isinstance(f, VirtualField) else f.type) for k, f in self.fields().items()})

    def type(self, field: str) -> types.Type:
        if field not in self._fields:
            raise ValueError("Field does not exist")
        if isinstance(self._fields[field], VirtualField):
            return self._fields[field].type
        return self._fields[field]

    def fields(self) -> typing.Mapping[str, typing.Union[types.Type, VirtualField]]:
        return self._fields

    def field(self, name) -> typing.Union[types.Type, VirtualField]:
        return self._fields[name]

    def meta(self) -> typing.Mapping[str, typing.Union[types.Type, VirtualField]]:
        return {k: v for k, v in self.elements.items() if k not in self._fields}

    def virtual(self, field: str) -> bool:
        return field in self._fields and isinstance(self._fields[field], VirtualField)

    def has(self, name):
        return name in self._fields

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, ResourceList):
            fields = {k: v.common(typ.field(k)) for k, v in self.fields().items() if typ.has(k)}
            size = self.size if self.size == typ.size else None
            return ResourceList(fields, size=size)
        else:
            return super().common(typ)

class SegmentedResourceList(ResourceList):

    def __init__(self, segments=None, fields=None):
        meta = {}
        if segments is None:
            size = None
            meta = dict(_begin=List(Integer(), None), _end=List(Integer(), None), _length=List(Integer(), None))
        else:
            size = sum(segments)
            meta = dict(_begin=List(Integer(), len(segments)), \
                _end=List(Integer(), len(segments)), _length=List(Integer(), len(segments)))
        super().__init__(fields=fields, size=size, meta=meta)
        self._segments = segments

    def segment(self, index):
        if self._segments is None:
            return None
        return self._segments[index]

    def total(self):
        if self._segments is None:
            return None
        return len(self._segments)

    def common(self, typ: "Type") -> "Type":
        if isinstance(typ, SegmentedResourceList):
            fields = {k: v.common(typ.field(k)) for k, v in self.fields().items() if typ.has(k)}
            return SegmentedResourceList(fields=fields)
        else:
            return super().common(typ)

@hidden
class ResourceListSource(Macro):

    def _init(self):
        self._cache_id = class_fullname(self) + "@" + make_hash(self.dump())

    def _output(self):
        return ResourceList(self.fields(), self.size)

    @property
    def size(self):
        data = self._get_data()
        return data["size"]

    def expand(self, inputs, parent: "Reference"):
        data = self._get_data()

        with GraphBuilder(prefix=parent) as builder:
            for field, typ in self.fields().items():
                if isinstance(typ, VirtualField):
                    continue
                if field not in data["lists"]:
                    raise ValueError("Real field not backed up by a list")
                constructor = data["lists"][field][0]
                args = data["lists"][field][1:]
                constructor(*args, _name="." + field)
            return builder.nodes()

    def _get_data(self):
        if not self._cache_id in _RESOURCE_CACHE:
            data = self._load()
            _RESOURCE_CACHE[self._cache_id] = data
            return data
        return _RESOURCE_CACHE[self._cache_id]

    def fields(self):
        raise NotImplementedError()

    def _load(self):
        raise NotImplementedError()

@hidden
class SegmentedResourceListSource(ResourceListSource):

    def segments(self):
        data = self._get_data()
        return data["segments"]

    def _output(self):
        return SegmentedResourceList(segments=self.segments(), fields=self.fields())

    def expand(self, inputs, parent: "Reference"):
        graph = super().expand(inputs, parent)

        with GraphBuilder(prefix=parent) as builder:

            segments = self.segments()
            beginnings = [0]
            endings = [segments[0]-1]
            for l in segments[1:]:
                beginnings.append(endings[-1]+1)
                endings.append(endings[-1]+l)

            ConstantList(beginnings, _name="._begin")
            ConstantList(endings, _name="._end")
            ConstantList(segments, _name="._length")

            graph.update(builder.nodes())
            return graph

    def fields(self):
        raise NotImplementedError()

    def _load(self):
        raise NotImplementedError()

class GetResourceListLength(Macro):

    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.Integer(value=inputs["resources"].size)

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                Constant(value=resources_type.size, _name=parent)
            else:
                real_fields = [n for n in resources_type.fields() if not resources_type.virtual(n)]
                field = real_fields[0]
                length = ListLength(parent=resources_type.access(field, inputs["resources"]))
                Add(a=length, b=int(-1), _name=parent)

            return builder.nodes()

class GetResource(Macro):
    
    resources = Input(ResourceList())
    index = Input(Integer())

    def validate(self, **inputs):
        super().validate(**inputs)
        
        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resources_type = inputs["resources"].type

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    ListElement(parent=resources_type.access(field, inputs["resources"]), index=self.index, _name="." + field)

            return builder.nodes()

class RepeatResource(Macro):
    """Repeat resource

    Returns a list of resources where an input resource is repeated a number of times

    Inputs:
     - resource (Resource)
     - length (Integer)

    Category: resources, sampling
    """

    resource = Input(Resource())
    length = Input(Integer())

    def validate(self, **inputs):
        super().validate(**inputs)
        return ResourceList(inputs["resource"].fields(), inputs["length"].value)

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_type = inputs["resource"].type

            for field, _ in resource_type.fields().items():
                field_source = resource_type.access(field,  inputs["resource"])
                RepeatElement(source=field_source, length=inputs["length"], _name="." + field)

            return builder.nodes()

class GetLastResource(Macro):
    
    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                last = resources_type.size - 1
            else:
                length = GetResourceListLength(inputs["resources"])
                last = Add(a=length, b=int(-1))

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    ListElement(parent=resources_type.access(field, inputs["resources"]), index=last, _name="." + field)

            return builder.nodes()

class GetRandomResource(Macro):
    """Random resource

    Select a random resource from an input list of resources

    Inputs:
     - resources: Resource list that is sampled for elements

    Category: resources, sampling
    """

    resources = Input(ResourceList())
    seed = SeedInput()

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                generator = UniformDistribution(min=0, max=resources_type.size-1, seed=self.seed)
            else:
                length = GetResourceListLength(inputs["resources"])
                generator = UniformDistribution(min=0, max=length-1, seed=self.seed)

            index = Round(generator)

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    ListElement(parent=resources_type.access(field, inputs["resources"]), index=index, _name="." + field)

            return builder.nodes()

class PermuteResources(Macro):
    """Permute resource list

    Randomly permutes the resource list

    Inputs:
     - resources: Resource list that is sampled for elements

    Category: resources, sampling
    """

    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["resources"]

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                indices = ListPermutation(resources_type.size-1)
            else:
                length = GetResourceListLength(inputs["resources"])
                indices = ListPermutation(min=0, max=length-1)

            for field, _ in resources_type.fields().items():
                if not resources_type.virtual(field):
                    ListRemap(field, indices, _name="." + field)

            return builder.nodes()
class ExtractField(Macro):
    """Extract field from resource

    This macro exposes only selected field of an input resource as an output,
    enabling processing of that data.

    Inputs:
     - resource: Input resource
     - field: Name of the resource field

    Category: resource
    """

    resource = Input(Resource())
    field = String()

    def validate(self, **inputs):
        super().validate(**inputs)

        if not self.field in inputs["resource"].fields():
            raise ValueError("Field {} not in resource".format(self.field))

        return inputs["resource"].type(self.field)

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_type = inputs["resource"].type
            Copy(source=resource_type.access(self.field, inputs["resource"]), _name=parent)

            return builder.nodes()

class GetResourceInterval(Macro):
    
    resources = Input(ResourceList())
    begin = Input(Integer())
    end = Input(Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        size = None
        if inputs["begin"].constant() and inputs["end"].constant():
            size = inputs["end"].value - inputs["begin"].value
            assert size >= 0

        return ResourceList(fields=inputs["resources"].fields(), size=size)

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue

                SublistSelect(parent=resources_type.access(field, inputs["resources"]), \
                    begin=inputs["begin"], end=inputs["end"], _name="." + field)

            return builder.nodes()

class GetResourceSegment(Macro):
    
    resources = Input(SegmentedResourceList())
    index = Input(Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        size = None
        if inputs["index"].constant():
            size = inputs["resources"].segment(inputs["index"].value)

        return ResourceList(fields=inputs["resources"].fields(), size=size)

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            begin_reference = ListElement(parent=resources_type.access("_begin", inputs["resources"]), index=self.index)
            end_reference = ListElement(parent=resources_type.access("_end", inputs["resources"]), index=self.index)

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                SublistSelect(parent=resources_type.access(field, inputs["resources"]), begin=begin_reference, end=end_reference, _name="." + field)

            return builder.nodes()

class GetRandomResourceSegment(Macro):
    
    resources = Input(SegmentedResourceList())
    seed = SeedInput()

    def validate(self, **inputs):
        super().validate(**inputs)

        return ResourceList(fields=inputs["resources"].fields(), size=None)

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:


            if resources_type.total() is not None:
                generator = UniformDistribution(min=0, max=resources_type.total()-1, seed=self.seed)
            else:
                field = next(resources_type.fields())
                total = ListLength(parent=resources_type.access("_begin", inputs["resources"]))
                last = Add(a=total, b=int(-1))
                generator = UniformDistribution(min=0, max=last, seed=self.seed)

            index = Round(generator)

            begin_reference = ListElement(parent=resources_type.access("_begin", inputs["resources"]), index=index)
            end_reference = ListElement(parent=resources_type.access("_end", inputs["resources"]), index=index)

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                SublistSelect(parent=resources_type.access(field, inputs["resources"]), begin=begin_reference, end=end_reference, _name="." + field)
            
            return builder.nodes()

class PermuteResourceSegments(Macro):
    
    resources = Input(SegmentedResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["resources"]

    def expand(self, inputs, parent: "Reference"):

        resources_type = inputs["resources"].type

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.total() is not None:
                indices = ListPermutation(resources_type.total()-1)
            else:
                field = next(resources_type.fields())
                total = ListLength(parent=resources_type.access("_begin", inputs["resources"]))
                last = Add(a=total, b=int(-1))
                indices = ListPermutation(max=last)

            ListRemap(resources_type.access("_begin", inputs["resources"]), indices, _name="._begin")
            ListRemap(resources_type.access("_end", inputs["resources"]), indices, _name="._end")
            ListRemap(resources_type.access("_length", inputs["resources"]), indices, _name="._length")

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                Copy(field, _name="." + field)
            
            return builder.nodes()



class MakeResource(Macro):
    """Generate a resource

    Macro that generates a resource from given inputs

    Inputs:
     - fields: a map of inputs that are inserted into the expression

    Category: resource, macro
    """

    fields = Map(Input(types.Primitive()))

    def input_values(self):
        return [self.fields[name] for name, _ in self.get_inputs()]

    def get_inputs(self):
        return [(k, types.Primitive()) for k in self.fields.keys()]

    def duplicate(self, _origin=None, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            assert k in config["fields"]
            config["fields"][k] = v
        return self.__class__(_origin=_origin, **config)

    def validate(self, **inputs):
        super().validate(**inputs)
        return Resource(inputs)

    def expand(self, inputs, parent: "Reference"):
        
        with GraphBuilder(prefix=parent) as builder:

            for name, _ in self.fields.items():
                Copy(source=inputs[name], _name="." + name)

            return builder.nodes()
        

class AppendField(Macro):
    """Append a field to a resource

    Macro that generates a resource from an input resource and another field

    Inputs:
     - source: a map of inputs that are inserted into the expression

    Category: resource, macro
    """

    source = Input(Resource())
    name = String()
    value = Input(types.Primitive())

    def validate(self, **inputs):
        super().validate(**inputs)

        fields = dict(**inputs["source"].fields())
        if self.name in fields:
            raise ValidationException("Field already exists, cannot override")

        fields[self.name] = inputs["value"]

        return Resource(fields)

    def expand(self, inputs, parent: "Reference"):
        
        with GraphBuilder(prefix=parent) as builder:

            resource_type = inputs["source"].type

            for field, typ in resource_type.fields().items():
                Copy(source=resource_type.access(field, inputs["source"]), _name="." + field)
                
            Copy(source=inputs["value"], _name="." + self.name)

            return builder.nodes()
        