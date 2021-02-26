
import os
import typing
from collections import Mapping

from attributee import String
from attributee.containers import Map
from attributee.object import class_fullname

from pixelpipes import Macro, Input, Copy, GraphBuilder, hidden, Constant
from pixelpipes.types import List, Image, Complex, Integer, TypeException
import pixelpipes.nodes as nodes
import pixelpipes.types as types

import pixelpipes.engine as engine

_RESOURCE_CACHE = dict()

def make_hash(o):
    """
    makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
    """

    def freeze(o):
        if isinstance(o, (tuple, list)):
            return tuple((freeze(e) for e in o))

        if isinstance(o, Mapping):
            return tuple(sorted((k, freeze(v)) for k, v in o.items()))

        if isinstance(o, (set, frozenset)):
            return tuple(sorted(freeze(e) for e in o))
            
        return o

    return hash(freeze(o)) 

class VirtualField(object):

    def __init__(self, typ: types.Type):
        self._typ = typ

    @property
    def type(self) -> types.Type:
        return self._typ

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_id = (class_fullname(self), make_hash(self.dump()))

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
                nodes.ListSource(data["lists"][field], element_type=typ, _name="." + field)
            return builder.nodes()

    def _get_data(self):
        if not self._cache_id in _RESOURCE_CACHE:
            _RESOURCE_CACHE[self._cache_id] = self._load()
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

            nodes.ListSource(engine.IntegerList(beginnings), _name="._begin")
            nodes.ListSource(engine.IntegerList(endings), _name="._end")
            nodes.ListSource(engine.IntegerList(segments), _name="._length")

            graph.update(builder.nodes())
            return graph

    def fields(self):
        raise NotImplementedError()

    def _load(self):
        raise NotImplementedError()

class ImageDirectory(ResourceListSource):

    EXTENSIONS = [".jpg", ".jpeg", ".png"]

    path = String()

    def _load(self):
        files = [fi for fi in os.listdir(self.path) if os.path.splitext(fi)[1].lower() in ImageDirectory.EXTENSIONS]
        path = self.path if self.path.endswith(os.sep) else (self.path + os.sep)
        return {"lists": {"image": engine.ImageFileList(files, path)}, "size": len(files)}

    def fields(self):
        return dict(image=Image())

class GetResourceListLength(Macro):

    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)
        return types.Integer(value=inputs["resources"].size)

    def expand(self, inputs, parent: "Reference"):

        resources_reference, resources_type = inputs["resources"]

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                Constant(value=resources_type.size, _name=parent)
            else:
                real_fields = [n for n in resources_type.fields() if not resources_type.virtual(n)]
                field = real_fields[0]
                length = nodes.ListLength(parent=resources_type.access(field, resources_reference))
                nodes.Add(a=length, b=int(-1), _name=parent)

            return builder.nodes()

class GetResource(Macro):
    
    resources = Input(ResourceList())
    index = Input(Integer())

    def validate(self, **inputs):
        super().validate(**inputs)
        
        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resources_reference, resources_type = inputs["resources"]

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    nodes.ListElement(parent=resources_type.access(field, resources_reference), index=self.index, _name="." + field)

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

            resource_reference, resource_type = inputs["resource"]
            length_reference, _ = inputs["length"]

            for field, _ in resource_type.fields().items():
                field_source = resource_type.access(field, resource_reference)
                nodes.RepeatElement(source=field_source, length=length_reference, _name="." + field)

            return builder.nodes()

class GetLastResource(Macro):
    
    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        resources_reference, resources_type = inputs["resources"]

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                last = resources_type.size - 1
            else:
                real_fields = [n for n in resources_type.fields() if not resources_type.virtual(n)]
                field = real_fields[0]
                length = nodes.ListLength(parent=resources_type.access(field, resources_reference))
                last = nodes.Add(a=length, b=int(-1))

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    nodes.ListElement(parent=resources_type.access(field, resources_reference), index=last, _name="." + field)

            return builder.nodes()

class GetRandomResource(Macro):
    """Random resource

    Select a random resource from an input list of resources

    Inputs:
     - resources: Resource list that is sampled for elements

    Category: resources, sampling
    """

    resources = Input(ResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)

        return inputs["resources"].element()

    def expand(self, inputs, parent: "Reference"):

        resources_reference, resources_type = inputs["resources"]

        with GraphBuilder(prefix=parent) as builder:

            if resources_type.size is not None:
                generator = nodes.UniformDistribution(min=0, max=resources_type.size-1)
            else:
                real_fields = [n for n in resources_type.fields() if not resources_type.virtual(n)]
                field = real_fields[0]
                length = nodes.ListLength(parent=resources_type.access(field, resources_reference))
                last = nodes.Add(a=length, b=int(-1))
                generator = nodes.UniformDistribution(min=0, max=last)

            index = nodes.Round(source=generator)

            resource_type = resources_type.element()

            for field, typ in resources_type.fields().items():
                if resources_type.virtual(field):
                    Copy(source=typ.generate(parent, resource_type), _name="." + field)
                else:
                    nodes.ListElement(parent=resources_type.access(field, resources_reference), index=index, _name="." + field)

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

            resource_reference, resource_type = inputs["resource"]
            Copy(source=resource_type.access(self.field, resource_reference), _name=parent)

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

        resources_reference, resources_type = inputs["resources"]
        begin_reference, _ = inputs["begin"]
        end_reference, _ = inputs["end"]

        with GraphBuilder(prefix=parent) as builder:

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                nodes.SublistSelect(parent=resources_type.access(field, resources_reference), \
                    begin=begin_reference, end=end_reference, _name="." + field)

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

        resources_reference, resources_type = inputs["resources"]

        with GraphBuilder(prefix=parent) as builder:

            begin_reference = nodes.ListElement(parent=resources_type.access("_begin", resources_reference), index=self.index)
            end_reference = nodes.ListElement(parent=resources_type.access("_end", resources_reference), index=self.index)

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                nodes.SublistSelect(parent=resources_type.access(field, resources_reference), begin=begin_reference, end=end_reference, _name="." + field)

            return builder.nodes()

class GetRandomResourceSegment(Macro):
    
    resources = Input(SegmentedResourceList())

    def validate(self, **inputs):
        super().validate(**inputs)

        return ResourceList(fields=inputs["resources"].fields(), size=None)

    def expand(self, inputs, parent: "Reference"):

        resources_reference, resources_type = inputs["resources"]

        with GraphBuilder(prefix=parent) as builder:


            if resources_type.total() is not None:
                generator = nodes.UniformDistribution(min=0, max=resources_type.total()-1)
            else:
                field = next(resources_type.fields())
                total = nodes.ListLength(parent=resources_type.access("_begin", resources_reference))
                last = nodes.Add(a=total, b=int(-1))
                generator = nodes.UniformDistribution(min=0, max=last)

            index = nodes.Round(source=generator)

            begin_reference = nodes.ListElement(parent=resources_type.access("_begin", resources_reference), index=index)
            end_reference = nodes.ListElement(parent=resources_type.access("_end", resources_reference), index=index)

            for field in resources_type.fields():
                if resources_type.virtual(field):
                    continue
                nodes.SublistSelect(parent=resources_type.access(field, resources_reference), begin=begin_reference, end=end_reference, _name="." + field)
            
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

    def duplicate(self, **inputs):
        config = self.dump()
        for k, v in inputs.items():
            assert k in config["fields"]
            config["fields"][k] = v
        return self.__class__(**config)

    def validate(self, **inputs):
        super().validate(**inputs)

        return Resource(inputs)

    def expand(self, inputs, parent: "Reference"):
        
        with GraphBuilder(prefix=parent) as builder:

            for name, _ in self.fields.items():
                Copy(source=inputs[name][0], _name="." + name)

            return builder.nodes()
        