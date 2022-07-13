
import os
import typing
import json
import hashlib
import numpy as np

from attributee.object import class_fullname

from ..list import GetElement, Permutation, Remap, Repeat, SublistSelect
from ..numbers import Round, SampleUnform
from ..types import Data, Integer, TypeException
from ..graph import Constant, Macro, Input, Node, NodeOperation, Copy, SeedInput, ValidationException, hidden
from . import Resource, ResourceField, ResourceProxy, TokenField, real_field
from ..utilities import PersistentDict
from ..list import FileList as FileListConstant

if "PIXELPIPES_CACHE" in os.environ:
    _RESOURCE_CACHE = PersistentDict(os.environ.get("PIXELPIPES_CACHE"))
else:
    _RESOURCE_CACHE = dict()

class FileList(list):
    pass

def make_hash(o):
    """
    makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
    """

    sha1 = hashlib.sha1()

    sha1.update(json.dumps(o, sort_keys=True).encode("utf-8"))

    return sha1.hexdigest()

def is_resource_list(typ: Data):
    return isinstance(typ, Resource) and "__list_length" in typ

def is_segmented_resource_list(typ: Data):
    return is_resource_list(typ) and "__list_seg_length" in typ

def ResourceList(**fields: typing.Mapping[str, ResourceField]):

    if "__list_length" in fields:
        raise TypeError("Illegal field name")

    fields["__list_length"] = TokenField("__list_length")

    return Resource(**fields)

def SegmentedResourceList(**fields: typing.Mapping[str, ResourceField]):

    if "__list_length" in fields:
        raise TypeError("Illegal field name")

    fields["__list_length"] = TokenField("__list_length")
    fields["__list_seg_begin"] = TokenField("__list_seg_begin")
    fields["__list_seg_end"] = TokenField("__list_seg_end")

    return Resource(**fields)

@hidden
class ResourceListSource(Macro):
    """Resource list source is an abstract macro that makes it easier to write resource list dataset providers. Implementations
    must implement `load` function that provides the data that is injected into a graph.
    """

    def _init(self):
        self._cache_id = class_fullname(self) + "@" + make_hash(self.dump())

    def _generate(self, data):

        fields = {}
        forward = {}

        lenghts = []

        for name, field in data.items():
            if isinstance(field, ResourceField):
                fields[name] = field
            elif isinstance(field, FileList):
                forward[name] = FileListConstant(field, _name="." + name)
                lenghts.append(len(field))
            elif isinstance(field, list):
                forward[name] = Constant(field, _name="." + name)
                lenghts.append(len(field))
            elif isinstance(field, np.ndarray):
                forward[name] = Constant(field, _name="." + name)
                lenghts.append(field.shape[0])
            else:
                raise ValidationException("Not a supported resource list field: {!s:.100}".format(field))
    
        if len(lenghts) == 0:
            raise ValidationException("Empty resource list, no fields defined")

        if not all([lenghts[0] == x for x in lenghts]):
            raise ValidationException("Field size mismatch for resource list")

        forward["__list_length"] = Constant(lenghts[0], _name=".__list_length")

        return fields, forward

    def expand(self):
        data = self._get_data()
        fields, forward = self._generate(data)
        return ResourceProxy(**forward, _fields = fields)

    def _get_data(self):
        if not self._cache_id in _RESOURCE_CACHE:
            data = self.load()
            _RESOURCE_CACHE[self._cache_id] = data
            return data
        return _RESOURCE_CACHE[self._cache_id]

    def load(self):
        raise NotImplementedError()

@hidden
class SegmentedResourceListSource(ResourceListSource):

    def _generate(self, data):

        if "_segments" in data:
            segments = data["_segments"]
            del data["_segments"]

            beginnings = [0]
            endings = [segments[0]-1]
            for l in segments[1:]:
                beginnings.append(endings[-1]+1)
                endings.append(endings[-1]+l)
        else:
            raise TypeException("Must include segment information")

        fields, forward = super()._generate(data)

        forward["__list_seg_begin"] = Constant(beginnings, _name=".__list_seg_begin")
        forward["__list_seg_end"] = Constant(endings, _name=".__list_seg_end")
        forward["__list_seg_length"] = Constant(segments, _name=".__list_seg_length")

        return fields, forward

    def load(self):
        raise NotImplementedError()


class GetResourceListLength(Macro):

    resources = Input(ResourceList())

    def expand(self, resources):
        resources_type = resources.type
        return Copy(resources_type["__list_length"].access(resources))

Node.register_operation(NodeOperation.LENGTH, GetResourceListLength, ResourceList())

class GetResource(Macro):
    
    resources = Input(ResourceList())
    index = Input(Integer())

    def expand(self, resources, index):

        resources_type = resources.type

        forward = {}
        fields = {}
        
        # Only copy real fields
        for name, field in resources_type:
            # Strip list meta fields
            if name.startswith("__list_"):
                continue

            if real_field(field):
                forward[name] = GetElement(resources_type[name].access(resources), index, _name="." + name)
            else:
                fields[name] = field

        return ResourceProxy(_fields=fields, **forward)

Node.register_operation(NodeOperation.INDEX, GetResource, ResourceList(), Integer())

class RepeatResource(Macro):
    """Returns a list of resources where an input resource is repeated a number of times"""

    resource = Input(Resource(), description="Resource to repeat")
    length = Input(Integer(), description="Number of repetitions")

    def expand(self, resource, length):

        resource_type = resource.type

        if is_resource_list(resource_type):
            raise ValidationException("Element must not be a resource list")

        forward = {}
        fields = {}

        for name, field in resource_type:
            if not real_field(field):
                fields[name] = field
            else:
                field_source = resource_type[name].access(resource)
                forward[name] = Repeat(field_source, length, _name="." + name)

        forward["__list_length"] = Constant(self.length)

        return ResourceProxy(_fields=fields, **forward)

class GetLastResource(Macro):
    
    resources = Input(ResourceList())

    def expand(self, resources):
        length = GetResourceListLength(resources)
        return GetResource(resources, length - 1)

class RandomResource(Macro):
    """Select a random resource from an input list of resources
    """

    resources = Input(ResourceList())
    seed = SeedInput()

    def expand(self, resources, seed):

        length = GetResourceListLength(resources)
        generator = SampleUnform(0, length-1, seed=self.seed)

        index = Round(generator)
        
        return GetResource(resources, index)

class PermuteResources(Macro):
    """ Randomly permutes the resource list
    """

    resources = Input(ResourceList())

    def expand(self, resources):

        resources_type = resources.type

        length = GetResourceListLength(resources)
        indices = Permutation(min=0, max=length-1)

        forward = {}
        fields = {}

        for name, field in resources_type:
            if name.startswith("__list_"):
                continue

            if not real_field(field):
                fields[name] = field
            else:
                forward[name] = Remap(field.access(resources), indices, _name="." + name)

        forward["__list_length"] = Copy(length,  _name=".__list_length")

        return ResourceProxy(_fields=fields, **forward)

class ListInterval(Macro):
    
    resources = Input(ResourceList())
    begin = Input(Integer())
    end = Input(Integer())

    def expand(self, resources, begin, end):

        resources_type = resources.type
        fields = {}
        forward = {}

        for name, field in resources_type:
            if name.startswith("__list_"):
                continue

            if not real_field(field):
                fields[name] = field
            else:
                forward[name] = SublistSelect(parent=field.access(resources), \
                    begin=begin, end=end, _name="." + name)

        forward["__list_length"] = Copy(end - begin,  _name=".__list_length")

        return ResourceProxy(_fields=fields, **forward)

class SegmentCount(Macro):

    resources = Input(SegmentedResourceList())

    def expand(self, resources):
        return len(resources.type["__list_seg_begin"].access(resources))

class ResourceSegment(Macro):
    
    resources = Input(SegmentedResourceList())
    index = Input(Integer())

    def expand(self, resources, index):

        resources_type = resources.type

        begin = GetElement(resources_type["__list_seg_begin"].access(resources), index)
        end = GetElement(resources_type["__list_seg_eng"].access(resources), index)

        return ListInterval(resources, begin, end)

class RandomResourceSegment(Macro):
    
    resources = Input(SegmentedResourceList())
    seed = SeedInput()

    def expand(self, resources, seed):

        index = Round(SampleUnform(0, SegmentCount(resources) - 1, seed=seed))
        return ResourceSegment(resources, index)

class PermuteResourceSegments(Macro):
    
    resources = Input(SegmentedResourceList())
    seed = SeedInput()

    def expand(self, resources, seed):

        resources_type = resources.type

        fields = {}
        forward = {}

        indices = Permutation(SegmentCount(resources_type) - 1, seed)

        for name, field in resources_type:
            if name.startswith("__list_seg"):
                continue

            if not real_field(field):
                fields[name] = field
            else:
                forward[name] = Remap(field.access(resources), indices, _name="." + name)

        forward["__list_seg_begin"] = Remap(resources_type["__list_seg_begin"].access(resources), indices, _name=".__list_seg_begin")
        forward["__list_seg_end"] = Remap(resources_type["__list_seg_end"].access(resources), indices, _name=".__list_seg_end")

        return ResourceProxy(_fields=fields, **forward)
