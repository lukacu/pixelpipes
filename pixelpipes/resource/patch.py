
from attributee import String
from pixelpipes.image.geometry import MaskBoundingBox

from .. import types
from ..graph import SeedInput, types, Input, Macro, Copy
from ..numbers import SampleUnform
from ..geometry.view import CenterView, FocusView, TranslateView, Chain
from ..geometry.points import ViewPoints
from ..geometry.rectangle import PointsBounds

from ..image import GetImageProperties
from ..image.geometry import ViewImage

from . import Resource

class ResourceView(Macro):
    """Apply a view transform to resource fields (where possible, i.e. to images and points)"""

    resource = Input(Resource())
    view = Input(types.View())
    width = Input(types.Integer())
    height = Input(types.Integer())

    def expand(self, resource, view, width, height):
 
        resource_type = resource.type

        width_2 = width / 2
        height_2 = height / 2

        offset = TranslateView(x=width_2, y=height_2)

        offset_view = Chain(inputs=[offset, view])

        for field in resource_type.fields():
            element_reference = resource_type.access(field, resource)
            typ = resource_type.type(field)
            if isinstance(typ, types.Image):
                interpolation = "Nearest" if typ.purpose == types.ImagePurpose.MASK else "Linear"
                border = "ConstantLow" if typ.purpose == types.ImagePurpose.MASK else "Replicate"
                ViewImage(source=element_reference, view=offset_view,
                    width=width, height=height, interpolation=interpolation, border=border, _name="." + field)
            elif Points().castable(typ):
                ViewPoints(element_reference, offset_view, _name="." + field)
            else:
                Copy(source=element_reference, _name="." + field)

  
class ResourceCenter(Macro):
    """Center resource to the information in its field"""

    resource = Input(Resource())
    field = String(default="region")
    scale = Input(types.Float(), default=1)

    def expand(self, resource, field, scale):

        resource_type = resource.type

        field_reference = resource_type.access(self.field, inputs["resource"])
        typ = resource_type.type(self.field)

        if types.Image().castable(typ):
            bbox = MaskBoundingBox(source=field_reference)
        elif Points().castable(typ):
            bbox = PointsBounds(points=field_reference)

        center = CenterView(bbox)
        focus = FocusView(source=bbox, scale=scale)

        return Chain([focus, center])

class RandomPatchView(Macro):
    """Returns a view that focuses on a random patch in an image"""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Width of a patch")
    height = Input(types.Integer(), description="Height of a patch")
    padding = Input(types.Integer(), default=0, description="Padding for sampling")
    seed = SeedInput()

    def expand(self, source, width, height, padding, seed):

        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        x = SampleUnform(- padding, image_width - width + padding, seed=seed)
        y = SampleUnform(- padding, image_height - height + padding, seed=seed * 2)

        return TranslateView(x=-x, y=-y)

