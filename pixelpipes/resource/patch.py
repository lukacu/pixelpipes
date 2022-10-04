
from attributee import String
from pixelpipes.image.geometry import MaskBoundingBox

from .. import types
from ..graph import types, Input, Macro, Copy
from ..geometry.view import CenterView, FocusView, TranslateView, Chain
from ..geometry.points import ViewPoints
from ..geometry.rectangle import PointsBounds

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
            elif types.Points().castable(typ):
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

        field_reference = resource_type.access(self.field, resource)
        typ = resource_type.type(self.field)

        if types.Image().castable(typ):
            bbox = MaskBoundingBox(source=field_reference)
        elif types.Points().castable(typ):
            bbox = PointsBounds(points=field_reference)

        center = CenterView(bbox)
        focus = FocusView(source=bbox, scale=scale)

        return Chain([focus, center])


