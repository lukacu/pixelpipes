
from attributee import String, Float
from pixelpipes.image.geometry import MaskBoundingBox

from .. import types
from ..graph import Constant, SeedInput, types, Input, Macro, ValidationException, Reference, Copy, GraphBuilder
from ..numbers import UniformDistribution, Add, Subtract, Divide
from ..complex import GetElement
from ..geometry.view import CenterView, FocusView, TranslateView, Chain
from ..geometry.points import ViewPoints
from ..geometry.rectangle import PointsBounds
from ..geometry.types import View, Points

from ..image import GetImageProperties
from ..image.geometry import ViewImage

from . import Resource

class ResourceView(Macro):
    
    node_name = "Resource View"
    node_description = "Apply a view transform to resource fields (where possible)"
    node_category = "resources"

    resource = Input(Resource())
    view = Input(View())
    width = Input(types.Integer())
    height = Input(types.Integer())

    def validate(self, **inputs):
        super().validate(**inputs)

        fields = {}

        for name in inputs["resource"].fields():
            typ = inputs["resource"].type(name)
            if isinstance(typ, types.Image):
                fields[name] = types.Image(inputs["width"].value, inputs["height"].value, typ.channels, typ.depth, typ.purpose)
            else:
                fields[name] = typ

        return Resource(fields=fields)

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_type = inputs["resource"].type

            width_2 = Divide(inputs["width"], 2)
            height_2 = Divide(inputs["height"], 2)

            offset = TranslateView(x=width_2, y=height_2)

            offset_view = Chain(inputs=[offset, inputs["view"]])

            for field in resource_type.fields():
                element_reference = resource_type.access(field, inputs["resource"])
                typ = resource_type.type(field)
                if isinstance(typ, types.Image):
                    interpolation = "Nearest" if typ.purpose == types.ImagePurpose.MASK else "Linear"
                    border = "ConstantLow" if typ.purpose == types.ImagePurpose.MASK else "Replicate"
                    ViewImage(source=element_reference, view=offset_view,
                        width=inputs["width"], height=inputs["height"], interpolation=interpolation, border=border, _name="." + field)
                elif Points().castable(typ):
                    ViewPoints(source=element_reference, view=offset_view, _name="." + field)
                else:
                    Copy(source=element_reference, _name="." + field)

            return builder.nodes()

class ResourceCenter(Macro):
    """Center view to mask or bounding box

    Returns a view that centers to a given mask or a bounding box

    Inputs:
      - resource
      - scale
      - field

    Category: resource, view, macro
    """

    node_name = "Resource Center"
    node_description = "Center resource to the information in its field"
    node_category = "resources"

    resource = Input(Resource())
    field = String(default="region")
    scale = Input(types.Float(), default=1)

    def validate(self, **inputs):
        super().validate(**inputs)

        if self.field not in inputs["resource"].fields():
            raise ValidationException("Field {} not present in resource".format(self.field), node=self)

        typ = inputs["resource"].type(self.field)

        if not Points().castable(typ) and not types.Image().castable(typ):
            raise ValidationException("Field {} not point set or image: {}".format(self.field, typ), node=self)

        return View()


    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_type = inputs["resource"].type

            field_reference = resource_type.access(self.field, inputs["resource"])
            typ = resource_type.type(self.field)

            if types.Image().castable(typ):
                bbox = MaskBoundingBox(source=field_reference)
            elif Points().castable(typ):
                bbox = PointsBounds(points=field_reference)

            center = CenterView(source=bbox)
            focus = FocusView(source=bbox, scale=inputs["scale"])

            Chain([focus, center], _name=parent)

            return builder.nodes()

class RandomPatchView(Macro):
    """Generates a random patch view for a given image

    Returns a view that centers to a given mask or a bounding box

    Inputs:
      - source
      - padding
      - width
      - height

    Category: resource, view, macro
    """

    source = Input(types.Image())
    width = Input(types.Integer())
    height = Input(types.Integer())
    padding = Input(types.Integer(), default=0)
    seed = SeedInput()

    def _output(self):
        return View()

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            properties = GetImageProperties(inputs["source"])
            image_width = GetElement(properties, "width")
            image_height = GetElement(properties, "height")

            x = UniformDistribution(- inputs["padding"], image_width - inputs["width"] + inputs["padding"], seed=inputs["seed"])
            y = UniformDistribution(- inputs["padding"], image_height - inputs["height"] + inputs["padding"], seed=inputs["seed"] * 2)

            TranslateView(x=-x, y=-y, _name=parent)

            return builder.nodes()
