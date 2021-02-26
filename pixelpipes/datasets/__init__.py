
from attributee import String, Float

from pixelpipes import Copy, Input, Macro, GraphBuilder, ValidationException
import pixelpipes.nodes as nodes
import pixelpipes.types as types
from pixelpipes.nodes.numeric import UniformDistribution, Add, Subtract, Divide
from pixelpipes.nodes.image import GetImageProperties
from pixelpipes.nodes.complex import GetElement

from pixelpipes.nodes.resources import Resource, VirtualField

class ResourceView(Macro):
    
    node_name = "Resource View"
    node_description = "Apply a view transform to resource fields (where possible)"
    node_category = "resources"

    resource = Input(Resource())
    view = Input(types.View())
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

            resource_reference, resource_type = inputs["resource"]
            view_reference, _ = inputs["view"]
            width_reference, _ = inputs["width"]
            height_reference, _ = inputs["height"]

            width_2 = nodes.Divide(a=width_reference, b=2)
            height_2 = nodes.Divide(a=height_reference, b=2)

            offset = nodes.TranslateView(x=width_2, y=height_2)

            offset_view = nodes.Chain(inputs=[offset, view_reference])

            for field in resource_type.fields():
                element_reference = resource_type.access(field, resource_reference)
                typ = resource_type.type(field)
                if isinstance(typ, types.Image):
                    interpolation = "NEAREST" if typ.purpose == types.ImagePurpose.MASK else "LINEAR"
                    border = "CONSTANT_LOW" if typ.purpose == types.ImagePurpose.MASK else "REPLICATE"
                    nodes.ViewImage(source=element_reference, view=offset_view,
                        width=width_reference, height=height_reference, interpolation=interpolation, border=border, _name="." + field)
                elif types.Points().castable(typ):
                    nodes.ViewPoints(source=element_reference, view=offset_view, _name="." + field)
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
    scale = Input(types.Float(), default=1)
    field = String(default="region")

    def validate(self, **inputs):
        super().validate(**inputs)

        if self.field not in inputs["resource"].fields():
            raise ValidationException("Field {} not present in resource".format(self.field), node=self)

        typ = inputs["resource"].type(self.field)

        if not types.Points().castable(typ) and not types.Image().castable(typ):
            raise ValidationException("Field {} not point set or image: {}".format(self.field, typ), node=self)

        return types.View()


    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_reference, resource_type = inputs["resource"]
            scale_reference, _ = inputs["scale"]

            field_reference = resource_type.access(self.field, resource_reference)
            typ = resource_type.type(self.field)

            if types.Image().castable(typ):
                bbox = nodes.MaskBoundingBox(source=field_reference)
            elif types.Points().castable(typ):
                bbox = nodes.BoundingBox(points=field_reference)

            center = nodes.CenterView(source=bbox)
            focus = nodes.FocusView(source=bbox, scale=scale_reference)

            nodes.Chain(inputs=[focus, center], _name=parent)

            return builder.nodes()



class RandomPatch(Macro):
    """Generates a random patch view for a given image

    Returns a view that centers to a given mask or a bounding box

    Inputs:
      - source
      - scale
      - field

    Category: resource, view, macro
    """

    source = Input(types.Image())
    padding = Input(types.Integer())
    width = Input(types.Integer())
    height = Input(types.Integer())

    def _output(self):
        return types.View()

    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            image, _ = inputs["source"]
            padding, _ = inputs["padding"]
            width, _ = inputs["width"]
            height, _ = inputs["height"]

            properties = GetImageProperties(source=image)
            image_width = GetElement(source=properties, element="width")
            image_height = GetElement(source=properties, element="height")

            offsetx = Add(padding, Divide(width, 2))
            offsety = Add(padding, Divide(height, 2))

            x = UniformDistribution(min=offsetx, max=Subtract(image_width, offsetx))
            y = UniformDistribution(min=offsety, max=Subtract(image_height, offsety))

            nodes.TranslateView(x=-x, y=-y, _name=parent)

            return builder.nodes()
