
from attributee import String, Float

from pixelpipes import Copy, Input, Macro, GraphBuilder
import pixelpipes.nodes as nodes
import pixelpipes.types as types

from pixelpipes.resources import Resource, VirtualField

class ResourceView(Macro):
    
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
                    interpolate = not typ.purpose == types.ImagePurpose.MASK
                    border = "CONSTANT_LOW" if typ.purpose == types.ImagePurpose.MASK else "REPLICATE"
                    nodes.ViewImage(source=element_reference, view=offset_view,
                        width=width_reference, height=height_reference, interpolate=interpolate, border=border, _name="." + field)
                elif isinstance(typ, types.Points):
                    nodes.ViewPoints(source=element_reference, view=offset_view, _name="." + field)
                else:
                    Copy(source=element_reference, _name="." + field)

            return builder.nodes()

class ResourceCenter(Macro):

    resource = Input(Resource())
    scale = Input(types.Float(), default=1)
    field = String(default="region")

    def validate(self, **inputs):
        super().validate(**inputs)

        if self.field not in inputs["resource"].fields():
            raise types.TypeException("Field {} not present in resource".format(self.field))

        typ = inputs["resource"].type(self.field)

        if not isinstance(typ, types.Points) and not isinstance(typ, types.Image):
            raise types.TypeException("Field {} not point set or image".format(self.field))

        return types.View()


    def expand(self, inputs, parent: "Reference"):

        with GraphBuilder(prefix=parent) as builder:

            resource_reference, resource_type = inputs["resource"]
            scale_reference, _ = inputs["scale"]

            field_reference = resource_type.access(self.field, resource_reference)
            typ = resource_type.type(self.field)

            if isinstance(typ, types.Image):
                bbox = nodes.MaskBoundingBox(source=field_reference)
            elif isinstance(typ, types.Points):
                bbox = nodes.BoundingBox(points=field_reference)

            center = nodes.CenterView(source=bbox)
            focus = nodes.FocusView(source=bbox, scale=scale_reference)

            nodes.Chain(inputs=[focus, center], _name=parent)

            return builder.nodes()