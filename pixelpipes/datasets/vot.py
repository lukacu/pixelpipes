
from __future__ import absolute_import

from attributee import String, Boolean

try:
    from vot.dataset import VOTDataset as _VOTDataset
    from vot.region import Polygon, RegionType
except ImportError as ie:
    raise RuntimeError("VOT toolkit required") from ie

from pixelpipes.resources import SegmentedResourceListSource, SegmentedResourceList, Resource
import pixelpipes.engine as engine
import pixelpipes.types as types

def _extract_region(region, segmentation=False):
    poly = region.convert(RegionType.POLYGON)
    return poly._points

class VOTDataset(SegmentedResourceListSource):

    path = String()
    segmentation = Boolean(default=False)

    def _load(self):
        dataset = _VOTDataset(self.path)

        segments = []
        fields = {k : [] for k in dataset[dataset.list()[0]].channels()}
        groundtruth = []

        for sequence in dataset:
            segments.append(len(sequence))
            for name in sequence.channels():
                fields[name].extend([f.filename(name) for f in sequence])
            groundtruth.extend([_extract_region(f, self.segmentation) for f in sequence.groundtruth()])

        field_lists = {}
        field_types = {}

        for k, v in fields.items():
            field_lists[k] = engine.ImageFileList(v)
            if k == "color":
                field_types[k] = types.Image(channels=3, depth=8, purpose=types.ImagePurpose.VISUAL)
            elif k == "depth":
                field_types[k] = types.Image(channels=1, depth=8, purpose=types.ImagePurpose.VISUAL)
            elif k == "ir":
                field_types[k] = types.Image(channels=1, depth=8, purpose=types.ImagePurpose.VISUAL)
            else:
                raise RuntimeError("Unknown channel, unable to determine its parameters")
            
        if self.segmentation:
            field_lists["region"] = engine.ImageList(groundtruth)
            field_types["region"] = types.Image(channels=1, depth=8, purpose=types.ImagePurpose.MASK)
        else:
            field_lists["region"] = engine.PointsList(groundtruth)
            field_types["region"] = types.Points()

        return {"fields": field_lists, "size": len(groundtruth), "segments": segments, "types": field_types}

    def fields(self):
        data = self._get_data()
        return data["types"]
