import unittest

from pixelpipes.image import Invert, InvertImg, ToFloat, ToGray, ToRGB
from pixelpipes.image.augmentation import (
    AdditiveNoise,
    GaussNoise,
    ImageNoise,
    ImagePiecewiseAffine,
    PiecewiseAffine,
)
from pixelpipes.image.filter import GaussianBlur, GaussianFilter
from pixelpipes.graph import Constant, Graph
from pixelpipes.image.geometry import Crop, HorizontalFlip, ImageCrop, VerticalFlip
from pixelpipes.image.processing import (
    CoarseDropout,
    Erasing,
    ImageCoarseDropout,
    ImageCut,
    ImageDropout,
    ImageNormalize,
    ImageSolarize,
    Normalize,
    PixelDropout,
    Solarize,
)


class TestsImageAliases(unittest.TestCase):

    def test_operation_aliases_preserve_existing_implementations(self):
        self.assertIs(InvertImg, Invert)
        self.assertIs(AdditiveNoise, ImageNoise)
        self.assertIs(GaussNoise, ImageNoise)
        self.assertIs(PiecewiseAffine, ImagePiecewiseAffine)
        self.assertIs(GaussianBlur, GaussianFilter)
        self.assertIs(Crop, ImageCrop)
        self.assertIs(Normalize, ImageNormalize)
        self.assertIs(PixelDropout, ImageDropout)
        self.assertIs(CoarseDropout, ImageCoarseDropout)
        self.assertIs(Erasing, ImageCut)
        self.assertIs(Solarize, ImageSolarize)

    def test_defaulted_conversion_aliases(self):
        for operation, name, expected in (
                (ToFloat, "depth", "Float"),
                (ToGray, "convert", "RGB_GRAY"),
                (ToRGB, "convert", "GRAY_RGB")):
            attribute = operation.attributes()[name]
            self.assertEqual(attribute.dump(attribute.default), expected)

    def test_directional_flips_only_expose_source(self):
        with Graph():
            source = Constant(value=0)
            horizontal = HorizontalFlip(source=source, _auto=False)
            vertical = VerticalFlip(source=source, _auto=False)

        self.assertEqual(horizontal.input_names(), ["source"])
        self.assertEqual(vertical.input_names(), ["source"])
