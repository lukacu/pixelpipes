import unittest

import numpy as np

from pixelpipes.compiler import Compiler
from pixelpipes.graph import Constant, Graph, outputs
from pixelpipes.image.processing import (
    AdditiveNoise,
    CLAHE,
    ChannelDropout,
    ChannelShuffle,
    ColorJitter,
    HueSaturationValue,
    MultiplicativeNoise,
    RGBShift,
    RandomBrightnessContrast,
    RandomGamma,
    SaltAndPepper,
    ShotNoise,
)


class TestsImageColorNoise(unittest.TestCase):

    def test_all_wrappers_keep_shape_dtype_and_seed(self):
        image = np.random.default_rng(17).integers(1, 255, (3, 24, 24), dtype=np.uint8)

        with Graph() as graph:
            source = Constant(image)
            transforms = [
                RandomBrightnessContrast(source, seed=11),
                ColorJitter(source, seed=11),
                HueSaturationValue(source, seed=11),
                RGBShift(source, seed=11),
                RandomGamma(source, seed=11),
                CLAHE(source, seed=11),
                ChannelDropout(source, seed=11),
                ChannelShuffle(source, seed=11),
                AdditiveNoise(source, seed=11),
                MultiplicativeNoise(source, seed=11),
                ShotNoise(source, seed=11),
                SaltAndPepper(source, seed=11),
            ]
            outputs(*(transforms + [
                type(transform)(source, seed=11)
                for transform in transforms
            ]))

        result = Compiler().build(graph).run(1)

        for value in result:
            self.assertEqual(value.shape, image.shape)
            self.assertEqual(value.dtype, image.dtype)
            self.assertTrue(np.isfinite(value).all())

        for first, second in zip(result[:12], result[12:]):
            np.testing.assert_array_equal(first, second)

    def test_fixed_parameters_include_identity_cases(self):
        image = np.random.default_rng(23).integers(1, 255, (3, 12, 12), dtype=np.uint8)

        with Graph() as graph:
            source = Constant(image)
            outputs(
                RandomBrightnessContrast(source, brightness_range=(0, 0), contrast_range=(0, 0), seed=1),
                AdditiveNoise(source, noise_range=(0, 0), seed=1),
                MultiplicativeNoise(source, multiplier=(1, 1), seed=1),
                SaltAndPepper(source, amount_range=(0, 0), seed=1),
            )

        result = Compiler().build(graph).run(1)
        for value in result:
            np.testing.assert_array_equal(value, image)

