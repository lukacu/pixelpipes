import unittest

import numpy as np

from pixelpipes.compiler import Compiler
from pixelpipes.graph import Constant, Graph, outputs
from pixelpipes.image.augmentation import (
    AutoContrast,
    Downscale,
    ImageCompression,
    MotionBlur,
    RandomToneCurve,
    Sharpen,
)


class TestsImageDegradation(unittest.TestCase):

    def test_all_wrappers_keep_shape_dtype_and_seed(self):
        image = np.random.default_rng(31).integers(1, 255, (3, 32, 32), dtype=np.uint8)

        with Graph() as graph:
            source = Constant(image)
            transforms = [
                Sharpen(source, alpha_range=(0.2, 0.2), seed=11),
                MotionBlur(source, blur_range=(5, 5), angle_range=(30, 30),
                           direction_range=(0, 0), allow_shifted=False, seed=11),
                Downscale(source, scale_range=(0.5, 0.5), seed=11),
                ImageCompression(source, quality_range=(70, 70), compression_type="jpeg", seed=11),
                RandomToneCurve(source, scale=0.1, per_channel=False, seed=11),
                AutoContrast(source, cutoff=0, method="cdf"),
            ]
            repeated = [
                Sharpen(source, alpha_range=(0.2, 0.2), seed=11),
                MotionBlur(source, blur_range=(5, 5), angle_range=(30, 30),
                           direction_range=(0, 0), allow_shifted=False, seed=11),
                Downscale(source, scale_range=(0.5, 0.5), seed=11),
                ImageCompression(source, quality_range=(70, 70), compression_type="jpeg", seed=11),
                RandomToneCurve(source, scale=0.1, per_channel=False, seed=11),
                AutoContrast(source, cutoff=0, method="cdf"),
            ]
            outputs(*(transforms + repeated))

        result = Compiler().build(graph).run(1)

        for value in result:
            self.assertEqual(value.shape, image.shape)
            self.assertEqual(value.dtype, image.dtype)
            self.assertTrue(np.isfinite(value).all())

        for first, second in zip(result[:6], result[6:]):
            np.testing.assert_array_equal(first, second)

    def test_identity_parameters(self):
        image = np.random.default_rng(37).integers(1, 255, (3, 12, 12), dtype=np.uint8)

        with Graph() as graph:
            source = Constant(image)
            outputs(
                Sharpen(source, alpha_range=(0, 0), method="kernel", seed=1),
                Downscale(source, scale_range=(1, 1), seed=1),
                RandomToneCurve(source, scale=0, seed=1),
            )

        result = Compiler().build(graph).run(1)
        np.testing.assert_array_equal(result[0], image)
        np.testing.assert_array_equal(result[1], image)
        np.testing.assert_array_equal(result[2], image)

        constant = np.full((3, 12, 12), 77, dtype=np.uint8)
        with Graph() as graph:
            outputs(AutoContrast(Constant(constant), cutoff=0, method="pil"))
        np.testing.assert_array_equal(Compiler().build(graph).run(1)[0], constant)

