import unittest

import numpy as np

from pixelpipes.compiler import Compiler
from pixelpipes.graph import Constant, Graph, outputs
from pixelpipes.image.geometry import (
    CenterCrop,
    LetterBox,
    LongestMaxSize,
    Pad,
    PadIfNeeded,
    RandomCropFromBorders,
    RandomResizedCrop,
    RandomSizedCrop,
    SmallestMaxSize,
)


class TestsImageGeometryWrappers(unittest.TestCase):

    def test_crop_resize_wrapper_batch(self):
        image = np.arange(1, 81, dtype=np.uint8).reshape(8, 10)

        with Graph() as graph:
            source = Constant(image)
            outputs(
                CenterCrop(source, width=6, height=4),
                RandomResizedCrop(source, width=6, height=4, seed=1),
                RandomCropFromBorders(source, crop_left=0.2, crop_right=0.2,
                                      crop_top=0.2, crop_bottom=0.2, seed=2),
                Pad(source, padding=2),
                PadIfNeeded(source, min_width=14, min_height=12),
                LetterBox(source, width=12, height=12),
                LongestMaxSize(source, max_size=6),
                SmallestMaxSize(source, max_size=6),
                RandomSizedCrop(source, min_height=3, max_height=6,
                                width=6, height=4, w2h_ratio=1.0, seed=3),
            )

        result = Compiler().build(graph).run(1)

        np.testing.assert_array_equal(result[0], image[2:6, 2:8])
        self.assertEqual(result[1].shape, (1, 4, 6))
        self.assertGreater(result[2].shape[1], 0)
        self.assertGreater(result[2].shape[2], 0)
        self.assertLessEqual(result[2].shape[1], image.shape[0])
        self.assertLessEqual(result[2].shape[2], image.shape[1])
        self.assertEqual(result[3].shape, (1, 12, 14))
        self.assertEqual(result[4].shape, (1, 12, 14))
        self.assertEqual(result[5].shape, (1, 12, 12))
        self.assertEqual(max(result[6].shape[1:]), 6)
        self.assertEqual(min(result[7].shape[1:]), 6)
        self.assertEqual(result[8].shape, (1, 4, 6))

