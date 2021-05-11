
import unittest
import numpy as np


from ..core import Output, Constant
from ..graph import GraphBuilder
from ..compiler import Compiler
from ..core.complex import GetElement
from . import ConstantImage, Channel, Equals, Grayscale, Invert, Merge, Threshold, Moments, GetImageProperties, ConvertDepth
from .arithemtic import ImageAdd, ImageMultiply, ImageSubtract
from .processing import ImageBlend, ImageCoarseDropout, ImageCut, ImageDropout, ImageSolarize
from .geometry import Flip, MaskBoundingBox, Resize, Rotate, Scale, ImageCrop
from .augmentation import ImageBrightness, ImageNoise, ImagePiecewiseAffine
from .filter import AverageBlur, BilateralFilter, GaussianBlur, ImageFilter, MedianBlur
from .render import LinearImage, NormalNoise, UniformNoise
from ..geometry.rectangle import MakeRectangle

class TestPipes(unittest.TestCase):    

    test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)
    test_image_rgb = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)
    test_kernel = np.ones((5,5), dtype=np.uint8)

    def test_constant_image(self):

        with GraphBuilder() as graph:
            n1 = ConstantImage(source=self.test_image)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], self.test_image)

    def test_aug_noise(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(self.test_image)
            n1 = ImageNoise(test_image, 20)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_aug_brightness(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(self.test_image)
            n1 = ImageBrightness(test_image, 20)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_aug_piecewiselinear(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(self.test_image)
            n1 = ImagePiecewiseAffine(test_image, 1.5, 4)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertEqual(sample[0].shape, self.test_image.shape)

    """
    nodes/image.py
    """

    def test_image_get_image_properties(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(self.test_image)
            n1 = GetImageProperties(test_image)
            n2 = GetElement(n1, element="width")
            n3 = GetElement(n1, element="height")
            n4 = GetElement(n1, element="channels")
            n5 = GetElement(n1, element="depth")
            Output(outputs=[n2,n3,n4,n5])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 256)        
        self.assertEqual(sample[1], 256)
        self.assertEqual(sample[2], 1)
        self.assertEqual(sample[3], 8)

    def test_image_convert_depth(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(self.test_image)
            n1 = ConvertDepth(source=test_image, depth="Byte")
            n2 = ConvertDepth(source=test_image, depth="Short")
            n3 = ConvertDepth(source=test_image, depth="Float")
            n4 = ConvertDepth(source=test_image, depth="Double")
            Output(outputs=[n1, n2, n3, n4])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIs(sample[0].dtype, np.dtype('ubyte'))
        self.assertIs(sample[1].dtype, np.dtype('short'))
        self.assertIs(sample[2].dtype, np.dtype('float32'))
        self.assertIs(sample[3].dtype, np.dtype('float64'))

    def test_image_grayscale(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image_rgb)
            n1 = Grayscale(source=test_image)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_threshold(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Threshold(source=test_image, threshold=100)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_invert(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Invert(source=test_image)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_equals(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Equals(source=test_image, value=128)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_channel(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Channel(source=test_image, index=0)
            Output(outputs=[test_image, n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_image_merge(self):

        with GraphBuilder() as graph:
            test_image_1 = ConstantImage(source=self.test_image)
            test_image_2 = ConstantImage(source=self.test_image)
            test_image_3 = ConstantImage(source=self.test_image)
            n1 = Merge(source1=test_image_1, source2=test_image_2, source3=test_image_3)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_moments(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=[[1, 0, 1], [1, 0, 1]])
            n1 = Moments(source=test_image)
            Output(outputs=[n1])
            
        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], np.array([[4], [2], [4], [2]]))

    def test_image_add(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            number = Constant(value=0)
            n1 = ImageAdd(source1=test_image, source2=test_image)
            n2 = ImageAdd(source1=test_image, source2=number)
            n3 = ImageAdd(source1=number, source2=test_image)
            Output(outputs=[test_image, n1, n2, n3])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[1], np.ndarray)
        np.testing.assert_array_equal(sample[0], sample[2])
        np.testing.assert_array_equal(sample[0], sample[3])

    def test_image_subtract(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            number = Constant(value=0)
            n1 = ImageSubtract(source1=test_image, source2=test_image)
            n2 = ImageSubtract(source1=test_image, source2=number)
            n3 = ImageSubtract(source1=number, source2=test_image)
            Output(outputs=[test_image, n1, n2, n3])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[1], np.ndarray)
        np.testing.assert_array_equal(sample[0], sample[2])
        np.testing.assert_array_equal(sample[0], sample[3])

    def test_image_multiply(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            number = Constant(value=1)
            n1 = ImageMultiply(source1=test_image, source2=test_image)
            n2 = ImageMultiply(source1=test_image, source2=number)
            n3 = ImageMultiply(source1=number, source2=test_image)
            Output(outputs=[test_image, n1, n2, n3])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[1], np.ndarray)
        np.testing.assert_array_equal(sample[0], sample[2])
        np.testing.assert_array_equal(sample[0], sample[3])

    def test_image_blend(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = ImageBlend(source1=test_image, source2=test_image, alpha=0.2)
            n2 = ImageBlend(source1=test_image, source2=test_image, alpha=0.5)
            Output(outputs=[n1, n2])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_image_rotate(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            cw = Constant(value=1)
            flip = Constant(value=0)
            ccw = Constant(value=-1)
            n1 = Rotate(source=test_image, clockwise=cw)
            n2 = Rotate(source=test_image, clockwise=flip)
            n3 = Rotate(source=test_image, clockwise=ccw)
            Output(outputs=[n1, n2, n3])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)
        self.assertIsInstance(sample[2], np.ndarray)

    def test_image_scale_resize(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Scale(source=test_image, scale=2.0)
            n2 = Resize(source=test_image, width=512, height=512)
            Output(outputs=[n1, n2])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_image_filter_operations(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = ConvertDepth(source=test_image, depth="Byte")

            n2 = GaussianBlur(source=test_image, size_x=3, size_y=3, sigma_x=1.0, sigma_y=1.0)
            n3 = MedianBlur(source=n1, size=7)
            n4 = AverageBlur(source=test_image, size=7)
            n5 = BilateralFilter(source=test_image, diameter=5, sigma_color=5, sigma_space=5)
            Output(outputs=[n2, n3, n4, n5])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)
        self.assertIsInstance(sample[2], np.ndarray)
        self.assertIsInstance(sample[3], np.ndarray)

    def test_image_2dfilter(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            test_kernel = ConstantImage(source=self.test_kernel)
            n1 = ImageFilter(source=test_image, kernel=test_kernel)        
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_noise_generation(self):

        with GraphBuilder() as graph:
            n1 = NormalNoise(width=10, height=10, mean=0.5, std=0.05)  
            n2 = UniformNoise(width=10, height=10, min=0.0, max=1.0)
            Output(outputs=[n1, n2])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)

    def test_image_linear_generation(self):

        with GraphBuilder() as graph:
            h = LinearImage(100, 50, 1, 50, flip=False)
            v = LinearImage(100, 50, 1, 50, flip=True)
            Output(outputs=[h, v])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0][0, 0], 1)
        self.assertEqual(sample[0][0, -1], 50)
        self.assertEqual(sample[0][-1, 0], 1)
        self.assertEqual(sample[0][-1, -1], 50)

        self.assertEqual(sample[1][0, 0], 1)
        self.assertEqual(sample[1][-1, 0], 50)
        self.assertEqual(sample[1][0, -1], 1)
        self.assertEqual(sample[1][-1, -1], 50)

    def test_image_dropout_operations(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = ImageCoarseDropout(source=test_image, probability=0.5, size_percent=0.5)
            n2 = ImageDropout(source=test_image, probability=0.5)
            Output(outputs=[n1, n2])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)

    def test_image_cut(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            bbox = MaskBoundingBox(source=test_image)
            n1 = ImageCut(source=test_image, bbox=bbox)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_solarize(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = ImageSolarize(source=test_image, threshold=128)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)

    def test_image_flip(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = Flip(source=test_image, flip=0) 
            n2 = Flip(source=test_image, flip=1) 
            n3 = Flip(source=test_image, flip=-1)        
            Output(outputs=[n1, n2, n3])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)
        self.assertIsInstance(sample[2], np.ndarray)

    def test_image_crop(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            bbox = MakeRectangle(x1=50, x2=150, y1=50, y2=150)
            n1 = ImageCrop(source=test_image, bbox=bbox)
            Output(outputs=[n1])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0].shape[0], 100)
        self.assertEqual(sample[0].shape[1], 100)

    def test_image_bounding_box(self):

        with GraphBuilder() as graph:
            test_image = ConstantImage(source=self.test_image)
            n1 = MaskBoundingBox(source=test_image)
            n2 = MakeRectangle(x1=0, x2=256, y1=0, y2=256)
            Output(outputs=[n1, n2])

        pipeline = Compiler().compile(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], sample[1])
