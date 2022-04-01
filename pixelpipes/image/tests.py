
import unittest
import numpy as np
from pixelpipes.list import ListElement

from pixelpipes.resource import GetResource

from ..graph import GraphBuilder, Output, Constant
from ..compiler import Compiler
from ..complex import GetElement
from . import ConstantImage, Channel, Equals, Grayscale, Invert, Merge, Threshold, Moments, GetImageProperties, ConvertDepth, ConstantImageList
from .arithemtic import ImageAdd, ImageMultiply, ImageSubtract
from .processing import ImageBlend, ImageCoarseDropout, ImageCut, ImageDropout, ImageSolarize
from .geometry import Flip, MaskBoundingBox, Resize, Rotate, Scale, ImageCrop
from .augmentation import ImageBrightness, ImageNoise, ImagePiecewiseAffine
from .filter import AverageFilter, BilateralFilter, GaussianFilter, LinearFilter, MedianBlur
from .render import LinearImage, NormalNoise, UniformNoise
from ..geometry.rectangle import MakeRectangle
import pixelpipes


class Tests(unittest.TestCase):

    test_image = np.random.randint(0, 254, (256,256), dtype=np.uint8)
    test_image_rgb = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)


    """
    arithmetic.py
    """

    def test_image_constant(self):

        img1 = self.test_image_rgb.astype(np.uint8)
        img2 = self.test_image_rgb.astype(np.uint16)
        img3 = self.test_image_rgb.astype(np.float32)
        img4 = self.test_image_rgb.astype(np.float64)

        with GraphBuilder() as graph:
            n1 = ConstantImage(source=img1)
            n2 = ConstantImage(source=img2)
            n3 = ConstantImage(source=img3)
            n4 = ConstantImage(source=img4)
            Output(outputs=[n1, n2, n3, n4])

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], img1)
        np.testing.assert_equal(sample[1], img2)
        np.testing.assert_equal(sample[2], img3)
        np.testing.assert_equal(sample[3], img4)

    def test_serialization(self):

        from pixelpipes.tests import compare_serialized

        img1 = self.test_image_rgb.astype(np.uint8)
        img2 = self.test_image_rgb.astype(np.uint16)
        img3 = self.test_image_rgb.astype(np.float32)
        img4 = self.test_image_rgb.astype(np.float64)

        with GraphBuilder() as graph:
            n1 = ConstantImage(source=img1)
            n2 = ConstantImage(source=img2)
            n3 = ConstantImage(source=img3)
            n4 = ConstantImage(source=img4)
            l = ConstantImageList([img1, img2, img3])
            Output(outputs=[n1, n2, n3, n4, ListElement(l, 1)])

        compare_serialized(graph)

    def test_image_arithmetic_ImageAdd(self):


        with GraphBuilder() as graph:
            n0 = ConstantImage(source=self.test_image)
            n1 = Constant(value=0)
            n2 = Constant(value=1)       
            o0 = ImageAdd(source1=n0, source2=n0)       
            o1 = ImageAdd(source1=n0, source2=n1)
            o2 = ImageAdd(source1=n1, source2=n0)
            o3 = ImageAdd(source1=n0, source2=n2)
            o4 = ImageAdd(source1=n2, source2=n0)     
            Output(outputs=[o0, o1, o2, o3, o4])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        temp = self.test_image.astype(dtype=np.uint16, copy=True)
        temp = temp + temp
        temp[temp>255] = 255
        temp = temp.astype(dtype=np.uint8, copy=False)
        np.testing.assert_array_equal(output[0], temp)
        np.testing.assert_array_equal(output[1], self.test_image)
        np.testing.assert_array_equal(output[2], self.test_image)
        np.testing.assert_array_equal(output[3], self.test_image + 1)
        np.testing.assert_array_equal(output[4], self.test_image + 1)

    def test_image_arithmetic_ImageSubtract(self):

        test_image = np.random.randint(1, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = Constant(value=0)
            n2 = Constant(value=1)    
            o0 = ImageSubtract(source1=n0, source2=n0)       
            o1 = ImageSubtract(source1=n0, source2=n1)
            o2 = ImageSubtract(source1=n1, source2=n0)
            o3 = ImageSubtract(source1=n0, source2=n2)
            o4 = ImageSubtract(source1=n2, source2=n0)  
            Output(outputs=[o0, o1, o2, o3, o4])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], np.zeros((256,256), dtype=np.uint8))
        np.testing.assert_array_equal(output[1], test_image)
        np.testing.assert_array_equal(output[2], test_image)
        np.testing.assert_array_equal(output[3], test_image - 1)
        np.testing.assert_array_equal(output[4], test_image - 1)

    def test_image_arithmetic_ImageMultiply(self):

        test_image = np.random.randint(1, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = Constant(value=0)
            n2 = Constant(value=1)
            n3 = Constant(value=255)    
            o0 = ImageMultiply(source1=n0, source2=n0)
            o1 = ImageMultiply(source1=n0, source2=n1)
            o2 = ImageMultiply(source1=n1, source2=n0)
            o3 = ImageMultiply(source1=n0, source2=n2)
            o4 = ImageMultiply(source1=n2, source2=n0)
            o5 = ImageMultiply(source1=n0, source2=n3)
            o6 = ImageMultiply(source1=n3, source2=n0)
            Output(outputs=[o0, o1, o2, o3, o4, o5, o6])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[1], test_image * 0)
        np.testing.assert_array_equal(output[2], test_image * 0)
        np.testing.assert_array_equal(output[3], test_image)
        np.testing.assert_array_equal(output[4], test_image)
        np.testing.assert_array_equal(output[5], np.ones((256,256), dtype=np.uint8) * 255)
        np.testing.assert_array_equal(output[6], np.ones((256,256), dtype=np.uint8) * 255)

    """
    augmentation.py
    """

    def test_image_augmentation_ImageNoise(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = Constant(value=0.0)
            n2 = Constant(value=0.25)
            o0 = ImageNoise(source=n0, amount=n1)
            o1 = ImageNoise(source=n0, amount=n2)
            Output(outputs=[o0, o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)

    def test_image_augmentation_ImageBrightness(self):

        test_image = np.random.randint(0, 254, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = Constant(value=0)
            n2 = Constant(value=1)
            o0 = ImageBrightness(source=n0, amount=n1)
            o1 = ImageBrightness(source=n0, amount=n2)
            Output(outputs=[o0, o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], np.ndarray)
        np.testing.assert_array_equal(output[0], test_image)
        np.testing.assert_array_equal(output[1], test_image + 1)

    def test_image_augmentation_ImagePiecewiseAffine(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = ImagePiecewiseAffine(n0, 1.5, 4)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        assert not np.array_equal(output[0], test_image)

    """
    filter.py
    """

    """
    def test_image_filter_GaussianFunction(self):

        # TODO FIX Operation missing

        with GraphBuilder() as graph:
            o1 = GaussianFunction(size_x=256, size_y=256, mean_x=0.5, mean_y=0.5, sigma_x=1.0, sigma_y=1.0)
            Output(outputs=[o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0].shape[0], 256)   
        self.assertEqual(output[0].shape[1], 256)   
    """

    def test_image_filter_GaussianFilter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = GaussianFilter(source=n0, size_x=3, size_y=3)
            o1 = GaussianFilter(source=n0, size_x=5, size_y=5)
            o2 = GaussianFilter(source=n0, size_x=7, size_y=7)
            o3 = GaussianFilter(source=n0, size_x=9, size_y=9)
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_image_filter_MedianBlur(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = MedianBlur(source=n0, size=3)
            o1 = MedianBlur(source=n0, size=5)
            o2 = MedianBlur(source=n0, size=7)
            o3 = MedianBlur(source=n0, size=9)     
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_image_filter_AverageFilter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = AverageFilter(n0, 3, 1)
            o1 = AverageFilter(n0, 5, 5)
            o2 = AverageFilter(n0, 7, 1)
            o3 = AverageFilter(n0, 1, 9)          
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_image_filter_BilateralFilter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = BilateralFilter(source=n0, diameter=3, sigma_color=1.0, sigma_space=2.0)
            o1 = BilateralFilter(source=n0, diameter=5, sigma_color=2.0, sigma_space=1.0)
            o2 = BilateralFilter(source=n0, diameter=7, sigma_color=2.0, sigma_space=2.0)
            o3 = BilateralFilter(source=n0, diameter=9, sigma_color=3.0, sigma_space=3.0)
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_image_filter_LinearFilter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_kernel_0 = np.ones((1,1), dtype=np.uint8)
        test_kernel_1 = np.ones((3,3), dtype=np.uint8)
        test_kernel_2 = np.ones((5,5), dtype=np.uint8)
        test_kernel_3 = np.ones((6,6), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = ConstantImage(source=test_kernel_0)
            n2 = ConstantImage(source=test_kernel_1)
            n3 = ConstantImage(source=test_kernel_2)
            n4 = ConstantImage(source=test_kernel_3)
            o0 = LinearFilter(source=n0, kernel=n1)
            o1 = LinearFilter(source=n0, kernel=n2)
            o2 = LinearFilter(source=n0, kernel=n3)
            o3 = LinearFilter(source=n0, kernel=n4)
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    """
    geometry.py
    """

    def test_image_geometry_Scale(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Scale(source=n0, scale=0.25)
            o1 = Scale(source=n0, scale=0.5)
            o2 = Scale(source=n0, scale=1.0)
            o3 = Scale(source=n0, scale=2.0)
            o4 = Scale(source=n0, scale=4.0)
            Output(outputs=[o0, o1, o2, o3, o4])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0].shape[0], 64)   
        self.assertEqual(output[1].shape[0], 128)  
        self.assertEqual(output[2].shape[0], 256)   
        np.testing.assert_array_equal(output[2], test_image)
        self.assertEqual(output[3].shape[0], 512)   
        self.assertEqual(output[4].shape[0], 1024)   

    def test_image_geometry_Rotate(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            r_0_0 = Rotate(source=n0, clockwise=-1)
            r_0_1 = Rotate(source=r_0_0, clockwise=-1)
            r_0_2 = Rotate(source=r_0_1, clockwise=-1)
            o0 = Rotate(source=r_0_2, clockwise=-1)
            r_1_0 = Rotate(source=n0, clockwise=0)
            o1 = Rotate(source=r_1_0, clockwise=0)
            r_2_0 = Rotate(source=n0, clockwise=1)
            r_2_1 = Rotate(source=r_2_0, clockwise=1)
            r_2_2 = Rotate(source=r_2_1, clockwise=1)
            o2 = Rotate(source=r_2_2, clockwise=1)
            r_3_0 = Rotate(source=n0, clockwise=1)
            o3 = Rotate(source=r_3_0, clockwise=-1)
            r_4_0 = Rotate(source=n0, clockwise=-1)
            o4 = Rotate(source=r_4_0, clockwise=1)
            r_5_0 = Rotate(source=n0, clockwise=0)
            r_5_1 = Rotate(source=r_5_0, clockwise=1)
            o5 = Rotate(source=r_5_1, clockwise=1)
            Output(outputs=[o0, o1, o2, o3, o4, o5])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        np.testing.assert_array_equal(output[1], test_image)
        np.testing.assert_array_equal(output[2], test_image)
        np.testing.assert_array_equal(output[3], test_image)
        np.testing.assert_array_equal(output[4], test_image)
        np.testing.assert_array_equal(output[5], test_image)

    def test_image_geometry_Flip(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            f0 = Flip(source=n0, flip=-1)
            o0 = Flip(source=f0, flip=-1)
            f1 = Flip(source=n0, flip=0)
            o1 = Flip(source=f1, flip=0)
            f2 = Flip(source=n0, flip=1)
            o2 = Flip(source=f2, flip=1)
            Output(outputs=[o0, o1, o2])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        np.testing.assert_array_equal(output[1], test_image)
        np.testing.assert_array_equal(output[2], test_image)

    def test_image_geometry_Resize(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Resize(source=n0, width=256, height=256)
            o1 = Resize(source=n0, width=128, height=128)
            o2 = Resize(source=n0, width=256, height=128)
            o3 = Resize(source=n0, width=128, height=256)
            o4 = Resize(source=n0, width=512, height=512)
            o5 = Resize(source=n0, width=512, height=256)
            o6 = Resize(source=n0, width=256, height=512)
            Output(outputs=[o0, o1, o2, o3, o4, o5, o6])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        self.assertEqual(output[0].shape[0], 256)
        self.assertEqual(output[0].shape[1], 256)
        self.assertEqual(output[1].shape[0], 128)  
        self.assertEqual(output[1].shape[1], 128)      
        self.assertEqual(output[2].shape[0], 256)  
        self.assertEqual(output[2].shape[1], 128)   
        self.assertEqual(output[3].shape[0], 128)  
        self.assertEqual(output[3].shape[1], 256) 
        self.assertEqual(output[4].shape[0], 512)  
        self.assertEqual(output[4].shape[1], 512) 
        self.assertEqual(output[5].shape[0], 512)  
        self.assertEqual(output[5].shape[1], 256) 
        self.assertEqual(output[6].shape[0], 256)  
        self.assertEqual(output[6].shape[1], 512) 

    def test_image_geometry_MaskBoundingBox(self):

        test_image_0 = np.random.randint(0, 255, (64,64), dtype=np.uint8)
        test_image_1 = np.random.randint(0, 255, (128,128), dtype=np.uint8)
        test_image_2 = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image_0)
            n1 = ConstantImage(source=test_image_1)
            n2 = ConstantImage(source=test_image_2)
            o0 = MaskBoundingBox(source=n0)
            o1 = MakeRectangle(x1=0, x2=64, y1=0, y2=64)
            o2 = MaskBoundingBox(source=n1)
            o3 = MakeRectangle(x1=0, x2=128, y1=0, y2=128)
            o4 = MaskBoundingBox(source=n2)
            o5 = MakeRectangle(x1=0, x2=256, y1=0, y2=256)
            Output(outputs=[o0, o1, o2, o3, o4, o5])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(output[2], output[3])
        np.testing.assert_array_equal(output[4], output[5])

    def test_image_geometry_ImageCrop(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = MakeRectangle(x1=0, x2=256, y1=0, y2=256)
            n2 = MakeRectangle(x1=64, x2=192, y1=64, y2=192)
            n3 = MakeRectangle(x1=65, x2=193, y1=65, y2=193)
            o0 = ImageCrop(source=n0, bbox=n1)
            o1 = ImageCrop(source=n0, bbox=n2)
            o2 = ImageCrop(source=n0, bbox=n3)
            Output(outputs=[o0, o1, o2])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        self.assertEqual(output[1].shape[0], 128)
        self.assertEqual(output[1].shape[1], 128)
        self.assertEqual(output[2].shape[0], output[1].shape[0])
        self.assertEqual(output[2].shape[1], output[1].shape[1])

    # TODO ViewImage
    # TODO ImageRemap

    """
    image.py
    """

    def test_image_image_ConstantImage(self):

        test_image_gray_0 = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_image_gray_1 = np.random.random_sample((256,256))
        test_image_rgb_0 = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)
        test_image_rgb_1 = np.random.random_sample((256,256,3))

        with GraphBuilder() as graph:
            o0 = ConstantImage(source=test_image_gray_0)
            o1 = ConstantImage(source=test_image_gray_1)
            o2 = ConstantImage(source=test_image_rgb_0)
            o3 = ConstantImage(source=test_image_rgb_1)
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_gray_0)
        np.testing.assert_array_equal(output[1], test_image_gray_1)
        np.testing.assert_array_equal(output[2], test_image_rgb_0)
        np.testing.assert_array_equal(output[3], test_image_rgb_1)

    def test_image_image_ConstantImageList(self):

        from ..list import ListElement

        test_image_list = [
            np.random.randint(0, 255, (256,256), dtype=np.uint8),
            np.random.randint(0, 255, (256,256), dtype=np.uint8)
        ]

        with GraphBuilder() as graph:
            o0 = ConstantImageList(test_image_list)
            Output(outputs=[ListElement(o0, 0), ListElement(o0, 1)])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_list[0])
        np.testing.assert_array_equal(output[1], test_image_list[1])

    def test_image_image_ImageList(self):
        
        import os
        from pixelpipes.image import ImageDirectory

        example_images = os.path.join(os.path.dirname(os.path.dirname(pixelpipes.__file__)), "examples", "images")

        with GraphBuilder() as graph:
            l = ImageDirectory(path=example_images, grayscale=False)
            Output(outputs=[GetElement(GetResource(l, 0), "image")])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)

    def test_image_image_GetImageProperties(self):
        
        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = GetImageProperties(source=n0)
            o0 = GetElement(n1, element="width")
            o1 = GetElement(n1, element="height")
            o2 = GetElement(n1, element="channels")
            o3 = GetElement(n1, element="depth")
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 256)        
        self.assertEqual(output[1], 256)
        self.assertEqual(output[2], 1)
        self.assertEqual(output[3], 8)

    def test_image_image_ConvertDepth(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(test_image)
            o0 = ConvertDepth(source=n0, depth="Byte")
            o1 = ConvertDepth(source=n0, depth="Short")
            o2 = ConvertDepth(source=n0, depth="Float")
            o3 = ConvertDepth(source=n0, depth="Double")
            Output(outputs=[o0, o1, o2, o3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIs(output[0].dtype, np.dtype('uint8'))
        self.assertIs(output[1].dtype, np.dtype('uint16'))
        self.assertIs(output[2].dtype, np.dtype('float32'))
        self.assertIs(output[3].dtype, np.dtype('float64'))

    def test_image_image_Grayscale(self):
  
        test_image_gray = np.random.randint(1, 254, (256,256), dtype=np.uint8)
        test_image_rgb = np.ndarray((256,256,3), dtype=np.uint8)
        test_image_rgb[:,:,0] = test_image_gray - 1
        test_image_rgb[:,:,1] = test_image_gray 
        test_image_rgb[:,:,2] = test_image_gray + 1

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image_rgb)
            o0 = Grayscale(source=n0)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_gray)

    def test_image_image_Threshold(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Threshold(source=n0, threshold=200)
            o1 = Threshold(source=n0, threshold=100)
            Output(outputs=[o0, o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        temp_0 = np.copy(test_image)
        temp_0[temp_0>200] = 255
        temp_0[temp_0<=200] = 0
        temp_1 = np.copy(test_image)
        temp_1[temp_1>100] = 255
        temp_1[temp_1<=100] = 0
        np.testing.assert_array_equal(output[0], temp_0)
        np.testing.assert_array_equal(output[1], temp_1)

    def test_image_image_Invert(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Invert(source=n0)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], 255 - test_image)

    def test_image_image_Equals(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Equals(source=n0, value=128)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        temp = np.copy(test_image)
        temp[temp!=128] = 0
        temp[temp==128] = 255
        np.testing.assert_array_equal(output[0], temp)

    def test_image_image_Channel(self):

        test_image = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = Channel(source=n0, index=0)
            o1 = Channel(source=n0, index=1)
            o2 = Channel(source=n0, index=2)
            Output(outputs=[o0, o1, o2])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image[..., 0])
        np.testing.assert_array_equal(output[1], test_image[..., 1])
        np.testing.assert_array_equal(output[2], test_image[..., 2])

    def test_image_image_Merge(self):

        test_image = np.random.randint(0, 255, (4,4,3), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image[..., 0])
            n1 = ConstantImage(source=test_image[..., 1])
            n2 = ConstantImage(source=test_image[..., 2])
            o0 = Merge(source1=n0, source2=n1, source3=n2)
            Output(outputs=[o0, n0, n1, n2])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)

    def test_image_image_Moments(self):

        m_int = np.random.randint(0, 255, (10,10), dtype=np.uint8)
        m_bin = np.random.randint(0, 1, (10,10), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=m_int)
            n1 = ConstantImage(source=m_bin)
            o0 = Moments(source=n0, binary=False)
            o1 = Moments(source=n1, binary=True)
            Output(outputs=[o0, o1])
            
        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        def m(i, j, I):
            sum = 0
            for y in range(I.shape[0]):
                for x in range(I.shape[1]):
                    sum += x**i * y**j * I[y,x]
            return float(sum)

        moments_int = np.array([[m(0,0,m_int)], [m(0,1,m_int)], [m(1,0,m_int)], [m(1,1,m_int)]])
        moments_bin = np.array([[m(0,0,m_bin)], [m(0,1,m_bin)], [m(1,0,m_bin)], [m(1,1,m_bin)]])
        np.testing.assert_array_equal(output[0], moments_int)
        np.testing.assert_array_equal(output[1], moments_bin)

    """
    processing.py
    """

    def test_image_processing_ImageBlend(self):

        test_image_0 = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_image_1 = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image_0)
            n1 = ConstantImage(source=test_image_1)
            o0 = ImageBlend(source1=n0, source2=n0, alpha=0.25)
            o1 = ImageBlend(source1=n0, source2=n1, alpha=0.50)
            o2 = ImageBlend(source1=n0, source2=n1, alpha=1.0)
            Output(outputs=[o0, o1, o2])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        blend = (test_image_0.astype(np.float32) + test_image_1.astype(np.float32)) / 2
        np.testing.assert_array_equal(output[0], test_image_0)
        np.testing.assert_array_equal(output[1], blend.round().astype(np.uint8))
        np.testing.assert_array_equal(output[2], test_image_0)

    def test_image_processing_ImageDropout(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = ImageDropout(source=n0, probability=0.5)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)

    def test_image_processing_ImageCoarseDropout(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = ImageCoarseDropout(source=n0, probability=0.5, size_percent=0.5)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)

    def test_image_processing_ImageCut(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            n1 = MakeRectangle(x1=0, x2=64, y1=0, y2=64)
            n2 = MaskBoundingBox(source=n0)
            o0 = ImageCut(source=n0, bbox=n1)
            o1 = ImageCut(source=n0, bbox=n2)
            Output(outputs=[o0, o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        test_image[0:64, 0:64] = 0
        np.testing.assert_array_equal(output[0], test_image)
        test_image[:, :] = 0
        np.testing.assert_array_equal(output[1], test_image)

    def test_image_processing_ImageSolarize(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with GraphBuilder() as graph:
            n0 = ConstantImage(source=test_image)
            o0 = ImageSolarize(source=n0, threshold=-1)
            o1 = ImageSolarize(source=n0, threshold=255)
            Output(outputs=[o0, o1])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        max_val = np.amax(test_image)
        np.testing.assert_array_equal(output[0], max_val - test_image + 1)
        np.testing.assert_array_equal(output[1], test_image)

    """
    render.py
    """

    def test_image_render_NormalNoise(self):

        with GraphBuilder() as graph:
            o0 = NormalNoise(width=10, height=10, mean=0.5, std=0.05)  
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)

    def test_image_render_UniformNoise(self):

        with GraphBuilder() as graph:
            o0 = UniformNoise(width=10, height=10, min=0.0, max=1.0)
            Output(outputs=[o0])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)

    def test_image_render_LinearImage(self):

        with GraphBuilder() as graph:
            h = LinearImage(100, 50, 1, 50, flip=False)
            v = LinearImage(100, 50, 1, 50, flip=True)
            Output(outputs=[h, v])

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0][0, 0], 1)
        self.assertEqual(sample[0][0, -1], 50)
        self.assertEqual(sample[0][-1, 0], 1)
        self.assertEqual(sample[0][-1, -1], 50)
        self.assertEqual(sample[1][0, 0], 1)
        self.assertEqual(sample[1][-1, 0], 50)
        self.assertEqual(sample[1][0, -1], 1)
        self.assertEqual(sample[1][-1, -1], 50)
