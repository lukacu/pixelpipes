
import unittest
import numpy as np

from ..graph import Graph, Constant, outputs
from ..compiler import Compiler
from . import Channel, Equals, Grayscale, Invert, Merge, Threshold, Moments, GetImageProperties, ConvertDepth
from .processing import ImageBlend, ImageCoarseDropout, ImageCut, ImageDropout, ImageSolarize
from .geometry import Flip, MaskBoundingBox, Resize, Rotate90, Scale, ImageCrop
from .augmentation import ImageBrightness, ImageNoise, ImagePiecewiseAffine
from .filter import AverageFilter, BilateralFilter, GaussianFilter, LinearFilter, MedianBlur
from .render import LinearImage, NormalNoise, UniformNoise
from ..geometry.rectangle import MakeRectangle

np.random.seed(0)
test_image = np.random.randint(0, 255, (32,32), dtype=np.uint8) 
test_image_rgb = np.random.randint(0, 255, (32,32,3), dtype=np.uint8) 

def clamp_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)

class TestsArithmetic(unittest.TestCase):

    def test_image_constant(self):

        img1 = test_image_rgb.astype(np.uint8)
        img2 = test_image_rgb.astype(np.uint16)
        img3 = test_image_rgb.astype(np.float32)
        img4 = test_image_rgb.astype(np.int32)

        with Graph() as graph:
            n1 = Constant(img1)
            n2 = Constant(img2)
            n3 = Constant(img3)
            n4 = Constant(img4)
            outputs(n1, n2, n3, n4)

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        np.testing.assert_equal(sample[0], img1)
        np.testing.assert_equal(sample[1], img2)
        np.testing.assert_equal(sample[2], img3)
        np.testing.assert_equal(sample[3], img4)

    def test_serialization(self):

        from pixelpipes.tests import compare_serialized

        img1 = test_image_rgb.astype(np.uint8)
        img2 = test_image_rgb.astype(np.uint16)
        img3 = test_image_rgb.astype(np.float32)
        img4 = test_image_rgb.astype(np.int32)

        with Graph() as graph:
            n1 = Constant(img1)
            n2 = Constant(img2)
            n3 = Constant(img3)
            n4 = Constant(img4)
            l = Constant([img1, img2, img3])
            #outputs(n1, n2, n3, n4, l[1])
            outputs(n1)

        compare_serialized(graph)

class TestsAugmentation(unittest.TestCase):

    def test_image_noise(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            n1 = ImageNoise(n0, 0.25)
            outputs(n0, n1)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)

    def test_image_brightness(self):

        test_image = np.random.randint(0, 255, (10,10), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ImageBrightness(n0, 0)
            o1 = ImageBrightness(n0, 1)
            outputs(o0, o1)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], np.ndarray)
        np.testing.assert_array_equal(output[0], clamp_uint8(test_image))
        np.testing.assert_array_equal(output[1], clamp_uint8(test_image + 1))

    def test_image_piecewise_affine(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ImagePiecewiseAffine(n0, 1.5, 4)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        assert not np.array_equal(output[0], test_image)

class TestsFiltering(unittest.TestCase):

    def test_gaussian_filter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = GaussianFilter(source=n0, size_x=3, size_y=3)
            o1 = GaussianFilter(source=n0, size_x=5, size_y=5)
            o2 = GaussianFilter(source=n0, size_x=7, size_y=7)
            o3 = GaussianFilter(source=n0, size_x=9, size_y=9)
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_median_blur(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = MedianBlur(source=n0, size=3)
            o1 = MedianBlur(source=n0, size=5)
            o2 = MedianBlur(source=n0, size=7)
            o3 = MedianBlur(source=n0, size=9)     
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_average_filter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = AverageFilter(n0, 3, 1)
            o1 = AverageFilter(n0, 5, 5)
            o2 = AverageFilter(n0, 7, 1)
            o3 = AverageFilter(n0, 1, 9)          
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_bilateral_filter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = BilateralFilter(source=n0, diameter=3, sigma_color=1.0, sigma_space=2.0)
            o1 = BilateralFilter(source=n0, diameter=5, sigma_color=2.0, sigma_space=1.0)
            o2 = BilateralFilter(source=n0, diameter=7, sigma_color=2.0, sigma_space=2.0)
            o3 = BilateralFilter(source=n0, diameter=9, sigma_color=3.0, sigma_space=3.0)
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

    def test_linear_filter(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_kernel_0 = np.ones((1,1), dtype=np.uint8)
        test_kernel_1 = np.ones((3,3), dtype=np.uint8)
        test_kernel_2 = np.ones((5,5), dtype=np.uint8)
        test_kernel_3 = np.ones((6,6), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            n1 = Constant(test_kernel_0)
            n2 = Constant(test_kernel_1)
            n3 = Constant(test_kernel_2)
            n4 = Constant(test_kernel_3)
            o0 = LinearFilter(source=n0, kernel=n1)
            o1 = LinearFilter(source=n0, kernel=n2)
            o2 = LinearFilter(source=n0, kernel=n3)
            o3 = LinearFilter(source=n0, kernel=n4)
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        assert not np.array_equal(output[1], test_image)
        assert not np.array_equal(output[2], test_image)
        assert not np.array_equal(output[3], test_image)

class TestsGeometry(unittest.TestCase):

    def test_image_scale(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Scale(source=n0, scale=0.25)
            o1 = Scale(source=n0, scale=0.5)
            o2 = Scale(source=n0, scale=1.0)
            o3 = Scale(source=n0, scale=2.0)
            o4 = Scale(source=n0, scale=4.0)
            outputs(o0, o1, o2, o3, o4)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0].shape[0], 64)   
        self.assertEqual(output[1].shape[0], 128)  
        self.assertEqual(output[2].shape[0], 256)   
        np.testing.assert_array_equal(output[2], test_image)
        self.assertEqual(output[3].shape[0], 512)   
        self.assertEqual(output[4].shape[0], 1024)   

    def test_image_rotate(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            r_0_0 = Rotate90(source=n0, clockwise=-1)
            r_0_1 = Rotate90(source=r_0_0, clockwise=-1)
            r_0_2 = Rotate90(source=r_0_1, clockwise=-1)
            o0 = Rotate90(source=r_0_2, clockwise=-1)
            r_1_0 = Rotate90(source=n0, clockwise=0)
            o1 = Rotate90(source=r_1_0, clockwise=0)
            r_2_0 = Rotate90(source=n0, clockwise=1)
            r_2_1 = Rotate90(source=r_2_0, clockwise=1)
            r_2_2 = Rotate90(source=r_2_1, clockwise=1)
            o2 = Rotate90(source=r_2_2, clockwise=1)
            r_3_0 = Rotate90(source=n0, clockwise=1)
            o3 = Rotate90(source=r_3_0, clockwise=-1)
            r_4_0 = Rotate90(source=n0, clockwise=-1)
            o4 = Rotate90(source=r_4_0, clockwise=1)
            r_5_0 = Rotate90(source=n0, clockwise=0)
            r_5_1 = Rotate90(source=r_5_0, clockwise=1)
            o5 = Rotate90(source=r_5_1, clockwise=1)
            outputs(o0, o1, o2, o3, o4, o5)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        np.testing.assert_array_equal(output[1], test_image)
        np.testing.assert_array_equal(output[2], test_image)
        np.testing.assert_array_equal(output[3], test_image)
        np.testing.assert_array_equal(output[4], test_image)
        np.testing.assert_array_equal(output[5], test_image)

    def test_image_flip(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            f0 = Flip(n0, True, True)
            o0 = Flip(f0, True, True)
            f1 = Flip(n0, False, True)
            o1 = Flip(f1, False, True)
            f2 = Flip(n0, True, False)
            o2 = Flip(f2, True, False)
            outputs(o0, o1, o2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        np.testing.assert_array_equal(output[1], test_image)
        np.testing.assert_array_equal(output[2], test_image)

    def test_image_resize(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Resize(source=n0, width=256, height=256)
            o1 = Resize(source=n0, width=128, height=128)
            o2 = Resize(source=n0, width=256, height=128)
            o3 = Resize(source=n0, width=128, height=256)
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        self.assertEqual(output[0].shape[0], 256)
        self.assertEqual(output[0].shape[1], 256)
        self.assertEqual(output[1].shape[0], 128)  
        self.assertEqual(output[1].shape[1], 128)      
        self.assertEqual(output[2].shape[0], 128)   
        self.assertEqual(output[2].shape[1], 256) 
        self.assertEqual(output[3].shape[0], 256)  
        self.assertEqual(output[3].shape[1], 128)  

    def test_image_mask_bounds(self):

        test_image_0 = np.random.randint(0, 255, (64,64), dtype=np.uint8)
        test_image_1 = np.random.randint(0, 255, (128,128), dtype=np.uint8)
        test_image_2 = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image_0)
            n1 = Constant(test_image_1)
            n2 = Constant(test_image_2)
            o0 = MaskBoundingBox(source=n0)
            o1 = MakeRectangle(x1=0, x2=64, y1=0, y2=64)
            o2 = MaskBoundingBox(source=n1)
            o3 = MakeRectangle(x1=0, x2=128, y1=0, y2=128)
            o4 = MaskBoundingBox(source=n2)
            o5 = MakeRectangle(x1=0, x2=256, y1=0, y2=256)
            outputs(o0, o1, o2, o3, o4, o5)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(output[2], output[3])
        np.testing.assert_array_equal(output[4], output[5])

    def test_image_crop(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            n1 = MakeRectangle(x1=0, x2=256, y1=0, y2=256)
            n2 = MakeRectangle(x1=64, x2=192, y1=64, y2=192)
            n3 = MakeRectangle(x1=65, x2=193, y1=65, y2=193)
            o0 = ImageCrop(n0, n1)
            o1 = ImageCrop(n0, n2)
            o2 = ImageCrop(n0, n3)
            outputs(o0, o1, o2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image)
        self.assertEqual(output[1].shape[0], 128)
        self.assertEqual(output[1].shape[1], 128)
        self.assertEqual(output[2].shape[0], output[1].shape[0])
        self.assertEqual(output[2].shape[1], output[1].shape[1])

    # TODO ViewImage
    # TODO ImageRemap

class TestsImage(unittest.TestCase):

    def test_constants(self):

        test_image_gray_0 = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_image_gray_1 = np.random.random_sample((256,256)).astype(dtype=np.float32)
        test_image_rgb_0 = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)
        test_image_rgb_1 = np.random.random_sample((256,256,3)).astype(dtype=np.float32)

        with Graph() as graph:
            o0 = Constant(test_image_gray_0)
            o1 = Constant(test_image_gray_1)
            o2 = Constant(test_image_rgb_0)
            o3 = Constant(test_image_rgb_1)
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_gray_0)
        np.testing.assert_array_equal(output[1], test_image_gray_1)
        np.testing.assert_array_equal(output[2], test_image_rgb_0)
        np.testing.assert_array_equal(output[3], test_image_rgb_1)

    def test_constat_list(self):
        from ..list import GetElement

        test_image_list = [
            np.random.randint(0, 255, (256,256), dtype=np.uint8),
            np.random.randint(0, 255, (256,256), dtype=np.uint8)
        ]

        with Graph() as graph:
            o0 = Constant(test_image_list)
            outputs(GetElement(o0, 0), GetElement(o0, 1))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_list[0])
        np.testing.assert_array_equal(output[1], test_image_list[1])

    def test_image_properties(self):
        
        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            n1 = GetImageProperties(n0)
            o0 = n1["width"]
            o1 = n1["height"]
            o2 = n1["channels"]
            o3 = n1["depth"]
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 256)        
        self.assertEqual(output[1], 256)
        self.assertEqual(output[2], 1)
        self.assertEqual(output[3], 8)

    def test_image_depth(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ConvertDepth(n0, depth="Char")
            o1 = ConvertDepth(n0, depth="Short")
            o2 = ConvertDepth(n0, depth="Float")
            o3 = ConvertDepth(n0, depth="Integer")
            outputs(o0, o1, o2, o3)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIs(output[0].dtype, np.dtype('uint8'))
        self.assertIs(output[1].dtype, np.dtype('int16'))
        self.assertIs(output[2].dtype, np.dtype('float32'))
        self.assertIs(output[3].dtype, np.dtype('int32'))

    def test_grayscale(self):
  
        test_image_gray = np.random.randint(1, 254, (256,256), dtype=np.uint8)
        test_image_rgb = np.ndarray((256,256,3), dtype=np.uint8)
        test_image_rgb[:,:,0] = test_image_gray - 1
        test_image_rgb[:,:,1] = test_image_gray 
        test_image_rgb[:,:,2] = test_image_gray + 1

        with Graph() as graph:
            n0 = Constant(test_image_rgb)
            o0 = Grayscale(n0)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image_gray)

    def test_image_threshold(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Threshold(n0, threshold=200)
            o1 = Threshold(n0, threshold=100)
            outputs(o0, o1)

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

    def test_image_invert(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Invert(n0)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], 255 - test_image)

    def test_image_equals(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Equals(n0, value=128)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        temp = np.copy(test_image)
        temp[temp!=128] = 0
        temp[temp==128] = 255
        np.testing.assert_array_equal(output[0], temp)

    def test_image_channel(self):

        test_image = np.random.randint(0, 255, (256,256,3), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Channel(n0, index=0)
            o1 = Channel(n0, index=1)
            o2 = Channel(n0, index=2)
            outputs(o0, o1, o2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image[..., 0])
        np.testing.assert_array_equal(output[1], test_image[..., 1])
        np.testing.assert_array_equal(output[2], test_image[..., 2])

    def test_image_merge(self):

        test_image = np.random.randint(0, 255, (4,4,3), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image[..., 0])
            n1 = Constant(test_image[..., 1])
            n2 = Constant(test_image[..., 2])
            o0 = Merge(n0, n1, n2)
            outputs(o0, n0, n1, n2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[1], test_image[..., 0])
        np.testing.assert_array_equal(output[0], test_image)


    def test_image_moments(self):

        m_int = np.random.randint(0, 255, (10,10), dtype=np.uint8)
        m_bin = np.random.randint(0, 1, (10,10), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(m_int)
            n1 = Constant(m_bin)
            o0 = Moments(n0, binary=False)
            o1 = Moments(n1, binary=True)
            outputs(o0, o1)
            
        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        def m(i, j, I):
            sum = 0
            for y in range(I.shape[0]):
                for x in range(I.shape[1]):
                    sum += x**i * y**j * I[y,x]
            return float(sum)

        moments_int = np.array([m(0,0,m_int), m(0,1,m_int), m(1,0,m_int), m(1,1,m_int)])
        moments_bin = np.array([m(0,0,m_bin), m(0,1,m_bin), m(1,0,m_bin), m(1,1,m_bin)])
        np.testing.assert_array_equal(output[0], moments_int)
        np.testing.assert_array_equal(output[1], moments_bin)

class TestsProcessing(unittest.TestCase):

    def test_image_blend(self):

        test_image_0 = np.random.randint(0, 255, (256,256), dtype=np.uint8)
        test_image_1 = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image_0)
            n1 = Constant(test_image_1)
            o0 = ImageBlend(n0, n0, alpha=0.25)
            o1 = ImageBlend(n0, n1, alpha=0.50)
            o2 = ImageBlend(n0, n1, alpha=1.0)
            outputs(o0, o1, o2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        blend = (test_image_0.astype(np.float32) + test_image_1.astype(np.float32)) / 2
        np.testing.assert_array_equal(output[0], test_image_0)
        np.testing.assert_array_equal(output[1], blend.round().astype(np.uint8))
        np.testing.assert_array_equal(output[2], test_image_0)

    def test_dropout(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ImageDropout(source=n0, probability=0.5)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)

    def test_coarse_dropout(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ImageCoarseDropout(source=n0, probability=0.5, size=0.5)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        assert not np.array_equal(output[0], test_image)

    def test_cut(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            n1 = MakeRectangle(x1=0, x2=64, y1=0, y2=64)
            n2 = MaskBoundingBox(source=n0)
            o0 = ImageCut(n0, n1)
            o1 = ImageCut(n0, n2)
            outputs(o0, o1)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        test_image[0:64, 0:64] = 0
        np.testing.assert_array_equal(output[0], test_image)
        test_image[:, :] = 0
        np.testing.assert_array_equal(output[1], test_image)

    def test_solarize(self):

        test_image = np.random.randint(0, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = ImageSolarize(source=n0, threshold=-1)
            o1 = ImageSolarize(source=n0, threshold=255)
            outputs(o0, o1, n0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        max_val = np.amax(test_image)
        np.testing.assert_array_equal(output[0], max_val - test_image + 1)
        np.testing.assert_array_equal(output[1], test_image)

class TestsRender(unittest.TestCase):

    def test_normal_noise(self):

        with Graph() as graph:
            o0 = NormalNoise(width=10, height=10, mean=0.5, std=0.05)  
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        self.assertEqual(output[0].shape, (10, 10))

    def test_uniform_noise(self):

        with Graph() as graph:
            o0 = UniformNoise(width=10, height=10, min=0.0, max=1.0)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)
        self.assertEqual(output[0].shape, (10, 10))

    def test_linear_image(self):

        with Graph() as graph:
            h = LinearImage(100, 50, 1, 50, flip=False)
            v = LinearImage(100, 50, 1, 50, flip=True)
            outputs(h, v)

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
