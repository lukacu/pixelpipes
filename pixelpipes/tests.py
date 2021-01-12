
import unittest

import numpy as np

from pixelpipes import *
from pixelpipes.nodes.resources import *

class TestPipes(unittest.TestCase):

    def test_dropout_operations(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(ImageCoarseDropout(source=n3, probability=0.5, size_percent=0.5))
        n5 = builder.add(ImageDropout(source=n3, probability=0.5))

        builder.add(Output(outputs=[n4, n5]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)

    def test_convert_depth(self): 
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(ConvertDepth(source=n3, depth="BYTE"))
        n5 = builder.add(ConvertDepth(source=n3, depth="SHORT"))
        n6 = builder.add(ConvertDepth(source=n3, depth="FLOAT"))
        n7 = builder.add(ConvertDepth(source=n3, depth="DOUBLE"))

        builder.add(Output(outputs=[n4, n5, n6, n7]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIs(sample[0].dtype, np.dtype('ubyte'))
        self.assertIs(sample[1].dtype, np.dtype('short'))
        self.assertIs(sample[2].dtype, np.dtype('float32'))
        self.assertIs(sample[3].dtype, np.dtype('float64'))

    def test_noise_generation(self):
        builder = GraphBuilder()
        n1 = builder.add(NormalNoise(width=10, height=10, mean=0.5, std=0.05))  
        n2 = builder.add(UniformNoise(width=10, height=10, min=0.0, max=1.0))

        builder.add(Output(outputs=[n1, n2]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)

    def test_add_subtract(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(ImageSubtract(source_1=n3, source_2=n3))  
        n5 = builder.add(ImageAdd(source_1=n4, source_2=n3))

        builder.add(Output(outputs=[n3, n5]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_multiply(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        # CONVERT TO FLOAT TO AVOID ROUNDING ERRORS
        n4 = builder.add(ConvertDepth(source=n3, depth="FLOAT"))
        n5 = builder.add(ImageMultiply(source=n4, multiplier=0.5))  
        n6 = builder.add(ImageMultiply(source=n5, multiplier=2.0))  

        builder.add(Output(outputs=[n4, n6]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_blend(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(ImageBlend(source_1=n3, source_2=n3, alpha=0.5))  

        builder.add(Output(outputs=[n3, n4]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_filter_operations(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(GaussianBlur(source=n3, size_x=3, size_y=3, sigma_x=1.0, sigma_y=1.0))
        n5 = builder.add(ConvertDepth(source=n3, depth="BYTE"))
        n6 = builder.add(MedianBlur(source=n5, size=7))
        n7 = builder.add(AverageBlur(source=n3, size=7))
        n8 = builder.add(BilateralFilter(source=n3, d=5, sigma_color=5, sigma_space=5))

        builder.add(Output(outputs=[n4, n6, n7, n8]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertIsInstance(sample[1], np.ndarray)
        self.assertIsInstance(sample[2], np.ndarray)
        self.assertIsInstance(sample[3], np.ndarray)

    def test_channel_extract_merge(self):
        builder = GraphBuilder()
        n1 = builder.add(ImageDirectory(path="./pixelpipes/test_img"))
        n2 = builder.add(GetRandomResource(resources=n1))
        n3 = builder.add(ExtractField(resource=n2, field="image"))

        n4 = builder.add(Channel(source=n3, index=0))
        n5 = builder.add(Channel(source=n3, index=1))
        n6 = builder.add(Channel(source=n3, index=2))
        n7 = builder.add(Merge(source_0=n4, source_1=n5, source_2=n6))

        builder.add(Output(outputs=[n3, n7]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)
        sample = pipeline.run(1, engine.Convert.NUMPY)

        np.testing.assert_array_equal(sample[0], sample[1])

    def test_numerical(self):
        builder = GraphBuilder()
        n1 = builder.add(Constant(value=5))
        n2 = builder.add(Constant(value=15))
        n3 = builder.add(Expression(source="((x ^ 2 - y) * 2) / 5 + 2", variables=dict(x=n1, y=n2)))

        builder.add(Output(outputs=[n3]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        self.assertEqual(pipeline.run(1)[0], 6)

    def test_view(self):
        with GraphBuilder() as builder:
            n1 = AffineView(x=0, y=1)
            n2 = RotateView(angle=3)

            Output(outputs=[Chain(inputs=[n1, n2])])
        compiler = Compiler()
        pipeline = compiler.compile(builder)
        pipeline.run(1)

        # TODO: calculate transform manually, compare matrices
        #self.assertEqual(pipeline.run(1)[0], 3)

    def test_numpy_out(self):
        builder = GraphBuilder()
        n1 = builder.add(Constant(value=5))
        n2 = builder.add(IdentityView())

        builder.add(Output(outputs=[n1, n2]))
        compiler = Compiler()
        graph = builder.build()
        pipeline = compiler.compile(graph)

        sample = pipeline.run(1, engine.Convert.NUMPY)

        self.assertIsInstance(sample[0], np.ndarray)
        self.assertEqual(sample[0][0], 5)
        self.assertIsInstance(sample[1], np.ndarray)

    def test_jumps(self):

        with GraphBuilder() as graph:
            a = Constant(value=20)
            b = Constant(value=30)
            c = a + b
            d = Switch(inputs=[a, b, c], weights=[0.2, 0.2, 0.9])
            b = Constant(value=4)
            Output(outputs=[Switch(inputs=[d, b, a - b], weights=[0.5, 0.5, 0.5])])

        pipeline1 = Compiler().compile(graph)
        pipeline2 = Compiler(predictive=False).compile(graph)

        for i in range(1, 100):
            a = pipeline1.run(i)
            b = pipeline2.run(i)
            self.assertEqual(a[0], b[0])
