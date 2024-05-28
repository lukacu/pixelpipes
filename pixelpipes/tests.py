import unittest
import numpy as np

from .flow import Conditional
from .list import Table, Range
from .numbers import Floor, Round, SampleUnform, Ceil
from .types import Integer, Char, Float, IntegerList, FloatList, Token, Boolean
from .compiler import Compiler
from .graph import Constant, Debug, Graph, SampleIndex, outputs

def compare_serialized(graph, samples=100):
    import os
    import tempfile
    import numpy.testing as npt
    from .compiler import Compiler
    from . import write_pipeline, read_pipeline

    compiler = Compiler()

    pipeline1 = compiler.build(graph)

    tmpfd, tmpname = tempfile.mkstemp(suffix=".pxpi")

    os.close(tmpfd)

    write_pipeline(tmpname, pipeline1)

    pipeline2 = read_pipeline(tmpname)

    for i in range(1, samples):
        a = pipeline1.run(i)
        b = pipeline2.run(i)
        assert len(a) == len(b), "Output does not match"
        for x, y in zip(a, b):
            npt.assert_equal(x, y)

    #os.remove(tmpname)

class TestBase(unittest.TestCase):

    def setUp(self) -> None:
        from pixelpipes import pypixelpipes
        if hasattr(pypixelpipes, "_refcount"):
            self.__startobj = pypixelpipes._refcount()
        return super().setUp()

    def tearDown(self) -> None:
        from pixelpipes import pypixelpipes
        if hasattr(pypixelpipes, "_refcount"):
            leftobj = pypixelpipes._refcount() - self.__startobj
            print("\n **** Remaining objects:", leftobj, "(total:", pypixelpipes._refcount(), ")**** ")
        return super().tearDown()

    def compare_arrays(self, a, b):
        """ Compares two arrays ignoring singleton dimensions at the beginning or end"""
        a = np.squeeze(a)
        b = np.squeeze(b)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)

class TypesTests(TestBase):

    def test_casting_scalar(self):

        self.assertTrue(Integer().castable(Char()))
        self.assertTrue(Integer().castable(Boolean()))
        self.assertTrue(Float().castable(Char()))
        self.assertTrue(Float().castable(Integer()))
        self.assertFalse(Integer().castable(Float()))
        self.assertFalse(Boolean().castable(Char()))

    def test_modifiers(self):

        self.assertEqual(Integer().push(1).rank, 0)
        self.assertEqual(Integer().push(1)[1], 1)
        self.assertEqual(IntegerList().pop().rank, 0)
        self.assertEqual(IntegerList().push().rank, 2)

        self.assertEqual(Token(None, 5, 5).squeeze().rank, 2)
        self.assertEqual(Token(None, 5, 1, 1).squeeze().rank, 1)
        self.assertEqual(Token(None, 1, 1, 1).squeeze().rank, 0)
        self.assertEqual(Token(None, 5, 1, 1, 5, 1).squeeze().rank, 4)

    def test_casting_list(self):

        self.assertFalse(Integer().castable(IntegerList()))
        self.assertFalse(Float().castable(FloatList()))

        self.assertFalse(Integer().castable(IntegerList(5)))
        self.assertFalse(Float().castable(FloatList(5)))

        self.assertTrue(IntegerList().castable(Integer()))
        self.assertTrue(FloatList().castable(Float()))

        self.assertTrue(IntegerList().castable(IntegerList()))
        self.assertTrue(FloatList().castable(IntegerList()))

        self.assertTrue(IntegerList().castable(IntegerList(5)))
        self.assertTrue(IntegerList(5).castable(IntegerList()))
        self.assertTrue(IntegerList(5).castable(IntegerList(5)))
        self.assertFalse(IntegerList(1).castable(IntegerList(5)))


class GraphTests(TestBase):

    def test_constants(self):

        with Graph() as graph:
            c1 = Constant(value=6)
            c2 = Constant(value=0.5)
            c3 = Constant(value=6)
            outputs(c1, c2, c3)

        pipeline = Compiler().build(graph)
        pipeline.run(1)

        pipeline = None


    def test_serialization(self):
        from .numbers import SampleUnform
        from .flow import Conditional

        with Graph() as graph:
            a = SampleUnform(0, 30)
            b = Constant(value=3)
            c = b + 1
            d = Conditional(a, c, a > 15)
            e = Constant([1, 2, 3, 4])
            outputs(a, c, d, e)

        compare_serialized(graph)
        
    def test_debug(self):

        with Graph() as graph:
            c = Constant(value=6)
            outputs(Debug(c))

        compare_serialized(graph)

    def test_file_location(self):
        from .list import FileList
        from .compiler import Compiler
        from . import write_pipeline, read_pipeline
        import os

        files = ["a.txt", "b.txt", "c.txt"]

        import tempfile

        tmpfd, tmpname = tempfile.mkstemp()

        os.close(tmpfd)

        with Graph() as graph:
            a = FileList(files)
            outputs(a[0])

        write_pipeline(tmpname, Compiler().build(graph))
        pipeline = read_pipeline(tmpname)
        self.assertEqual(str(pipeline.run(1)[0]), os.path.abspath(files[0]))

    def test_output_filter(self):
        from .compiler import Compiler
        from .graph import Output

        with Graph() as graph:
            Output(1, label="a")
            Output(2, label="b")
            Output(3, label="c")

        pipeline = Compiler().build(graph, output=["a", "b"])

        self.assertEqual(len(pipeline.run(1)), 2)

class NumbersTests(TestBase):

    def test_arithmetic(self):

        with Graph() as graph:
            n1 = Constant(value=6)
            n2 = Constant(value=3)
            outputs(n1+n2, n1-n2, n1*n2, n1/n2, n1**n2, n1 % n2, -n1)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 9)
        self.assertEqual(output[1], 3)
        self.assertEqual(output[2], 18)
        self.assertEqual(output[3], 2)
        self.assertEqual(output[4], 216)
        self.assertEqual(output[5], 0)
        self.assertEqual(output[6], -6)

    def test_saturated(self):  
        from pixelpipes.numbers import Saturate
        from pixelpipes.tensor import Convert

        with Graph() as graph:
            n1 = Convert(Constant(value=255), "Char")
            n2 = Convert(Constant(value=2), "Char")
            with Saturate():
                outputs(n1+n2, n1-n2, n1 * n2, n1 / n2)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 255)
        self.assertEqual(output[1], 253)
        self.assertEqual(output[2], 255)
        self.assertEqual(output[3], 127)

    def test_sampling(self):

        from pixelpipes.numbers import SampleUnform, SampleNormal, SampleBinomial, SampleBernoulli

        with Graph() as graph:
            n1 = SampleUnform(0, 30)
            n2 = SampleNormal(0, 1)
            n3 = SampleBinomial(n=10, p=0.5)
            n4 = SampleBernoulli(0.5)
            outputs(n1, n2, n3, n4)

        pipeline = Compiler().build(graph)

        # Evaluate the pipeline multiple times, collect samples and check if they are within the expected range and match the distribution
        for i in range(1, 100):
            sample = pipeline.run(i)
            self.assertGreaterEqual(sample[0], 0)
            self.assertLess(sample[0], 30)
            self.assertGreaterEqual(sample[1], -5)
            self.assertLess(sample[1], 5)
            self.assertGreaterEqual(sample[2], 0)
            self.assertLess(sample[2], 10)
            self.assertTrue(sample[3] in [0, 1])

    def test_trigonometry(self):
        from pixelpipes.numbers import Sin, Cos, Tan, ArcCos, ArcSin, ArcTan

        with Graph() as graph:
            n1 = Constant(value=0.5)
            outputs(Sin(n1), Cos(n1), Tan(n1), ArcCos(n1), ArcSin(n1), ArcTan(n1))

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertAlmostEqual(output[0], np.sin(0.5))
        self.assertAlmostEqual(output[1], np.cos(0.5))
        self.assertAlmostEqual(output[2], np.tan(0.5))

    def test_constant(self):

        with Graph() as graph:
            n0 = Constant(1)
            n1 = Constant(1.5)
            n2 = Constant([0, 1, 2])
            n3 = Constant("test")
            n4 = Constant(True)
            outputs(n0, n1, n2, n3, n4)

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)
        np.testing.assert_array_equal(sample[2], [0, 1, 2])

    def test_conversion(self):

        from pixelpipes.tensor import Convert

        with Graph() as graph:
            n1 = Constant(value=3)
            n2 = Constant(value=3.6)
            n3 = Constant(value=1000)
            outputs(Convert(n1, "Float"), Convert(n2, "Integer"), Convert(n3, "Char"), Convert(n3, "Short"))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 3)
        self.assertEqual(sample[1], 3)
        self.assertEqual(sample[2], 255)
        self.assertEqual(sample[3], 1000)

    def test_truncation(self): 

        with Graph() as graph:
            n1 = Constant(value=3.6)
            outputs(Floor(n1), Round(n1), Ceil(n1))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 3)
        self.assertEqual(sample[1], 4)
        self.assertEqual(sample[2], 4)

    def test_min_max(self):

        from pixelpipes.numbers import Min, Max

        with Graph() as graph:
            n1 = Constant(value=3)
            n2 = Constant(value=4)
            outputs(Min(n1, n2), Max(n1, n2))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 3)
        self.assertEqual(sample[1], 4)

    def test_square_root(self):

        from pixelpipes.numbers import SquareRoot

        with Graph() as graph:
            n1 = Constant(value=9)
            outputs(SquareRoot(n1))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 3)

class ListTests(TestBase):

    def test_list_range(self):
        with Graph() as graph:
            r1 = Range(0, 6, 6, True)
            r2 = Range(0, 5, 10, False)
            outputs(r1, r2)

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(
            sample[0], np.array(range(0, 6, 1), dtype=np.int32))
        np.testing.assert_array_equal(sample[1], np.array(
            range(0, 10, 1), dtype=np.float32) / 2)

    def test_list_table(self):

        with Graph() as graph:
            n = Table([[0, 1, 2], [3, 4, 5]])
            outputs(n[0])

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], [[0], [1], [2]])

    def test_list_arithmetic(self):

        with Graph() as graph:
            n1 = Constant([1, 1, 1])
            n2 = Constant([2.0, 2.0, 2.0])
            n3 = Constant([4, 6, 8])

            outputs(n2 + n1, n1 + 5, 1.5 - n2, n1 - n2, n1 * n2, n3 / n2)

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], [3, 3, 3])
        np.testing.assert_array_equal(sample[1], [6, 6, 6])
        np.testing.assert_array_equal(sample[2], [-0.5, -0.5, -0.5])
        np.testing.assert_array_equal(sample[3], [-1, -1, -1])
        np.testing.assert_array_equal(sample[4], [2, 2, 2])
        np.testing.assert_array_equal(sample[5], [2, 3, 4])

    def test_list_comparison(self):

        with Graph() as graph:
            n1 = Constant([0, 1, 1, 1])
            n2 = Constant([2, 0, 2, 1])
            outputs(n1 > n2, n1 >= 1, n2 <= 1.5, n1 != n2, n1 == n2, n2 < 2)

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], [False, True, False, False])
        np.testing.assert_array_equal(sample[1], [False, True, True, True])
        np.testing.assert_array_equal(sample[2], [False, True, False, True])
        np.testing.assert_array_equal(sample[3], [True, True, True, False])
        np.testing.assert_array_equal(sample[4], [False, False, False, True])
        np.testing.assert_array_equal(sample[5], [False, True, False, True])

    def test_list_logical(self):

        with Graph() as graph:
            n1 = Constant([True, False, True])
            n2 = Constant([False, True, True])

            outputs(n1 & n2, n1 | n2, ~n2)

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)

        np.testing.assert_array_equal(sample[0], [False, False, True])
        np.testing.assert_array_equal(sample[1], [True, True, True])
        np.testing.assert_array_equal(sample[2], [True, False, False])

    def test_list_arrays(self):

        a1 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        a2 = np.random.rand(15, 15).astype(np.float32)
        a3 = np.random.randint(0, 1000, (20, 20), dtype=np.uint16)

        with Graph() as graph:
            n1 = Constant([a1, a2, a3])
            outputs(n1[0], n1[1], n1[2])

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0].shape, a1.shape)
        self.assertEqual(sample[1].shape, a2.shape)
        self.assertEqual(sample[2].shape, a3.shape)

    def test_list_build(self):
        from pixelpipes.list import MakeList

        with Graph() as graph:
            n1 = Constant(1)
            n2 = Constant(2.3)
            n3 = Constant([1, 2, 3])
            l = MakeList(inputs=[n1, n2, n3])
            outputs(l[0], l[1], l[2])

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[1], 2.3)
        np.testing.assert_array_equal(sample[2], [1, 2, 3])
    
    def test_list_permutation(self):

        from pixelpipes.list import Permute, Permutation

        ls = [10, 20, 30, 40, 50]

        with Graph() as graph:
            l = Constant(ls)
            outputs(Permutation(10), Permute(l))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(len(sample[0]), 10)
        self.assertTrue(len(np.unique(sample[0].squeeze())) == 10)
        self.assertEqual(len(sample[1]), 5)
        self.assertTrue(np.all(np.sort(sample[1].squeeze()) == ls))

    def test_list_concatenation(self):
        from .list import Concatenate

        with Graph() as graph:
            n1 = Constant([1, 2, 3])
            n2 = Constant([4, 5, 6])
            outputs(Concatenate([n1, n2]))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.compare_arrays(sample[0], [1, 2, 3, 4, 5, 6])

    def test_list_length(self):
        from .list import Length

        with Graph() as graph:
            n1 = Constant([1, 2, 3])
            outputs(Length(n1))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.assertEqual(sample[0], 3)

    def test_list_sublist(self):
        from .list import Sublist

        with Graph() as graph:
            n1 = Constant([1, 2, 3, 4, 5, 6])
            outputs(Sublist(n1, 2, 3))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.compare_arrays(sample[0], [3, 4])

    def test_list_repeat(self):
        from .list import Repeat

        with Graph() as graph:
            n1 = Constant(1)
            outputs(Repeat(n1, 3))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.compare_arrays(sample[0], [1, 1, 1])

    def test_list_remap(self):
        from .list import Remap

        with Graph() as graph:
            n1 = Constant([1, 2, 3])
            outputs(Remap(n1, Constant([2, 1, 0])))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.compare_arrays(sample[0], [3, 2, 1])

    def test_list_filter(self):
        from .list import Filter

        with Graph() as graph:
            n1 = Constant([1, 2, 3, 4, 5, 6])
            outputs(Filter(n1, Constant([True, False, True, False, True, False])))

        pipeline = Compiler().build(graph)
        sample = pipeline.run(1)

        self.compare_arrays(sample[0], [1, 3, 5])

class FlowTests(TestBase):

    def test_conditional_simple(self):

        with Graph() as graph:
            a = Round(SampleUnform(0, 30))
            b = Constant(value=2)
            d = Conditional(a, b, a > 15)
            outputs(a, b, d)

        pipeline = Compiler().build(graph)

        for i in range(1, 100):
            a = pipeline.run(i)
            self.assertEqual(a[0] if a[0] > 15 else a[1], a[2])

    def test_conditional_multiple(self):

        with Graph() as graph:
            c1 = Round(Floor(SampleIndex() / 4) % 2)
            c2 = Round(Floor(SampleIndex() / 2) % 2)
            c3 = (Round(SampleIndex() % 2))
            n1 = Conditional(true=1, false=0, condition=c1)
            n2 = Conditional(true=(n1*2)+1, false=n1*2, condition=c2)
            n3 = Conditional(true=(n2*2)+1, false=n2*2, condition=c3)
            outputs(n3)

        pipeline = Compiler().build(graph)
        for i in range(8):
            output = pipeline.run(i)
            self.assertEqual(output[0], i)

    def test_conditional_optimization(self):
        from .flow import Switch

        with Graph() as graph:
            a = Constant(value=20)
            b = Constant(value=30)
            c = a + b
            d = Switch(inputs=[a, b, c], weights=[0.2, 0.2, 0.9])
            b = Constant(value=4)
            outputs(Switch(inputs=[d, b, a - b], weights=[0.5, 0.5, 0.5]))

        pipeline1 = Compiler().build(graph)
        pipeline2 = Compiler().build(graph, optimize=False)

        self.assertNotEqual(len(pipeline1), len(pipeline2))

        for i in range(1, 100):
            a = pipeline1.run(i)
            b = pipeline2.run(i)
            self.assertEqual(a[0], b[0])

class TestTensor(TestBase):

    def test_tensor_add(self):

        test = np.random.randint(1, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test)
            n1 = Constant(value=0)
            n2 = Constant(value=1)       
            o0 = n0 + n0
            o1 = n0 + n1
            o2 = n1 + n0
            o3 = n0 + n2
            o4 = n2 + n0     
            outputs(o0, o1, o2, o3, o4)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        temp = test.astype(dtype=np.uint16, copy=True)
        temp = temp + temp
        #temp[temp>255] = 255
        temp = temp.astype(dtype=np.uint8, copy=False)
        np.testing.assert_array_equal(output[0], temp)
        np.testing.assert_array_equal(output[1], test)
        np.testing.assert_array_equal(output[2], test)
        np.testing.assert_array_equal(output[3], test + 1)
        np.testing.assert_array_equal(output[4], test + 1)

    def test_tensor_subtract(self):

        test0 = np.random.randint(1, 255, (256,256), dtype=np.uint8)
        test1 = np.random.randn(256,256).astype(dtype=np.float32)

        with Graph() as graph:
            n0 = Constant(test0)
            n1 = Constant(test1)
            c1 = Constant(1)    
            o0 = Debug(n0 - n0)
            o1 = n0 - c1
            o2 = c1 - n0
            o3 = n1 - n1
            o4 = n1 - c1
            o5 = c1 - n1
            outputs(o0, o1, o2, o3, o4, o5)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], np.zeros((256,256), dtype=np.uint8))
        # saturated ops
#        np.testing.assert_array_equal(output[1], np.clip((test0.astype(np.float32) - 1), 0, 255).astype(np.uint8))
#        np.testing.assert_array_equal(output[2], np.clip((1 - test0.astype(np.float32)), 0, 255).astype(np.uint8))

        np.testing.assert_array_equal(output[3], np.zeros((256,256), dtype=np.float32))
        np.testing.assert_array_equal(output[4], (test1.astype(np.float32) - 1))
        np.testing.assert_array_equal(output[5], 1 - test1.astype(np.float32))

    def test_tensor_multiply(self):

        test = np.random.randint(1, 255, (256,256), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test)
            n1 = Constant(0)
            n2 = Constant(1)
            n3 = Constant(255)    
            o0 = n0 * n0
            o1 = n0 * n1
            o2 = n1 * n0
            o3 = n0 * n2
            o4 = n2 * n0
            o5 = n0 * n3
            o6 = n3 * n0
            outputs(o0, o1, o2, o3, o4, o5, o6)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[1], test * 0)
        np.testing.assert_array_equal(output[2], test * 0)
        np.testing.assert_array_equal(output[3], test)
        np.testing.assert_array_equal(output[4], test)

        # saturated ops
        #np.testing.assert_array_equal(output[5], np.ones((256,256), dtype=np.uint8) * 255)
        #np.testing.assert_array_equal(output[6], np.ones((256,256), dtype=np.uint8) * 255)


    def test_tensor_list(self):

        test = np.random.rand(10,20,20).astype(np.float32)

        with Graph() as graph:
            n = Constant(test)
            outputs(n[5], n[8], n[2] + n[3])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], np.squeeze(test[5, :, :]))
        np.testing.assert_array_equal(output[1], np.squeeze(test[8, :, :]))
        np.testing.assert_array_equal(output[2], np.squeeze(test[2, :, :] + test[3, :, :]))

    def test_tensor_stack(self):

        from pixelpipes.tensor import Stack

        test_image = np.random.randint(0, 255, (4,20,40), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image[0, ...])
            n1 = Constant(test_image[1, ...])
            n2 = Constant(test_image[2, ...])
            n4 = Constant(test_image[3, ...])
            o0 = Stack(inputs=[n0, n1, n2, n4])
            outputs(o0, n0, o0[1], n1)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[1], test_image[0, ...])
        np.testing.assert_array_equal(output[2], output[3])
        np.testing.assert_array_equal(output[0], test_image)

    def test_tesor_reshape(self):

        from pixelpipes.tensor import Reshape

        test_image = np.random.randint(0, 255, (4,20,40), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Reshape(n0, Constant([20, 160]))
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], test_image.reshape(20, 160))

    def test_tensor_transpose(self):

        from pixelpipes.tensor import Transpose
        
        test_image = np.random.randint(0, 255, (4,20,40), dtype=np.uint8)
        test_image2 = np.random.randint(0, 255, (4,2), dtype=np.uint8)
        
        with Graph() as graph:
            o0 = Transpose(Constant(test_image), Constant([1, 2, 0]))
            o1 = Transpose(Constant(test_image2), Constant([1, 0]))
            outputs(o0, o1)
            
        pipeline = Compiler().build(graph)
        output = pipeline.run(1)
        
        np.testing.assert_array_equal(output[0], np.transpose(test_image, [1, 2, 0]))

    def test_tensor_squeeze(self):
            
            from pixelpipes.tensor import Squeeze
    
            test_image = np.random.randint(0, 255, (1,20,40,1), dtype=np.uint8)
    
            with Graph() as graph:
                n0 = Constant(test_image)
                o0 = Squeeze(n0)
                outputs(o0)
    
            pipeline = Compiler().build(graph)
            output = pipeline.run(1)
    
            np.testing.assert_array_equal(output[0], test_image.squeeze())

    def test_tensor_unsqueeze(self):
        
        from pixelpipes.tensor import Unsqueeze

        test_image = np.random.randint(0, 255, (20,40), dtype=np.uint8)

        with Graph() as graph:
            n0 = Constant(test_image)
            o0 = Unsqueeze(Unsqueeze(n0, Constant(0)), 3)
            outputs(o0)

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        np.testing.assert_array_equal(output[0], np.expand_dims(test_image, (0, 3)))

if __name__ == "__main__":
    # Special entrypoint for running tests and determining operation coverage afterwards
    from collections import Counter
    from pixelpipes import list_operations

    ignore_operations = ["debug"]

    # Monkey patch the compiler to track operations
    Compiler._operations = []
    Compiler._compile = Compiler.compile

    def compile(self, *args, **kwargs):
        operations = self._compile(*args, **kwargs)
        Compiler._operations.extend([o.name for o in operations])
        return operations
    
    Compiler.compile = compile
    
    unittest.main(exit=False, module=None)

    # Count appearances of unique operations

    used_operations = Counter(Compiler._operations)
    supported_operations = list_operations()

    unused = []
    for op in supported_operations:
        if op not in used_operations and op not in ignore_operations:
            unused.append(op)

    if unused:
        print("\n{}/{} untested operations:".format(len(unused), len(supported_operations)), ",".join(unused))

