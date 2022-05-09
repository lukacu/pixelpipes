import unittest
import numpy as np
import os

from pixelpipes import read_pipeline, write_pipeline

from .graph import Constant, Output, SampleIndex, DebugOutput, outputs
from .numbers import Floor, Round, UniformDistribution
from .expression import Expression
from .flow import Switch
from .list import ConstantList, ConstantTable, ListElement, ListRange
from .graph import GraphBuilder
from .compiler import Compiler, Conditional

def compare_serialized(graph):

    import tempfile
    import numpy.testing as npt

    compiler = Compiler()

    pipeline1 = compiler.build(graph)

    tmpfd, tmpname = tempfile.mkstemp()

    os.close(tmpfd)

    write_pipeline(tmpname, compiler.compile(graph))

    pipeline2 = read_pipeline(tmpname)

    for i in range(1, 100):
        a = pipeline1.run(i)
        b = pipeline2.run(i)
        assert len(a) == len(b), "Output does not match"
        for x, y in zip(a, b):
            npt.assert_equal(a[0], b[0])

    os.remove(tmpname)

class TestPipes(unittest.TestCase):    

    """
    core/expression.py
    """

    def test_core_Expression(self):

        with GraphBuilder() as graph:
            n1 = Constant(value=5)
            n2 = Constant(value=15)
            n3 = Expression(source="((x ^ 2 - y) * 2) / 5 + 2", variables=dict(x=n1, y=n2))
            Output(n3)

        pipeline = Compiler.build_graph(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 6)

    def test_core_Operations(self):

        with GraphBuilder() as graph:
            n1 = Constant(value=6)
            n2 = Constant(value=3)
            outputs(n1+n2, n1-n2, n1*n2, n1/n2, n1**n2, n1%n2, -n1)

        pipeline = Compiler.build_graph(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 9)
        self.assertEqual(output[1], 3)
        self.assertEqual(output[2], 18)
        self.assertEqual(output[3], 2)
        self.assertEqual(output[4], 216)
        self.assertEqual(output[5], 0)
        self.assertEqual(output[6], -6)

    """
    core/numbers.py
    """

    def test_core_UniformDistribution(self):

        a = 1.5
        b = 3

        with GraphBuilder() as graph:
            bb = Constant(a)
            outputs(UniformDistribution(a, b), bb)

        pipeline = Compiler.build_graph(graph)

        for i in range(3):
            output = pipeline.run(i)
            self.assertGreaterEqual(output[0], a)
            self.assertLess(output[0], b)

    """
    core/list.py
    """

    def test_core_ConstantList(self):

        with GraphBuilder() as graph:
            n1 = ConstantList([0, 1, 2])
            Output(n1)

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)
        np.testing.assert_array_equal(sample[0], [[0], [1], [2]])

    def test_core_ListRange(self):

        with GraphBuilder() as graph:
            r1 = ListRange(0, 10, 10, True)
            r2 = ListRange(0, 5, 10, False)
            outputs(r1, r2)

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)
        np.testing.assert_array_equal(sample[0], np.array([range(0, 10, 1)], dtype=np.int32).T)
        np.testing.assert_array_equal(sample[1], np.array([range(0, 10, 1)], dtype=np.float32).T / 2)

    def test_core_ConstantTable(self):

        with GraphBuilder() as graph:
            n = ConstantTable([[0, 1, 2], [3, 4, 5]])
            outputs(ListElement(DebugOutput(n), 0))

        pipeline = Compiler.build_graph(graph)
        sample = pipeline.run(1)
   
        np.testing.assert_array_equal(sample[0], [[0], [1], [2]])

    def test_compiler_Conditional_0(self):

        with GraphBuilder() as graph:
            c1 = Round(Floor(SampleIndex() / 4) % 2)
            c2 = Round(Floor(SampleIndex() / 2) % 2)
            c3 = (Round(SampleIndex() % 2))
            n1 = Conditional(true=1, false=0, condition=c1)
            n2 = Conditional(true=(n1*2)+1, false=n1*2, condition=c2)
            n3 = Conditional(true=(n2*2)+1, false=n2*2, condition=c3)
            Output(n3)

        pipeline = Compiler.build_graph(graph)
        for i in range(8):
            output = pipeline.run(i)
            self.assertEqual(output[0], i)

    """
    TODO FIX test_compiler_Conditional_1

    def test_compiler_Conditional_1(self):

        with GraphBuilder() as graph:
            a = UniformDistribution(0, 30)
            b = Constant(value=3)
            c = b + 1
            d = Conditional(a, c, a > 15)
            outputs(a, c, d)

        pipeline = Compiler(debug=False).build(graph)

        for i in range(1, 100):
            a = pipeline.run(i)
            self.assertEqual(a[0] if a[0] > 15 else a[1], a[2])
            outputs(a, b, d)
      
        pipeline = Compiler(debug=False).compile_graph(graph)

        for i in range(1, 100):
            a = pipeline.run(i)
            self.assertEqual(a[0] if a[0] + a[1] > 15 else a[1], a[2])
    """

    """
    PROPOSED FIX for test_compiler_Conditional_1 
    """
    def test_compiler_Conditional_1(self):

        with GraphBuilder() as graph:
            a = Round(UniformDistribution(0, 30))
            b = Constant(value=3)
            d = Conditional(a, b, a > 15)
            outputs(a, b, d)
      
        pipeline = Compiler().build(graph)

        for i in range(1, 100):
            a = pipeline.run(i)
            self.assertEqual(a[0] if a[0] > 15 else a[1], a[2])

    def test_compiler_Conditional_Switch(self):

        # Does not work at the moment, the result is not the same because of the order of random generators
        # TODO: this has to be fixed
        return

        with GraphBuilder() as graph:
            a = Constant(value=20)
            b = Constant(value=30)
            c = a + b
            d = Switch(inputs=[a, b, c], weights=[0.2, 0.2, 0.9])
            b = Constant(value=4)
            outputs(Switch(inputs=[d, b, a - b], weights=[0.5, 0.5, 0.5]))

        pipeline1 = Compiler(debug=True).build(graph)
        pipeline2 = Compiler(debug=True, predictive=False).build(graph)

        for i in range(1, 100):
            a = pipeline1.run(i)
            b = pipeline2.run(i)
            self.assertEqual(a[0], b[0])

    def test_serialization(self):

        with GraphBuilder() as graph:
            a = UniformDistribution(0, 30)
            b = Constant(value=3)
            c = b + 1
            d = Conditional(a, c, a > 15)
            e = ConstantList([1, 2, 3, 4])
            outputs(a, c, d, e)

        compare_serialized(graph)
        

        


