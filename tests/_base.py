import os
import unittest

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def compare_serialized(graph, samples=100):
    import os
    import tempfile
    import numpy.testing as npt

    from pixelpipes import read_pipeline, write_pipeline
    from pixelpipes.compiler import Compiler

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
            print(
                "\n **** Remaining objects:",
                leftobj,
                "(total:",
                pypixelpipes._refcount(),
                ")**** ",
            )
        return super().tearDown()

    def compare_arrays(self, a, b):
        """Compare arrays while ignoring singleton dimensions at either end."""
        a = np.squeeze(a)
        b = np.squeeze(b)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)
