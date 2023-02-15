import unittest

import numpy as np
from attributee import Integer

from pixelpipes.resource.list import ResourceListSource, SegmentedResourceListSource

from ..graph import Constant, SampleIndex, outputs
from ..graph import Graph
from ..compiler import Compiler
from . import AppendField, ConditionalResource, MakeResource
from ..tests import TestBase

class TestResourceList(ResourceListSource):

    length = Integer(default=10)

    def load(self):

        import random

        return {
            "a": random.choices(range(0, 5), k=self.length),
            "b": random.choices((True, False), k=self.length),
        }

class TestSegmentedResourceList(SegmentedResourceListSource):

    length = Integer(default=10)

    def load(self):

        return {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [True, False, True, True, False, True],
            "_segments": [4, 2]
        }


class TestResource(TestBase):    
    
    def test_create(self):

        with Graph() as graph:
            a = Constant(1)
            b = Constant(4.5)
            c = Constant("foo")
            resource = MakeResource(a = a, b = b, c = c)
            resource = AppendField(resource, "d", 6)
            outputs(resource["a"], resource["b"], resource["c"])
        
        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4.5)
        self.assertEqual(output[2], "foo")

    def test_conditional(self):

        with Graph() as graph:
            a = MakeResource(a = Constant(1), b = Constant(2))
            b = MakeResource(a = Constant(1.4), b = Constant(1.5))
            resource = ConditionalResource(a, b, SampleIndex() % 2 == 0)
            outputs(resource["a"], resource["b"])

        pipeline = Compiler().build(graph)

        output = pipeline.run(1)
        self.assertEqual(output[0], 1.4)
        self.assertEqual(output[1], 1.5)

        output = pipeline.run(2)
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 2)


class TestList(TestBase): 

    def test_resource_list(self):

        with Graph() as graph:
            resources = TestResourceList()
            outputs(resources[0]["a"])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0].dtype, np.int32)


    def test_segmented_resource_list(self):

        with Graph() as graph:
            resources = TestSegmentedResourceList()
            outputs(resources[0]["a"])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertEqual(output[0], 1)


    def test_resource_list_conditional(self):

        with Graph() as graph:
            r1 = TestResourceList(length=10)
            r2 = TestResourceList(length=5)
            resources = ConditionalResource(r1, r2, SampleIndex() % 2)
            outputs(resources[0]["a"])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

class TestLoading(TestBase):    
    
    def test_image_directory(self):
        
        import os
        import numpy as np

        import pixelpipes
        from pixelpipes.resource.loading import ImageDirectory

        example_images = os.path.join(os.path.dirname(os.path.dirname(pixelpipes.__file__)), "examples", "images")

        if not os.path.isdir(example_images):
            return

        with Graph() as graph:
            image_directory = ImageDirectory(path=example_images)
            outputs(image_directory[0]["image"])

        pipeline = Compiler().build(graph)
        output = pipeline.run(1)

        self.assertIsInstance(output[0], np.ndarray)