import numpy as np

from pixelpipes import Engine, LegacyEngine, ParallelEngine, pypixelpipes
from pixelpipes.compiler import Compiler
from pixelpipes.graph import Constant, Graph, SampleIndex, outputs

from tests._base import TestBase


class TestEngine(TestBase):

    def setUp(self):
        super().setUp()
        with Graph() as graph:
            outputs(
                Constant(np.arange(4, dtype=np.float32)),
                SampleIndex(),
            )
        self.pipeline = Compiler().build(graph)

    def test_next_batch_is_native_and_ordered(self):
        engine = Engine(self.pipeline, batch_size=4, workers=1,
                        max_workers=4, queue_size=2, first_index=3)

        first = engine.next_batch()
        second = engine.next_batch()

        self.assertEqual(first[0].shape, (4, 4))
        self.assertEqual(first[1].shape, (4, 1))
        np.testing.assert_array_equal(first[1].reshape(-1), np.arange(3, 7))
        np.testing.assert_array_equal(second[1].reshape(-1), np.arange(7, 11))

        for batch_value in first[0]:
            np.testing.assert_array_equal(batch_value, np.arange(4, dtype=np.float32))

    def test_native_binding_only_returns_complete_batches(self):
        engine = pypixelpipes.Engine(self.pipeline._pipeline, 3, 1, 2, 1, 5)
        batch = engine.next_batch()

        self.assertEqual(tuple(batch[0].shape), (3, 4))
        self.assertEqual(tuple(batch[1].shape), (3, 1))
        np.testing.assert_array_equal(np.asarray(batch[1]).reshape(-1), np.arange(5, 8))

    def test_engine_is_ordered_and_copies_tensor_slices(self):
        engine = Engine(self.pipeline, batch_size=4, workers=2,
                        max_workers=2, copy_workers=2,
                        queue_size=2, first_index=3)

        first = engine.next_batch()
        second = engine.next_batch()

        self.assertEqual(first[0].shape, (4, 4))
        np.testing.assert_array_equal(first[1].reshape(-1), np.arange(3, 7))
        np.testing.assert_array_equal(second[1].reshape(-1), np.arange(7, 11))
        for batch_value in first[0]:
            np.testing.assert_array_equal(batch_value, np.arange(4, dtype=np.float32))

    def test_legacy_engine_remains_available(self):
        engine = LegacyEngine(self.pipeline, batch_size=4, workers=1,
                              max_workers=1, queue_size=1, first_index=3)
        batch = engine.next_batch()

        self.assertEqual(batch[0].shape, (4, 4))
        np.testing.assert_array_equal(batch[1].reshape(-1), np.arange(3, 7))

    def test_parallel_engine_name_is_compatible_alias(self):
        self.assertIs(ParallelEngine, Engine)

    def test_batch_generation_errors_are_reported_by_blocking_call(self):
        native_pipeline = pypixelpipes.Pipeline()
        engine = pypixelpipes.Engine(native_pipeline, 2)

        with self.assertRaises(pypixelpipes.PipelineException):
            engine.next_batch()
