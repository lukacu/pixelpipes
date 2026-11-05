

import numpy as np

from pixelpipes.compiler import Compiler
from pixelpipes.graph import Graph, Constant, SampleIndex, outputs
from pixelpipes.sink import PipelineDataLoader

from tests._base import TestBase

def _run_sink_text(constructor, batch_size, workers):

    with Graph() as graph:
        outputs(Constant(np.zeros((10, 10), dtype=np.float32)))
        outputs(Constant(1), Constant([10, 20, 30]))

    batch_size = 10

    loader = constructor(Compiler().build(graph), batch=batch_size, workers=workers)

    for _ in zip(loader, range(1000)):
        pass

class TestSinks(TestBase):

    def test_sink_single_worker(self):
        
        _run_sink_text(PipelineDataLoader, 32, 1)

    def test_sink_multi_worker(self):
        
        _run_sink_text(PipelineDataLoader, 32, 50)

    def test_sink_multi_worker_shared_pipeline(self):
        from pixelpipes.flow import Conditional

        values = np.arange(32, dtype=np.float32)

        with Graph() as graph:
            constant = Constant(values)
            index = SampleIndex()
            outputs(constant, index, constant + index)
            outputs(Conditional(constant + index, constant - index, index % 2 == 0))

        loader = PipelineDataLoader(Compiler().build(graph), batch=16, workers=16)

        # Exercise repeated concurrent calls after the first batch has raced to
        # initialize the shared constant cache.
        for batch_index, batch in zip(range(32), loader):
            sample_indices = np.arange(batch_index * 16 + 1, batch_index * 16 + 17)
            np.testing.assert_array_equal(batch[0], np.broadcast_to(values, (16, values.size)))
            np.testing.assert_array_equal(batch[1].reshape(-1), sample_indices)
            np.testing.assert_array_equal(batch[2], values[None, :] + sample_indices[:, None])
            expected_conditional = np.where(
                (sample_indices % 2 == 0)[:, None],
                values[None, :] + sample_indices[:, None],
                values[None, :] - sample_indices[:, None],
            )
            np.testing.assert_array_equal(batch[3], expected_conditional)

    def test_sink_pytorch(self):

        try:
            from pixelpipes.sink import TorchDataLoader
            _run_sink_text(TorchDataLoader, 32, 1)
        except ImportError:
            return

        

  
