from __future__ import absolute_import

__version__ = "0.0.3"

import os
import logging

import pixelpipes.types as types

_logger = logging.getLogger(__name__)

def include_directories():
    root = os.path.dirname(__file__)
    return [os.path.join(root, "core"), os.path.join(root, "geometry"), os.path.join(root, "image")]
class Pipeline(object):
    """Wrapper for the C++ pipeline object, includes metadata
    """

    def __init__(self, pipeline, groups, types, operations):
        self._pipeline = pipeline
        self._groups = groups
        self._types = types
        self._operations = operations

    def __len__(self):
        return len(self._operations)

    def run(self, index):
        return self._pipeline.run(index)

    def output_size(self):
        return len(self._groups)

    def output_group(self, i):
        return self._groups[i]

    def output_type(self, i):
        return self._types[i]

    def outputs(self):
        return zip(self._groups, self._types)

    def stats(self):
        stats = self._pipeline.operation_time()
        for k, v in zip(self._operations, stats):
            print("%s: %.3f ms" % (k, v))
        