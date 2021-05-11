
raise NotImplementedError("This module is still work in progress, do not import it")

_CACHE_STACK = []

class Cache(object):

    def __init__(self):
        self._data = {}

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __enter__(self):
        _CACHE_STACK.append(self)

    def __exit__(self, a, b, c):
        _CACHE_STACK.pop()

def get_cache(identifier):
    for cache in reversed(_CACHE_STACK):
        if identifier in cache:
            return cache[identifier]
    return None

def put_cache(identifier, value):
    if not _CACHE_STACK:
        return
    _CACHE_STACK[-1][identifier] = value

class BlockingManager(object):

    def __init__(self, workers=1):
        self._engine = engine.Engine(workers)


class AsyncEngine(object):

    def __init__(self, workers=1):
        self._engine = engine.Engine(workers)




