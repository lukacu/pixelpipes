
from pixelpipes.graph import Graph, Constant, Debug, outputs, Output
from pixelpipes.compiler import Compiler

def refcouter():
    from pixelpipes import pypixelpipes
    if not hasattr(pypixelpipes, "_refcount"):
        raise RuntimeError("Reference counting is not enabled")
    return pypixelpipes._refcount

def test():
    
    counter = refcouter()

    print("Reference count: ", counter())
    with Graph() as graph:
        outputs(Constant(1))
    print("Reference count: ", counter())

    pipeline = Compiler().build(graph)

    print("Reference count: ", counter())

    for i in range(10):
        print("Iteration: ", i, "Reference count: ", counter())
        pipeline.run(i)
        
if __name__ == '__main__':
    test()
    