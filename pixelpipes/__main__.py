import sys

from attributee import Attributee
from attributee.primitives import String, Boolean
from attributee.object import Callable
from attributee.io import Entrypoint
from pixelpipes import Pipeline, write_pipeline

from pixelpipes.graph import Graph, ValidationException

class Compiler(Attributee, Entrypoint):

    graph = Callable()
    debug = Boolean(default=False)
    relocatable = Boolean(default=True)
    compress = Boolean(default=True)
    fixedout = Boolean(default=False)
    output = String(default=None)
    save = String(default="pipeline.pxpt")

    def __call__(self):

        from pixelpipes.compiler import Compiler

        try:
            graph = self.graph()

            if isinstance(graph, Graph):
                compiler = Compiler(fixedout=self.fixedout, debug=self.debug)
                pipeline = compiler.build(graph, output=self.output)
            elif isinstance(graph, Pipeline):
                pipeline = graph

            print("Compiled pipeline of %d operations, writing to %s" % (len(pipeline), self.save))

            write_pipeline(self.save, pipeline, self.compress)

            print("Done.")

        except ValidationException as ve:
            print(ve)
            ve.print_nodestack()
            sys.exit(-1)
    
if __name__ == "__main__":

    main = Compiler.parse()
    main()
