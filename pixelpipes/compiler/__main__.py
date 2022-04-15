import sys

from attributee import Attributee
from attributee.primitives import String, Boolean
from attributee.object import Callable
from attributee.io import Entrypoint
from pixelpipes import write_pipeline


from pixelpipes.graph import ValidationException

class Compiler(Attributee, Entrypoint):

    graph = Callable()
    debug = Boolean(default=False)
    predictive = Boolean(default=True)
    compress = Boolean(default=True)
    fixedout = Boolean(default=False)
    output = String(default=None)
    save = String(default="pipeline.pxpt")

    def __call__(self):

        from pixelpipes.compiler import Compiler

        compiler = Compiler(fixedout=self.fixedout, debug=self.debug, predictive=self.predictive)

        try:
            operations = compiler.compile(self.graph(), output=self.output)

            print("Compiled pipeline of %d operations, writing to %s" % (len(operations), self.save))

            write_pipeline(self.save, operations, self.compress)

            print("Done.")

        except ValidationException as ve:
            print(ve)
            ve.print_nodestack()
            sys.exit(-1)
    
if __name__ == "__main__":

    main = Compiler.parse()
    main()
