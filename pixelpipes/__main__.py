import sys
import os
import time
import argparse

from attributee.object import Callable, import_class
from pixelpipes import Pipeline, write_pipeline, read_pipeline
from pixelpipes.graph import Graph, ValidationException

def _pipeline(graph, debug=False, optimize=True):
    """Compiles a graph to a pipeline object
    
    Args:
        graph (str): Graph to compile
        debug (bool, optional): Enable debug mode. Defaults to False.
    
    Returns: Pipeline object
    """
    from pixelpipes.compiler import Compiler
    from pixelpipes.utilities import change_environment
    graph = import_class(graph)

    if isinstance(graph, Callable):
        raise ValueError("Graph must be a callable")

    with change_environment(PIXELPIPES_PIPELINE_OPTIMIZE=str(optimize)):
        graph = graph()

        if isinstance(graph, Graph):
            compiler = Compiler(debug=debug)
            return compiler.build(graph)
        elif isinstance(graph, Pipeline):
            return graph

def main():
    """Main entry point for CLI tool"""
    
    parser = argparse.ArgumentParser(prog="pixelpipes", description='Pixelpipes CLI tool')
    parser.add_argument("-d", "--debug", help="Enable debug mode", action="store_true")
    
    subparsers = parser.add_subparsers(help='commands', dest='action', title="Commands")
    
    # Compiler subcommand
    compiler = subparsers.add_parser("compile", help="Compile a pipeline")
    compiler.add_argument("graph", help="Graph to compile")
    compiler.add_argument("--relocatable", help="Enable relocatable mode", action="store_true")
    compiler.add_argument("--compress", help="Enable compression", action="store_true")
    compiler.add_argument("--output", help="Output file")
    compiler.add_argument("--fixedout", help="Fixed output size", action="store_true")
    compiler.add_argument("--save", help="Save compiled pipeline to file", required=True)
    
    # Benchmark subcommand
    benchmark = subparsers.add_parser("benchmark", help="Benchmark a pipeline")
    benchmark.add_argument("graph", help="Graph to benchmark (can also be a serialized pipeline)")
    benchmark.add_argument("--batch", default=1, help="Batch size to use", type=int)
    benchmark.add_argument("--workers", default=1, help="Number of workers to run", type=int)
    benchmark.add_argument("--iterations", help="Number of iterations to run", type=int, default=0)

    args = parser.parse_args()
    
    if args.action == "compile":
        try:
            pipeline = _pipeline(args.graph, debug=args.debug, optimize=False)

            print("Compiled pipeline of %d operations, writing to %s" % (len(pipeline), args.save))

            write_pipeline(args.save, pipeline, args.compress)

            print("Done.")

        except ValidationException as ve:
            print(ve)
            ve.print_nodestack()
            sys.exit(-1)
    elif args.action == "benchmark":
        try:
            from pixelpipes.sink import PipelineDataLoader
            
            if os.path.isfile(args.graph):
                pipeline = read_pipeline(args.graph)
            else:
                pipeline = _pipeline(args.graph)

            loader = PipelineDataLoader(pipeline, args.batch, workers=args.workers)
   
            i = 0

            try:
                start = time.time()
                for i, _ in enumerate(loader):
                    print("Batch %d" % i)
                    elapsed = time.time() - start

                    if args.iterations > 0 and i >= args.iterations:
                        break

            except KeyboardInterrupt:
                pass

            if i > 0:
                print("=====================================================")
                print("Elapsed time %.3fs" % elapsed)
                print("Total batches %d, Speed: %.3fs per batch" % (i, elapsed / i))
                print("Total samples %d, Speed: %.3fs per sample" % (i * args.batch, elapsed / (i * args.batch)))

        except ValidationException as ve:
            print(ve)
            ve.print_nodestack()
            sys.exit(-1)
    else:
        parser.print_help()
        sys.exit(-1)
    
if __name__ == "__main__":
    main()
