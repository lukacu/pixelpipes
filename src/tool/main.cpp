
//https://github.com/p-ranav/argparse

#include <string>
#include <iostream>
#include <chrono>
/**
 * A CLI tool to for various operations on pipelines, benchmarking, info, etc.
*/

// Argparse


#include <pixelpipes/serialization.hpp>
#include <pixelpipes/module.hpp>

#include "cmdparse.hpp"

using namespace pixelpipes;

int main(int argc, const char *argv[])
{

    cli::Parser parser(argc, argv);
    //parser.set_optional<bool>("h", "help", false, "Print this help message.");
	//parser.set_optional<bool>("o", "optimize", false, "Optimize the pipeline before running.");
    parser.set_optional<bool>("i", "info", false, "Display the metadata of the pipeline.");


    parser.set_optional<int>("r", "run", 0, "Run pipeline for the given number of samples.");
    //parser.set_optional<int>("w", "workers", 1, "Number of workers.");

	parser.set_required<std::string>("f", "file", "Pipeline file to load.");
	parser.run_and_exit_if_error();

    std::string filename = parser.get<std::string>("f");

    try
        {
            
            Pipeline p = read_pipeline(filename);

            if (parser.get<bool>("i"))
            {
                std::cout << "Pipeline: " << filename << std::endl;
                std::cout << "Metadata:" << std::endl;

                for (auto k : p.metadata().keys()) 
                {
                    std::cout << " * " << k << ": " << p.metadata().get(k) << std::endl;
                }

                std::cout << "Outputs:" << std::endl;

                for (auto k : p.get_labels()) 
                {
                    std::cout << k << std::endl;
                }
            }

            if (parser.get<int>("r") > 0)
            {
                unsigned long i = 1;
                unsigned long limit = parser.get<int>("r");

                // Benchmark time 

                auto start = std::chrono::high_resolution_clock::now();

                while (i < limit)
                {
                    // std::cout << i << std::endl;
                    p.run(i++);
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "Elapsed time: " << duration.count() << "ms" << std::endl;
                std::cout << "Per sample: " << duration.count() / limit << "ms" << std::endl;

                // Benchmark throughput


            }

        }
        catch (BaseException &e)
        {
            std::cout << e.what() << std::endl;
            return -1;
        }
    
    return 0;
}
