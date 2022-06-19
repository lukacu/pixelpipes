

#include <string>
#include <iostream>

#include <pixelpipes/serialization.hpp>

using namespace pixelpipes;

int main(int argc, const char* argv[]) {

    if (argc == 2) {

        std::string arg(argv[1]);

        try {

        Pipeline p = read_pipeline(arg);

        unsigned long i = 1;

        while (true) { 
            std::cout << i << std::endl;
            p.run(i++);
        }

        } catch (BaseException &e) {
            std::cout << e.what() << std::endl;
            return -1;
        }

    }

    return 0;

}