
#include <fstream>

#include <pixelpipes/serialization.hpp>

using namespace std;
using namespace pixelpipes;

int main(int argc, char* argv[]) {

    if (argc == 2) {
        string arg(argv[1]);

        PipelineReader reader;

        reader.read(arg);

    }

    return 0;

}
