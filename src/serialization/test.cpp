
#include <fstream>

#include "compression.hpp"

using namespace std;
using namespace pixelpipes;

int main(int argc, char* argv[]) {

    bool compress = true;

    if (argc == 1) {
        string arg(argv[1]);

        if (arg == string("decompress")) {
            compress = false;
        }
    }

    if (compress) {

        OutputCompressionStream cs(cout);

        cs << cin.rdbuf();

    } else {

        InputCompressionStream cs(cin);

        cout << cs.rdbuf();

    }

    return 0;

}
