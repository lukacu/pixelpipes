#include <pixelpipes/module.hpp>
#include <pixelpipes/operation.hpp>

using namespace pixelpipes;

PIXELPIPES_MODULE(example)

namespace
{
    int triple(int value)
    {
        return value * 3;
    }
}

PIXELPIPES_COMPUTE_OPERATION_AUTO(
    "example_triple",
    triple,
    constant_shape<int>
);
