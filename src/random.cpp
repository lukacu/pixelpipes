
#include <cmath>

#include <pixelpipes/operation.hpp>

#include "debug.h"

namespace pixelpipes
{

#define IS_INTEGER(x) ((x)->is<ContainerToken<int>>())

    TokenReference int_or_float(const TokenList &inputs)
    {
        bool integer = true;

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            integer &= IS_INTEGER(*input);
        }

        if (integer)
            return constant_shape<int>(inputs);
        else
            return constant_shape<float>(inputs);
    }

    float sample_normal(float mean, float stdev, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::normal_distribution<float> dst(mean, stdev);
        return dst(generator);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("sample_normal", sample_normal, constant_shape<float>);

    float sample_uniform(float min, float max, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::uniform_real_distribution<float> dst(min, max);
        return dst(generator);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("sample_uniform", sample_uniform, constant_shape<float>);

    // Other distributions

    float sample_bernoulli(float p, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::bernoulli_distribution dst(p);
        return dst(generator);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("sample_bernoulli", sample_bernoulli, constant_shape<float>);

    float sample_binomial(float p, int n, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::binomial_distribution dst(n, p);
        return dst(generator);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("sample_binomial", sample_binomial, constant_shape<float>);

}