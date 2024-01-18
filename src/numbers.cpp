
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

    PIXELPIPES_UNIT_OPERATION_AUTO("random_normal", sample_normal, constant_shape<float>);

    float sample_uniform(float min, float max, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::uniform_real_distribution<float> dst(min, max);
        return dst(generator);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("random_uniform", sample_uniform, constant_shape<float>);

    int round_value(float value)
    {
        return static_cast<int>(round(value));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_round", round_value, constant_shape<int>);

    int floor_value(float value)
    {
        return static_cast<int>(floor(value));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_floor", floor_value, constant_shape<int>);

    int ceil_value(float value)
    {
        return static_cast<int>(ceil(value));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_ceil", ceil_value, constant_shape<int>);

    TokenReference number_add(const TokenList &inputs)
    {
        VERIFY(inputs.size() >= 1, "Illegal number of inputs");

        float value = 0;
        bool integer = true;

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            integer &= IS_INTEGER(*input);
            value += extract<float>(*input);
        }

        if (integer)
            return wrap((int)value);
        else
            return wrap(value);
    }

    PIXELPIPES_UNIT_OPERATION("numbers_add", number_add, int_or_float);

    TokenReference number_subtract(const TokenReference &a, const TokenReference &b)
    {
        float v1 = extract<float>(a);
        float v2 = extract<float>(b);

        if (IS_INTEGER(a) && IS_INTEGER(b))
        {
            return wrap((int)(v1 - v2));
        }
        else
            return wrap(v1 - v2);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_subtract", number_subtract, int_or_float);

    TokenReference number_multiply(const TokenList &inputs)
    {

        VERIFY(inputs.size() >= 1, "Illegal number of inputs");

        float value = 1;
        bool integer = true;

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            integer &= IS_INTEGER(*input);
            value *= extract<float>(*input);
        }

        if (integer)
            return wrap((int)value);
        else
            return wrap(value);
    }

    PIXELPIPES_UNIT_OPERATION("numbers_multiply", number_multiply, int_or_float);

    float number_divide(float a, float b)
    {
        VERIFY(b != 0, "Division with zero");
        return a / b;
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_divide", number_divide, constant_shape<float>);

    float number_power(float value, float exponent)
    {
        return pow(value, exponent);
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_power", number_power, constant_shape<float>);

    int number_modulo(int value, int quotient)
    {
        return value % quotient;
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("numbers_modulo", number_modulo, constant_shape<int>);

    TokenReference number_maximum(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 1, "Illegal number of inputs, at least one required");

        float value = extract<float>(inputs[0]);
        bool integer = true;

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            integer &= IS_INTEGER(*input);
            value = MAX(value, extract<float>(*input));
        }

        if (integer)
            return wrap((int)value);
        else
            return wrap(value);
    }

    PIXELPIPES_UNIT_OPERATION("numbers_max", number_maximum, int_or_float);

    TokenReference number_minimum(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 1, "Illegal number of inputs, at least one required");

        float value = extract<float>(inputs[0]);
        bool integer = true;

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            integer &= IS_INTEGER(*input);
            value = MIN(value, extract<float>(*input));
        }

        if (integer)
            return wrap((int)value);
        else
            return wrap(value);
    }

    PIXELPIPES_UNIT_OPERATION("numbers_min", number_minimum, int_or_float);

    template <typename Op>
    TokenReference compare_binary(float a, float b)
    {
        Op operation;
        return wrap(operation(a, b));
    }

#define compare_equal compare_binary<std::equal_to<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_equal", compare_equal, constant_shape<bool>);

#define compare_not_equal compare_binary<std::not_equal_to<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_not_equal", compare_not_equal, constant_shape<bool>);

#define compare_greater compare_binary<std::greater<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_greater", compare_greater, constant_shape<bool>);

#define compare_less compare_binary<std::less<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_less", compare_less, constant_shape<bool>);

#define compare_greater_equal compare_binary<std::greater_equal<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_greater_equal", compare_greater_equal, constant_shape<bool>);

#define compare_less_equal compare_binary<std::less_equal<float>>
    PIXELPIPES_UNIT_OPERATION_AUTO("compare_less_equal", compare_less_equal, constant_shape<bool>);

}