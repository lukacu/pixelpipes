
#include <cmath>

#include <pixelpipes/operation.hpp>

#include "debug.h"

namespace pixelpipes
{

#define IS_INTEGER(x) ((x)->is<ContainerToken<int>>())

    float sample_normal(float mean, float stdev, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::normal_distribution<float> dst(mean, stdev);
        return dst(generator);
    }

    PIXELPIPES_OPERATION_AUTO("random_normal", sample_normal);

    float sample_uniform(float min, float max, int seed)
    {
        RandomGenerator generator = create_generator(seed);
        std::uniform_real_distribution<float> dst(min, max);
        return dst(generator);
    }

    PIXELPIPES_OPERATION_AUTO("random_uniform", sample_uniform);

    int round_value(float value)
    {
        return static_cast<int>(round(value));
    }

    PIXELPIPES_OPERATION_AUTO("numbers_round", round_value);

    int floor_value(float value)
    {
        return static_cast<int>(floor(value));
    }

    PIXELPIPES_OPERATION_AUTO("numbers_floor", floor_value);

    int ceil_value(float value)
    {
        return static_cast<int>(ceil(value));
    }

    PIXELPIPES_OPERATION_AUTO("numbers_ceil", ceil_value);

    TokenReference number_add(const TokenList& inputs)
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

    PIXELPIPES_OPERATION("numbers_add", number_add);

    TokenReference number_subtract(const TokenReference& a, const TokenReference& b)
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

    PIXELPIPES_OPERATION_AUTO("numbers_subtract", number_subtract);

    TokenReference number_multiply(const TokenList& inputs)
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

    PIXELPIPES_OPERATION("numbers_multiply", number_multiply);

    float number_divide(float a, float b)
    {
        VERIFY(b != 0, "Division with zero");
        return a / b;
    }

    PIXELPIPES_OPERATION_AUTO("numbers_divide", number_divide);

    float number_power(float value, float exponent)
    {
        return pow(value, exponent);
    }

    PIXELPIPES_OPERATION_AUTO("numbers_power", number_power);

    int number_modulo(int value, int quotient)
    {
        return value % quotient;
    }

    PIXELPIPES_OPERATION_AUTO("numbers_modulo", number_modulo);

    TokenReference Maximum(const TokenList& inputs)
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

    PIXELPIPES_OPERATION("numbers_max", Maximum);

    TokenReference number_minimum(const TokenList& inputs)
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

    PIXELPIPES_OPERATION("numbers_min", number_minimum);

    template <typename Op>
    TokenReference compare_binary(float a, float b)
    {
        Op operation;
        return wrap(operation(a, b));
    }

#define compare_equal compare_binary<std::equal_to<float>>
    PIXELPIPES_OPERATION_AUTO("compare_equal", compare_equal);

#define compare_not_equal compare_binary<std::not_equal_to<float>>
    PIXELPIPES_OPERATION_AUTO("compare_not_equal", compare_not_equal);

#define compare_greater compare_binary<std::greater<float>>
    PIXELPIPES_OPERATION_AUTO("compare_greater", compare_greater);

#define compare_less compare_binary<std::less<float>>
    PIXELPIPES_OPERATION_AUTO("compare_less", compare_less);

#define compare_greater_equal compare_binary<std::greater_equal<float>>
    PIXELPIPES_OPERATION_AUTO("compare_greater_equal", compare_greater_equal);

#define compare_less_equal compare_binary<std::less_equal<float>>
    PIXELPIPES_OPERATION_AUTO("compare_less_equal", compare_less_equal);

}