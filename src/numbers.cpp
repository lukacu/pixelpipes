
#include <cmath>

#include <pixelpipes/operation.hpp>
#include "random.hpp"

#ifndef max
#define max std::max
#endif

#ifndef min
#define min std::min
#endif

namespace pixelpipes {

class NormalDistribution : public StohasticOperation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 3, "Illegal number of inputs");

        RandomGenerator generator = StohasticOperation::create_generator(inputs[2]);

        std::normal_distribution<float> dst(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

        float num = dst(generator);

        return std::make_shared<Float>(num);
    } 

private:
    std::normal_distribution<float> distribution;
};

REGISTER_OPERATION("random_normal", NormalDistribution);


class UniformDistribution : public StohasticOperation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 3, "Illegal number of inputs");

        RandomGenerator generator = StohasticOperation::create_generator(inputs[2]);

        std::uniform_real_distribution<float> dst(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

        float num = dst(generator);

        return std::make_shared<Float>(num);

    }

};

REGISTER_OPERATION("random_uniform", UniformDistribution);

class Round : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);

        return std::make_shared<Integer>(static_cast<int>(round(value)));

    }

};

REGISTER_OPERATION("numbers_round", Round);

class Floor : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);

        return std::make_shared<Integer>(static_cast<int>(floor(value)));

    }

};

REGISTER_OPERATION("numbers_floor", Floor);

class Ceil : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);

        return std::make_shared<Integer>(static_cast<int>(ceil(value)));

    }

};

REGISTER_OPERATION("numbers_ceil", Ceil);

class Add : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() >= 1, "Illegal number of inputs");

        float value = 0;
        bool integer = true;

        for (auto input : inputs) {
            integer &= Integer::is(input);
            value += Float::get_value(input);
        }

        if (integer)
            return std::make_shared<Integer>((int)value);
        else
            return std::make_shared<Float>(value);

    }

};

REGISTER_OPERATION("numbers_add", Add);

class Subtract : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float v1 = Float::get_value(inputs[0]);
        float v2 = Float::get_value(inputs[1]);

        if (Integer::is(inputs[0]) && Integer::is(inputs[1])) {
                return std::make_shared<Integer>((int)(v1 - v2));
        } else
            return std::make_shared<Float>(v1 - v2);

    }

};

REGISTER_OPERATION("numbers_subtract", Subtract);

class Multiply : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() >= 1,"Illegal number of inputs");

        float value = 1;
        bool integer = true;

        for (auto input : inputs) {
            integer &= Integer::is(input);
            value *= Float::get_value(input);
        }

        if (integer)
            return std::make_shared<Integer>((int)value);
        else
            return std::make_shared<Float>(value);

    }

};

REGISTER_OPERATION("numbers_multiply", Multiply);

class Divide : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float v1 = Float::get_value(inputs[0]);
        float v2 = Float::get_value(inputs[1]);

        return std::make_shared<Float>(v1 / v2);

    }

};

REGISTER_OPERATION("numbers_divide", Divide);

class Power : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);
        float exponent = Float::get_value(inputs[1]);

        return std::make_shared<Float>(pow(value, exponent));

    }

};

REGISTER_OPERATION("numbers_power", Power);

class Modulo : public Operation {
public:

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        int value = Integer::get_value(inputs[0]);
        int quotient = Integer::get_value(inputs[1]);

        return std::make_shared<Integer>(value % quotient);

    }

};

REGISTER_OPERATION("numbers_modulo", Modulo);

SharedToken Maximum(TokenList inputs) {

    VERIFY(inputs.size() > 1, "Illegal number of inputs, at least one required");

    float value = Float::get_value(inputs[0]);
    bool integer = true;

    for (auto input : inputs) {
        integer &= Integer::is(input);
        value = max(value, Float::get_value(input));
    }

    if (integer)
        return std::make_shared<Integer>((int)value);
    else
        return std::make_shared<Float>(value);

}

REGISTER_OPERATION_FUNCTION("numbers_max", Maximum);

SharedToken Minimum(TokenList inputs) {

    VERIFY(inputs.size() > 1, "Illegal number of inputs, at least one required");

    float value = Float::get_value(inputs[0]);
    bool integer = true;

    for (auto input : inputs) {
        integer &= Integer::is(input);
        value = min(value, Float::get_value(input));
    }

    if (integer)
        return std::make_shared<Integer>((int)value);
    else
        return std::make_shared<Float>(value);

}

REGISTER_OPERATION_FUNCTION("numbers_min", Minimum);

inline bool compare_values(float a, float b, ComparisonOperation operation) {

    switch (operation) {
    case ComparisonOperation::EQUAL:
        return a == b;
    case ComparisonOperation::NOT_EQUAL:
        return a != b;
    case ComparisonOperation::GREATER_EQUAL:
        return a >= b;
    case ComparisonOperation::LOWER_EQUAL:
        return a <= b;
    case ComparisonOperation::LOWER:
        return a < b;
    case ComparisonOperation::GREATER:
        return a > b;
    }

    return false;

}

class Threshold : public Operation {
public:

    Threshold(float threshold, ComparisonOperation comparison):
        threshold(threshold), comparison(comparison) {}

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        bool result = compare_values(Float::get_value(inputs[0]), threshold, comparison);

        return std::make_shared<Boolean>(result);

    }

private:

    float threshold;
    ComparisonOperation comparison;

};

REGISTER_OPERATION("threshold", Threshold, float, ComparisonOperation);


SharedToken Comparison(TokenList inputs, ComparisonOperation comparison) {

    VERIFY(inputs.size() == 2, "Illegal number of inputs");

    bool result = compare_values(Float::get_value(inputs[0]), Float::get_value(inputs[1]), comparison);

    return std::make_shared<Boolean>(result);

}

REGISTER_OPERATION_FUNCTION("comparison", Comparison, ComparisonOperation);

class ThresholdsConjunction : public Operation {
public:

    ThresholdsConjunction(std::vector<float> thresholds, std::vector<ComparisonOperation> comparison):
        thresholds(thresholds), comparison(comparison) {
        VERIFY(thresholds.size() == comparison.size(), "Length mismatch");
        VERIFY(thresholds.size() >= 1, "At least one comparison required");
    }

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == thresholds.size(), "Illegal number of inputs");

        bool result = true;

        for (size_t i = 0; i < thresholds.size(); i++) {
            float value = Float::get_value(inputs[i]);
            result &= compare_values(value, thresholds[i], comparison[i]);
        }

        return std::make_shared<Integer>((int) result);

    }

private:

    std::vector<float> thresholds;
    std::vector<ComparisonOperation> comparison;
};

//REGISTER_OPERATION("threshold_c", ThresholdsConjunction, std::vector<float>, std::vector<ComparisonOperation>);

class ThresholdsDisjunction : public Operation {
public:

    ThresholdsDisjunction(std::vector<float> thresholds, std::vector<ComparisonOperation> comparison):
        thresholds(thresholds), comparison(comparison) {
        VERIFY(thresholds.size() == comparison.size(), "Length mismatch");
        VERIFY(thresholds.size() >= 1, "At least one comparison required");
    }

    virtual SharedToken run(TokenList inputs) {

        VERIFY(inputs.size() == thresholds.size(), "Illegal number of inputs");

        bool result = false;

        for (size_t i = 0; i < thresholds.size(); i++) {
            float value = Float::get_value(inputs[i]);
            result |= compare_values(value, thresholds[i], comparison[i]);
        }

        return std::make_shared<Integer>((int) result);

    }

private:

    std::vector<float> thresholds;
    std::vector<ComparisonOperation> comparison;
};

//REGISTER_OPERATION("numbers.threshold_d", ThresholdsDisjunction, std::vector<float>, std::vector<ComparisonOperation>);

}