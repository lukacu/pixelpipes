
#include "types.hpp"
#include "python.hpp"
#include "random.hpp"

namespace pixelpipes {

class NormalDistribution : public Sampling {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        if (inputs.size() != 2)
            throw OperationException("Illegal number of inputs", shared_from_this());

        RandomGenerator generator = create_generator(context);

        std::normal_distribution<float> distribution(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

        float num = distribution(generator);

        return std::make_shared<Float>(num);
    } 

private:
    std::normal_distribution<float> distribution;
};

REGISTER_OPERATION(NormalDistribution);


class UniformDistribution : public Sampling {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        RandomGenerator generator = create_generator(context);

        std::uniform_real_distribution<float> distribution(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

        float num = distribution(generator);

        return std::make_shared<Float>(num);

    }

};

REGISTER_OPERATION(UniformDistribution);

class Round : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);

        return std::make_shared<Integer>(static_cast<int>(round(value)));

    }

};

REGISTER_OPERATION(Round);

class Add : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() >= 1, "Illegal number of inputs");

        float value = 0;
        bool integer = true;

        for (auto input : inputs) {
            integer &= input->type() == VariableType::Integer;
            value += Float::get_value(input);
        }

        if (integer)
            return std::make_shared<Integer>((int)value);
        else
            return std::make_shared<Float>(value);

    }

};

REGISTER_OPERATION(Add);

class Subtract : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float v1 = Float::get_value(inputs[0]);
        float v2 = Float::get_value(inputs[1]);

        if (inputs[0]->type() == VariableType::Integer &&
            inputs[1]->type() == VariableType::Integer) {
                return std::make_shared<Integer>((int)(v1 - v2));
        } else
            return std::make_shared<Float>(v1 - v2);

    }

};

REGISTER_OPERATION(Subtract);

class Multiply : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() >= 1,"Illegal number of inputs");

        float value = 1;
        bool integer = true;

        for (auto input : inputs) {
            integer &= input->type() == VariableType::Integer;
            value *= Float::get_value(input);
        }

        if (integer)
            return std::make_shared<Integer>((int)value);
        else
            return std::make_shared<Float>(value);

    }

};

REGISTER_OPERATION(Multiply);

class Divide : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float v1 = Float::get_value(inputs[0]);
        float v2 = Float::get_value(inputs[1]);

        return std::make_shared<Float>(v1 / v2);

    }

};

REGISTER_OPERATION(Divide);

class Power : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        float value = Float::get_value(inputs[0]);
        float exponent = Float::get_value(inputs[1]);

        return std::make_shared<Float>(pow(value, exponent));

    }

};

REGISTER_OPERATION(Power);

class Modulo : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 2, "Illegal number of inputs");

        int value = Integer::get_value(inputs[0]);
        int exponent = Integer::get_value(inputs[1]);

        return std::make_shared<Float>(value % exponent);

    }

};

REGISTER_OPERATION(Modulo);

inline bool compare_values(float a, float b, ComparisonOperation operation) {

    switch (operation) {
    case ComparisonOperation::EQUAL:
        return a == b;
    case ComparisonOperation::GREATER_EQUAL:
        return a >= b;
    case ComparisonOperation::LOWER_EQUAL:
        return a <= b;
    case ComparisonOperation::LOWER:
        return a < b;
    case ComparisonOperation::GREATER:
        return a > b;
    }

}

class Threshold : public Operation {
public:

    Threshold(float threshold, ComparisonOperation comparison):
        threshold(threshold), comparison(comparison) {}

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == 1, "Illegal number of inputs");

        bool result = compare_values(Float::get_value(inputs[0]), threshold, comparison);

        return std::make_shared<Integer>((int) result);

    }

private:

    float threshold;
    ComparisonOperation comparison;

};

REGISTER_OPERATION(Threshold, float, ComparisonOperation);

class ThresholdsConjunction : public Operation {
public:

    ThresholdsConjunction(std::vector<float> thresholds, std::vector<ComparisonOperation> comparison):
        thresholds(thresholds), comparison(comparison) {
        VERIFY(thresholds.size() == comparison.size(), "Length mismatch");
        VERIFY(thresholds.size() >= 1, "At least one comparison required");
    }

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == thresholds.size(), "Illegal number of inputs");

        bool result = true;

        for (int i = 0; i < thresholds.size(); i++) {
            float value = Float::get_value(inputs[i]);
            result &= compare_values(value, thresholds[i], comparison[i]);
        }

        return std::make_shared<Integer>((int) result);

    }

private:

    std::vector<float> thresholds;
    std::vector<ComparisonOperation> comparison;
};

REGISTER_OPERATION(ThresholdsConjunction, std::vector<float>, std::vector<ComparisonOperation>);

class ThresholdsDisjunction : public Operation {
public:

    ThresholdsDisjunction(std::vector<float> thresholds, std::vector<ComparisonOperation> comparison):
        thresholds(thresholds), comparison(comparison) {
        VERIFY(thresholds.size() == comparison.size(), "Length mismatch");
        VERIFY(thresholds.size() >= 1, "At least one comparison required");
    }

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        VERIFY(inputs.size() == thresholds.size(), "Illegal number of inputs");

        bool result = false;

        for (int i = 0; i < thresholds.size(); i++) {
            float value = Float::get_value(inputs[i]);
            result |= compare_values(value, thresholds[i], comparison[i]);
        }

        return std::make_shared<Integer>((int) result);

    }

private:

    std::vector<float> thresholds;
    std::vector<ComparisonOperation> comparison;
};

REGISTER_OPERATION(ThresholdsDisjunction, std::vector<float>, std::vector<ComparisonOperation>);

}