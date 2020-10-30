
#include "numbers.hpp"

namespace pixelpipes {

int Integer::get_value(SharedVariable v) {
    if (v->type() != VariableType::Integer)
        throw VariableException("Not an integer value");

    return std::static_pointer_cast<Integer>(v)->get();
}

float Float::get_value(SharedVariable v) {
    if (v->type() == VariableType::Integer) {
        return std::static_pointer_cast<Integer>(v)->get();
    }

    if (v->type() != VariableType::Float)
        throw VariableException("Not a float value");

    return std::static_pointer_cast<Float>(v)->get();
}


Constant::Constant(SharedVariable value): value(value) {}

SharedVariable Constant::run(std::vector<SharedVariable> inputs, ContextHandle context) {
    return value;
}

OperationType Sampling::type() {
    return OperationType::Stohastic;
}

SharedVariable NormalDistribution::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

    std::default_random_engine generator(context->random());

    std::normal_distribution<float> distribution(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

    float num = distribution(generator);

    return std::make_shared<Float>(num);
} 

SharedVariable UniformDistribution::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

    std::default_random_engine generator(context->random());

    std::uniform_real_distribution<float> distribution(Float::get_value(inputs[0]), Float::get_value(inputs[1]));

    float num = distribution(generator);

    return std::make_shared<Float>(num);

}

SharedVariable Round::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1)
        throw OperationException("Illegal number of inputs", shared_from_this());

    float value = Float::get_value(inputs[0]);

    return std::make_shared<Integer>(static_cast<int>(round(value)));

}

SharedVariable Add::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() < 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

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

SharedVariable Subtract::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

    float v1 = Float::get_value(inputs[0]);
    float v2 = Float::get_value(inputs[1]);

    if (inputs[0]->type() == VariableType::Integer &&
        inputs[1]->type() == VariableType::Integer) {
            return std::make_shared<Integer>((int)(v1 - v2));
    } else
        return std::make_shared<Float>(v1 - v2);

}

SharedVariable Multiply::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() < 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

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

SharedVariable Divide::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

    float v1 = Float::get_value(inputs[0]);
    float v2 = Float::get_value(inputs[1]);

    return std::make_shared<Float>(v1 / v2);
}

SharedVariable Power::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2)
        throw OperationException("Illegal number of inputs", shared_from_this());

    float value = Float::get_value(inputs[0]);
    float exponent = Float::get_value(inputs[1]);

    return std::make_shared<Float>(pow(value, exponent));

}


}