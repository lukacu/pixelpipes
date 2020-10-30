
#ifndef __PP_NUMBERS_H__
#define __PP_NUMBERS_H__

#include "random"

#include "engine.hpp"

namespace pixelpipes {

enum class ComparisonOperation {EQUAL, LOWER, LOWER_EQUAL, GREATER, GREATER_EQUAL};
enum class LogicalOperation {AND, OR, NOT};

class Integer: public Variable {
public:

    Integer(int value): value(value) {};

    ~Integer() = default;

    int get() { return value; };

    virtual VariableType type() { return VariableType::Integer; };

    static int get_value(SharedVariable v);

private:

    int value;

};

class Float: public Variable {
public:

    Float(float value): value(value) {};

    ~Float() = default;

    float get() { return value; };

    virtual VariableType type() { return VariableType::Float; };

    static float get_value(SharedVariable v);

private:

    float value;

};

class Constant : public Operation {
public:

    Constant(SharedVariable value);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

private:

    SharedVariable value;

};


class Sampling : public Operation {
protected:

    virtual OperationType type();

};

class NormalDistribution : public Sampling {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

private:
    std::normal_distribution<float> distribution;
};

class UniformDistribution : public Sampling {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Round : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Add : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Subtract : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Multiply : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Divide : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

class Power : public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

};

}

#endif