#pragma once

#include <random>
#include <memory>

#include "engine.hpp"
#include "types.hpp"

namespace pixelpipes {

typedef std::default_random_engine RandomGenerator;

class Sampling : public Operation {
protected:
    virtual OperationType type();

    RandomGenerator create_generator(ContextHandle context);
};

}

