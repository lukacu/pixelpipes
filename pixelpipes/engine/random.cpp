
#include "random.hpp"

namespace pixelpipes {

OperationType Sampling::type() {

    return OperationType::Stohastic;

}

RandomGenerator Sampling::create_generator(ContextHandle context) {

    return std::default_random_engine(context->random());

}

}