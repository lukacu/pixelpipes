
#include <pixelpipes/operation.hpp>
#include <pixelpipes/module.hpp>

#include "debug.h"

namespace pixelpipes {


uint32_t xorshift32(uint32_t m_seed)
{
    m_seed ^= m_seed << 13;
    m_seed ^= m_seed >> 17;
    m_seed ^= m_seed << 15;
    return m_seed;
}


Operation::Operation() {}

TypeIdentifier Operation::type() {
    return AnyType;
}

StohasticOperation::StohasticOperation() {}

RandomGenerator StohasticOperation::create_generator(SharedToken seed) {
        return RandomGenerator(xorshift32, (uint32_t) Integer::get_value(seed));
}

typedef std::map<std::string, Factory> RegistryMap;

RegistryMap& registry() {
    static RegistryMap map;
    return map;
}

Factory get_operation(const std::string& key) {

    auto val = registry().find(key);

    if (registry().end() == val)
        throw ModuleException("Operation not registered");

    return val->second;
}

bool is_operation_registered(const std::string& key) {

    return registry().end() != registry().find(key);

}

SharedOperation make_operation(const std::string& key, TokenList inputs) {

    if (!is_operation_registered(key)) {
        throw ModuleException(Formatter() << "Name not found: " << key); // TODO: RegistryException
    }

    auto op = std::get<0>(get_operation(key))(inputs);

    return op;
}

OperationDescription describe_operation(const std::string& key) {

    if (!is_operation_registered(key)) {
        throw ModuleException(Formatter() << "Name not found: " << key); // TODO: RegistryException
    }

    return {std::get<2>(get_operation(key)), std::get<1>(get_operation(key))()};

}

void register_operation(const std::string& key, OperationConstructor constructor, OperationDescriber describer) {

    SharedModule context = Module::context();

    std::string global_key = (context) ? (context->name() + ":" + key) : key;

    if (is_operation_registered(global_key)) {
        throw ModuleException(Formatter() << "Name already used: " << global_key);
    }

    registry().insert(RegistryMap::value_type(global_key, Factory(constructor, describer, context)));
    DEBUGMSG("Registering operation: %s \n", global_key.c_str());

}

SharedOperation create_operation(const std::string& key, std::initializer_list<SharedToken> inputs) {

    return make_operation(key, make_span(std::vector<SharedToken>(inputs)));

}

SharedOperation create_operation(const std::string& key, TokenList inputs) {

    return make_operation(key, inputs);

}

}