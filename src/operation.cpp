
#include <pixelpipes/operation.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes {

TypeIdentifier OperationObserver::get_type(const SharedOperation& operation) const { 
    return operation->op_type();
}

Operation::Operation() {};

TypeIdentifier Operation::op_type() {
    return AnyType;
}

StohasticOperation::StohasticOperation() {};

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

SharedOperation make_operation(const std::string& key, std::vector<SharedToken> inputs) {

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

    return make_operation(key, std::vector<SharedToken>(inputs));

}

SharedOperation create_operation(const std::string& key, std::vector<SharedToken> inputs) {

    return make_operation(key, inputs);

}

}