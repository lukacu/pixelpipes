
#include <pixelpipes/operation.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes {

OperationType OperationObserver::getType(const SharedOperation& operation) const { 
    return operation->type();
}

Operation::Operation() {};

OperationType Operation::type() {
    return OperationType::Computation;
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

SharedOperation make_operation(const std::string& key, std::vector<SharedVariable> inputs) {

    if (!is_operation_registered(key)) {
        throw ModuleException(std::string("Name not found: ") + key); // TODO: RegistryException
    }

    auto op = std::get<0>(get_operation(key))(inputs);

    return op;
}

OperationDescription describe_operation(const std::string& key) {

    if (!is_operation_registered(key)) {
        throw ModuleException(std::string("Name not found: ") + key); // TODO: RegistryException
    }

    return {std::get<2>(get_operation(key)), std::get<1>(get_operation(key))()};

}

void register_operation(const std::string& key, OperationConstructor constructor, OperationDescriber describer) {

    SharedModule context = Module::context();

    std::string global_key = (context) ? (context->name() + ":" + key) : key;

    registry().insert(RegistryMap::value_type(key, Factory(constructor, describer, context)));
    DEBUGMSG("Registering operation: %s \n", key.c_str());

}

SharedOperation create_operation(const std::string& key, std::initializer_list<SharedVariable> inputs) {

    return make_operation(key, std::vector<SharedVariable>(inputs));

}

SharedOperation create_operation(const std::string& key, std::vector<SharedVariable> inputs) {

    return make_operation(key, inputs);

}

}