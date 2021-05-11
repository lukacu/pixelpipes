
#include <pixelpipes/operation.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes {

OperationType OperationObserver::getType(const SharedOperation& operation) const { 
    return operation->type();
}

Context::Context(unsigned long index) : index(index), generator(index) {

}

unsigned int Context::random() {
    return generator();
}

unsigned long Context::sample() {
    return index;
}

Operation::Operation() {};

OperationType Operation::type() {
    return OperationType::Deterministic;
}

StohasticOperation::StohasticOperation() {};

OperationType StohasticOperation::type() {
    return OperationType::Stohastic;
}

typedef void (*OperationRegistryCallback) (OperationRegistry&);

void register_module_operations(Module& m);

class PrefixOperationRegistry;

class CoreOperationRegistry: public OperationRegistry {
friend PrefixOperationRegistry;
public:

    CoreOperationRegistry() {

        Module::callback(register_module_operations);

    }

    ~CoreOperationRegistry() {}

    bool is_registered(const std::string& key) {
        return map.end() != map.find(key);
    }

    virtual SharedOperation make_operation(const std::string& key, std::vector<SharedVariable> inputs) {

        if (!is_registered(key)) {
            throw ModuleException(std::string("Name not found: ") + key); // TODO: RegistryException
        }

        auto op = get(key).first(inputs);

        return op;
    }


    virtual OperationDescription describe_operation(const std::string& key) {

        if (!is_registered(key)) {
            throw ModuleException(std::string("Name not found: ") + key); // TODO: RegistryException
        }

        return get(key).second();

    }


private: 

    RegistryMap map;

    virtual Factory get(const std::string& key) {

        auto val = map.find(key);

        if (map.end() == val)
            throw ModuleException("Operation not registered");

        return val->second;
    }

    virtual void set(const std::string& key, OperationRegistry::Factory& factory) {
        map.insert(OperationRegistry::RegistryMap::value_type(key, factory));
        DEBUGMSG("Registering operation: %s \n", key.c_str());
    }

};

static CoreOperationRegistry registry;

class PrefixOperationRegistry: public OperationRegistry {
public:

    PrefixOperationRegistry(const std::string& prefix): prefix(prefix) {}
    ~PrefixOperationRegistry() {}

    bool is_registered(const std::string& key) {
        return registry.is_registered(prefix + key);
    }

    virtual SharedOperation make_operation(const std::string& key, std::vector<SharedVariable> inputs) {

        throw ModuleException("Registry only used to register new operaitons here");

    }

    virtual OperationDescription describe_operation(const std::string& key) {

        throw ModuleException("Registry only used to register new operaitons here");

    }

private:

    std::string prefix;

    virtual Factory get(const std::string& key) {
        return registry.get(prefix + key);
    }

    virtual void set(const std::string& key, OperationRegistry::Factory& factory) {
        registry.set(prefix + key, factory);
    }

};


OperationDirectInitializer::OperationDirectInitializer(std::function<void(OperationRegistry&)> registrar) {
    registrar(registry);
}

OperationDescription describe_operation(const std::string& key) {

    return registry.describe_operation(key);

}

SharedOperation create_operation(const std::string& key, std::initializer_list<SharedVariable> inputs) {

    return registry.make_operation(key, std::vector<SharedVariable>(inputs));

}

SharedOperation create_operation(const std::string& key, std::vector<SharedVariable> inputs) {

    return registry.make_operation(key, inputs);

}

void register_module_operations(Module& m) {

    // We get pointer to pointer
    auto name = m.symbol<const char **>("pixelpipes_module");

    if (!name)
        return;


    auto cb = m.symbol<OperationRegistryCallback>("pixelpipes_register_operations");

    DEBUGMSG("Registering with prefix '%s'\n", name[0]);

    PrefixOperationRegistry prefix_registry(std::string(name[0]) + ":");

    if (cb) cb(prefix_registry);

}

}