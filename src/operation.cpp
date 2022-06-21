
#include <pixelpipes/operation.hpp>
#include <pixelpipes/module.hpp>

#include "debug.h"

namespace pixelpipes
{

    PIXELPIPES_REGISTER_ENUM("comparison", ComparisonOperation);
    PIXELPIPES_REGISTER_ENUM("logical", LogicalOperation);
    PIXELPIPES_REGISTER_ENUM("arithmetic", ArithmeticOperation);

    uint32_t xorshift32(uint32_t m_seed)
    {
        m_seed ^= m_seed << 13;
        m_seed ^= m_seed >> 17;
        m_seed ^= m_seed << 5;
        return m_seed;
    }

    RandomGenerator make_generator(uint32_t seed)
    {
        if (seed == 0)
        {
            seed = ~seed;
        }
        return RandomGenerator(xorshift32, xorshift32(seed));
    }

    Operation::Operation() {}

    TypeIdentifier Operation::type()
    {
        return AnyType;
    }

    RandomGenerator create_generator(TokenReference seed)
    {
        return make_generator((uint32_t)extract<int>(seed));
    }

    RandomGenerator create_generator(int seed)
    {
        return make_generator((uint32_t)seed);
    }

    typedef std::tuple<TypeIdentifier, OperationConstructor, OperationDescriber, ModuleReference> Factory;
    typedef std::map<std::string, Factory> RegistryMap;

    RegistryMap &_registry()
    {
        static RegistryMap map;
        return map;
    }

    template <size_t N>
    inline std::tuple_element_t<N, Factory> get_operation_data(const std::string &key)
    {

        auto val = _registry().find(key);

        if (_registry().end() == val)
            throw ModuleException("Operation not registered");

        return std::get<N>(val->second);
    }

    bool is_operation_registered(const std::string &key)
    {

        return _registry().end() != _registry().find(key);
    }

    OperationReference make_operation(const std::string &key, const TokenList &inputs)
    {

        if (!is_operation_registered(key))
        {
            throw ModuleException(Formatter() << "Name not found: " << key); // TODO: RegistryException
        }

        auto op = get_operation_data<1>(key)(inputs);

        return op;
    }

    OperationDescription describe_operation(const std::string &key)
    {

        if (!is_operation_registered(key))
        {
            throw ModuleException(Formatter() << "Name not found: " << key); // TODO: RegistryException
        }

        return get_operation_data<2>(key)();
    }

    ModuleReference operation_source(const std::string &key)
    {
        if (!is_operation_registered(key))
        {
            throw ModuleException(Formatter() << "Name not found: " << key); // TODO: RegistryException
        }

        auto val = _registry().find(key);

        if (_registry().end() == val)
            throw ModuleException("Operation not registered");

        return std::get<3>(val->second).reborrow();
    }

    void register_operation(const std::string &key, OperationConstructor constructor, OperationDescriber describer)
    {

        ModuleReference context = Module::context();

        std::string global_key = (context) ? (context->name() + ":" + key) : key;

        if (is_operation_registered(global_key))
        {
            throw ModuleException(Formatter() << "Name already used: " << global_key);
        }

        OperationDescription d = describer();

        _registry().insert(RegistryMap::value_type(global_key, Factory(d.identifier, constructor, describer, context.reborrow())));
        DEBUGMSG("Registering operation: %s \n", global_key.c_str());
    }

    OperationReference create_operation(const std::string &key, const std::initializer_list<TokenReference> &inputs)
    {
        return make_operation(key, Sequence<TokenReference>(inputs));
    }

    OperationReference create_operation(const std::string &key, const TokenList &inputs)
    {
        return make_operation(key, inputs);
    }

    std::string operation_name(const OperationReference &operation)
    {

        for (auto x = _registry().begin(); x != _registry().end(); x++)
        {

            if (std::get<0>(x->second) == operation->type())
                return x->first;
        }

        return std::string();
    }

}