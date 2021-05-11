#pragma once
#include <memory>
#include <functional>
#include <list>
#include <tuple>
#include <map>

#include <pixelpipes/types.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes {

class OperationException;

enum class OperationType {Deterministic, Stohastic, Output, Control};

enum class ComparisonOperation {EQUAL, LOWER, LOWER_EQUAL, GREATER, GREATER_EQUAL, NOT_EQUAL};
enum class LogicalOperation {AND, OR, NOT};
enum class ArithmeticOperation {ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER, MODULO};

enum class SamplingDistribution {Normal, Uniform};

enum class ContextData {SampleIndex};

class Context {
public:
    Context(unsigned long index);
    ~Context() = default;

    unsigned int random();
    unsigned long sample();

private:

    unsigned long index;
    std::default_random_engine generator;

};

typedef std::shared_ptr<Context> ContextHandle;

class Operation;

typedef std::shared_ptr<Operation> SharedOperation;

class OperationObserver {
protected:
    OperationType getType(const SharedOperation& operation) const;

};

class Operation: public std::enable_shared_from_this<Operation> {
friend OperationObserver;
public:
    
    ~Operation() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) = 0;

protected:

    Operation();

    virtual OperationType type();

};

typedef std::default_random_engine RandomGenerator;

class StohasticOperation: public Operation {
public:
    
    ~StohasticOperation() = default;

    static RandomGenerator create_generator(ContextHandle context) {
            return std::default_random_engine(context->random());
    }

protected:

    StohasticOperation();

    virtual OperationType type();

};


class OperationException : public BaseException {
public:

    OperationException(std::string reason, SharedOperation operation): BaseException(reason), operation(operation) {}

private:

    SharedOperation operation;
};

namespace details {

    template <class F, class Tuple, std::size_t... I>
    constexpr decltype(auto) apply_impl(F&& f, std::vector<SharedVariable> inputs, 
                ContextHandle context, Tuple t, std::index_sequence<I...>) {
        return std::invoke(std::forward<F>(f), inputs, context, std::get<I>(std::forward<Tuple>(t))...);
    }

    template <class F, class Tuple>
    constexpr decltype(auto) apply(F&& f, std::vector<SharedVariable> inputs, ContextHandle context, Tuple t)
    {
        return apply_impl(
            std::forward<F>(f), inputs, context, std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }


    template <class T, class Tuple, std::size_t... I>
    constexpr std::shared_ptr<T> make_from_tuple_impl(Tuple&& t, std::index_sequence<I...>) {
        return std::shared_ptr<T>(new T(std::get<I>(std::forward<Tuple>(t))...));
    }
    
    template <class T, class Tuple>
    constexpr std::shared_ptr<T> make_from_tuple(Tuple&& t) {
        return make_from_tuple_impl<T>(std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }


    typedef std::vector<SharedVariable>::const_iterator ArgIterator;

    template <typename A>
    std::tuple<> iterate_args(ArgIterator current, ArgIterator end) {
        if (current != end) {
            throw VariableException("Number of inputs does not match");
        }

        return std::make_tuple();
    }

    // base template with 1 argument (which will be called from the variadic one).
    template <typename A, typename Arg>
    std::tuple<Arg> iterate_args(ArgIterator current, ArgIterator end) {

        //std::cout << Type<Arg>::name << " - " << (*current)->describe() << std::endl;

        if (current == end) {
            throw VariableException("Number of inputs does not match");
        }


        return std::tuple(Conversion<Arg>::extract(*current));
    }

    template <typename A, typename First, typename Second, typename... Args>
    std::tuple<First, Second, Args...> iterate_args(ArgIterator current, ArgIterator end) {
        if (current == end) {
            throw VariableException("Number of inputs does not match");
        }

        return std::tuple_cat(iterate_args<A, First>(current, end), iterate_args<A, Second, Args...>(current+1, end));

    }

}

template<typename Fn, Fn fn, typename Base, typename ...Args>
class OperationWrapper: public Base {
public:

    OperationWrapper(Args... args) : args(std::forward<Args>(args)...) {
        static_assert(std::is_base_of<Operation, Base>::value, "Base class must inherit from Operation");
    };
    ~OperationWrapper() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        try {
            
            // LINUX
            return details::apply(std::function(fn), inputs, context, args); //  std::tuple_cat(std::make_tuple(inputs, context), args) );
            
            // WINDOWS
            //return apply(std::forward<Fn>(fn), inputs, context, args);
            //return fn(inputs, context, std::forward<Args>(args)...);

        } catch (VariableException &e) {
            throw OperationException(e.what(), this->shared_from_this());
        }

    }
    
protected:

    std::tuple<Args...> args;

};

typedef std::vector<TypeIdentifier> OperationDescription;

template <typename OperationClass = Operation, typename ...Args>
struct OperationFactory {


    static OperationDescription describe() { 

        return OperationDescription { GetTypeIdentifier<Args>()... };

    }

    static SharedOperation new_instance(std::vector<SharedVariable> inputs) { 

        if (inputs.size() != sizeof...(Args))
            throw VariableException("Wrong number of parameters");

        auto converted = details::iterate_args<OperationClass, Args...>(inputs.begin(), inputs.end());

        auto op = details::make_from_tuple<OperationClass>(converted);

        return op;

    }

};


class PIXELPIPES_API OperationRegistry {
public:
    typedef std::function<SharedOperation(std::vector<SharedVariable>)> FactoryConstructor;
    typedef std::function<OperationDescription()> FactoryDescriber;
    typedef std::pair<FactoryConstructor, FactoryDescriber> Factory;
    typedef std::map<std::string, Factory> RegistryMap;

    template <typename OperationClass = Operation, typename ...Args>
    void register_operation(const std::string& key) {

        if (is_registered(key)) {
            throw ModuleException(std::string("Name already used: ") + key);
        }

        auto fp = Factory(OperationFactory<OperationClass, Args...>::new_instance,
            OperationFactory<OperationClass, Args...>::describe);
        set(key, fp);

    }

    template <typename ...Args>
    SharedOperation make_operation(const std::string& key, Args&& ... args) {

        return make_operation(key, std::vector<SharedVariable>({args ...}));

    }

    virtual SharedOperation make_operation(const std::string& key, std::vector<SharedVariable> inputs) = 0;
    virtual OperationDescription describe_operation(const std::string& key) = 0;
    virtual bool is_registered(const std::string& key) = 0;

protected:
    
    virtual Factory get(const std::string& key) = 0;
    virtual void set(const std::string& key, Factory& factory) = 0;

};

typedef std::function<void(OperationRegistry&)> OperationRegistrar;

template<typename Fn, Fn fn, typename Base, typename ...Args>
void register_operation_function(const std::string& name, OperationRegistry& registry) {
    registry.register_operation<OperationWrapper<Fn, fn, Base, Args...>, Args...>(name);
}

template<typename OperationClass = Operation, typename ...Args>
void register_operation_class(const std::string& name, OperationRegistry& registry) {
    registry.register_operation<OperationClass, Args...>(name);
}

class OperationDirectInitializer {
public:
    OperationDirectInitializer(std::function<void(OperationRegistry&)> registrar);
};

#ifdef PIXELPIPES_BUILD_CORE
typedef OperationDirectInitializer OperationInitializer;
#else
class OperationInitializer {
public:
    OperationInitializer(std::function<void(OperationRegistry&)> registrar);
};
#endif

#define PIXELPIPES_MODULE(N) \
std::list<OperationRegistrar> & N ## _registry_initializers() { static std::list<OperationRegistrar> inits;  return inits; } \
OperationInitializer::OperationInitializer(OperationRegistrar registrar) { ( N ## _registry_initializers)().push_back(registrar); } \
extern "C" { \
const char* pixelpipes_module = STRINGIFY(N); \
void pixelpipes_register_operations(OperationRegistry& registry) { \
    for (auto initializer : ( N ## _registry_initializers)()) initializer(registry); \
} }

#define REGISTER_OPERATION_FUNCTION_WITH_BASE(N, FUNC, BASE, ...) OperationInitializer FUNC ##_initialization([](OperationRegistry& registry) { register_operation_function<decltype(&FUNC), &FUNC, BASE, ## __VA_ARGS__>(N, registry); })
#define REGISTER_OPERATION_FUNCTION(N, FUNC, ...) REGISTER_OPERATION_FUNCTION_WITH_BASE(N, FUNC, Operation, ## __VA_ARGS__)
#define REGISTER_OPERATION(N, O, ...) OperationInitializer O ##_initialization([](OperationRegistry& registry) { register_operation_class<O, ## __VA_ARGS__>(N, registry); } )

OperationDescription describe_operation(const std::string& key);
SharedOperation create_operation(const std::string& key, std::vector<SharedVariable> inputs);
SharedOperation create_operation(const std::string& key, std::initializer_list<SharedVariable> inputs);

}