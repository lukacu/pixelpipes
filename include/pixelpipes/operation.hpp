#pragma once
#include <memory>
#include <functional>
#include <list>
#include <tuple>
#include <map>

#include <pixelpipes/token.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes {

class OperationException;

enum class ComparisonOperation {EQUAL, LOWER, LOWER_EQUAL, GREATER, GREATER_EQUAL, NOT_EQUAL};
enum class LogicalOperation {AND, OR, NOT};
enum class ArithmeticOperation {ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER, MODULO};

enum class SamplingDistribution {Normal, Uniform};

enum class ContextData {SampleIndex, OperationIndex, RandomSeed};


PIXELPIPES_CONVERT_ENUM(ContextData)
PIXELPIPES_CONVERT_ENUM(SamplingDistribution)

PIXELPIPES_CONVERT_ENUM(ArithmeticOperation)
PIXELPIPES_CONVERT_ENUM(LogicalOperation)
PIXELPIPES_CONVERT_ENUM(ComparisonOperation)


class Operation;

typedef std::shared_ptr<Operation> SharedOperation;

class PIXELPIPES_API Operation: public std::enable_shared_from_this<Operation> {
public:
    
    ~Operation() = default;

    virtual SharedToken run(std::vector<SharedToken> inputs) = 0;

    virtual TypeIdentifier type();

protected:

    Operation();

};

typedef std::default_random_engine RandomGenerator;

class PIXELPIPES_API StohasticOperation: public Operation {
public:
    
    ~StohasticOperation() = default;

    static RandomGenerator create_generator(SharedToken seed) {
            return std::default_random_engine(Integer::get_value(seed));
    }

protected:

    StohasticOperation();

};


class PIXELPIPES_API OperationException : public BaseException {
public:

    OperationException(std::string reason, SharedOperation operation): BaseException(reason), operation(operation) {}

private:

    SharedOperation operation;
};

namespace details {

    template <class F, class Tuple, std::size_t... I>
    constexpr decltype(auto) apply_impl(F&& f, std::vector<SharedToken> inputs, 
                Tuple t, std::index_sequence<I...>) {
                    UNUSED(t); // don't know why this causes unused error?
        return std::invoke(std::forward<F>(f), inputs, std::get<I>(std::forward<Tuple>(t))...);
    }

    template <class F, class Tuple>
    constexpr decltype(auto) apply(F&& f, std::vector<SharedToken> inputs, Tuple t)
    {
        return apply_impl(
            std::forward<F>(f), inputs, std::forward<Tuple>(t),
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

    typedef std::vector<SharedToken>::const_iterator ArgIterator;

    template <typename A>
    std::tuple<> iterate_args(ArgIterator current, ArgIterator end) {
        if (current != end) {
            throw TypeException("Number of inputs does not match");
        }

        return std::make_tuple();
    }

    // base template with 1 argument (which will be called from the variadic one).
    template <typename A, typename Arg>
    std::tuple<Arg> iterate_args(ArgIterator current, ArgIterator end) {

        //std::cout << Type<Arg>::name << " - " << (*current)->describe() << std::endl;

        if (current == end) {
            throw TypeException("Number of inputs does not match");
        }


        return std::tuple(extract<Arg>(*current));
    }

    template <typename A, typename First, typename Second, typename... Args>
    std::tuple<First, Second, Args...> iterate_args(ArgIterator current, ArgIterator end) {
        if (current == end) {
            throw TypeException("Number of inputs does not match");
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

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        try {
            
            // LINUX
            return details::apply(std::function(fn), inputs, args); //  std::tuple_cat(std::make_tuple(inputs), args) );
            
            // WINDOWS
            //return apply(std::forward<Fn>(fn), inputs, args);
            //return fn(inputs, std::forward<Args>(args)...);

        } catch (TypeException &e) {
            throw OperationException(e.what(), this->shared_from_this());
        }

    }
    
    virtual TypeIdentifier type() {
        return GetTypeIdentifier<OperationWrapper<Fn, fn, Base, Args...>>();
    }

    template<typename T> bool is() {
        return type() == GetTypeIdentifier<T>();
    }

protected:

    std::tuple<Args...> args;

};

typedef std::vector<TypeIdentifier> OperationArguments;

struct OperationDescription {


    SharedModule source;
    OperationArguments arguments;

};

template <typename OperationClass = Operation, typename ...Args>
struct OperationFactory {

    static OperationArguments arguments() { 

        return OperationArguments { GetTypeIdentifier<Args>()... };

    }

    static SharedOperation new_instance(std::vector<SharedToken> inputs) { 

        if (inputs.size() != sizeof...(Args))
            throw TypeException("Wrong number of parameters");

        auto converted = details::iterate_args<OperationClass, Args...>(inputs.begin(), inputs.end());

        auto op = details::make_from_tuple<OperationClass>(converted);

        return op;

    }

};

typedef std::function<SharedOperation(std::vector<SharedToken>)> OperationConstructor;
typedef std::function<OperationArguments()> OperationDescriber;
typedef std::tuple<OperationConstructor, OperationDescriber, SharedModule> Factory;

SharedOperation PIXELPIPES_API make_operation(const std::string& key, std::vector<SharedToken> inputs);
void PIXELPIPES_API register_operation(const std::string& key, OperationConstructor constructor, OperationDescriber describer);
bool PIXELPIPES_API is_operation_registered(const std::string& key);

template <typename OperationClass = Operation, typename ...Args>
void register_operation(const std::string& key) {

    register_operation(key, OperationFactory<OperationClass, Args...>::new_instance, OperationFactory<OperationClass, Args...>::arguments);

}

template <typename ...Args>
SharedOperation make_operation(const std::string& key, Args&& ... args) {

    return make_operation(key, std::vector<SharedToken>({args ...}));

}

OperationDescription PIXELPIPES_API describe_operation(const std::string& key);
SharedOperation PIXELPIPES_API create_operation(const std::string& key, std::vector<SharedToken> inputs);
SharedOperation PIXELPIPES_API create_operation(const std::string& key, std::initializer_list<SharedToken> inputs);

template<typename Fn, Fn fn, typename Base, typename ...Args>
void register_operation_function(const std::string& name) {
    register_operation<OperationWrapper<Fn, fn, Base, Args...>, Args...>(name);
}

template<typename OperationClass = Operation, typename ...Args>
void register_operation_class(const std::string& name) {
    register_operation<OperationClass, Args...>(name);
}

#define REGISTER_OPERATION_FUNCTION_WITH_BASE(N, FUNC, BASE, ...) static AddModuleInitializer CONCAT(__operation_add_, __COUNTER__)([]() { register_operation_function<decltype(&FUNC), &FUNC, BASE, ## __VA_ARGS__>( N ); })
#define REGISTER_OPERATION_FUNCTION(N, FUNC, ...) REGISTER_OPERATION_FUNCTION_WITH_BASE(N, FUNC, Operation, ## __VA_ARGS__)
#define REGISTER_OPERATION(N, O, ...) static AddModuleInitializer CONCAT(__operation_add_, __COUNTER__)([]() { register_operation_class<O, ## __VA_ARGS__>( N ); } )

}