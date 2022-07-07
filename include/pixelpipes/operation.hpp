#pragma once

#include <type_traits>
#include <memory>
#include <functional>
#include <list>
#include <tuple>
#include <map>

#include <pixelpipes/token.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes
{

    enum class ComparisonOperation
    {
        EQUAL,
        LOWER,
        LOWER_EQUAL,
        GREATER,
        GREATER_EQUAL,
        NOT_EQUAL
    };
    enum class LogicalOperation
    {
        AND,
        OR,
        NOT
    };
    enum class ArithmeticOperation
    {
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        POWER,
        MODULO
    };

    enum class SamplingDistribution
    {
        Normal,
        Uniform
    };

    enum class ContextData
    {
        SampleIndex,
        OperationIndex,
        RandomSeed
    };

    PIXELPIPES_CONVERT_ENUM(ContextData)
    PIXELPIPES_CONVERT_ENUM(SamplingDistribution)

    PIXELPIPES_CONVERT_ENUM(ArithmeticOperation)
    PIXELPIPES_CONVERT_ENUM(LogicalOperation)
    PIXELPIPES_CONVERT_ENUM(ComparisonOperation)

    typedef View<TokenReference> TokenList;

    class Operation;

    typedef Pointer<Operation> OperationReference;

    RandomGenerator make_generator(uint32_t seed);

    RandomGenerator create_generator(TokenReference seed);

    RandomGenerator create_generator(int seed);

    class PIXELPIPES_API Operation
    {
    public:
        ~Operation() = default;

        virtual TokenReference run(const TokenList& inputs) = 0;

        virtual TypeIdentifier type();

        template <typename T>
        bool is()
        {
            return type() == GetTypeIdentifier<T>();
        }

        virtual Sequence<TokenReference> serialize() = 0;

    protected:
        Operation();
    };

    namespace details
    {

        template<typename T>
        using base_type = typename std::remove_cv<typename std::remove_reference<T>::type>;

        template<typename T>
        using base_type_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

        template <typename... T>
        using tuple_with_base_types = std::tuple<typename base_type<T>::type...>;

        template <typename... T>
        tuple_with_base_types<T...> tuple_base_type(std::tuple<T...> const& t) {
            return tuple_with_base_types<T...> { t };
        }

        template <typename>
        struct function_traits;

        template <typename R, typename... Args>
        struct function_traits<R(Args...)>
        {
            using output = R;
            using inputs = std::tuple<Args ...>;
            constexpr static size_t arity = sizeof...(Args);
        };

        template<class R, class... Args>
        struct function_traits<R(*)(Args...)> : public function_traits<R(Args...)>
        {};

        template<typename R, typename ...Args> 
        struct function_traits<std::function<R(Args...)>>
        {
            typedef R output;
            typedef std::tuple<Args ...> inputs;
            constexpr static size_t arity = sizeof...(Args);
        };

        template <class Tuple,
                  class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
        std::vector<T> to_vector(Tuple &&tuple)
        {
            return std::apply([](auto &&...elems)
                              { return std::vector<T>{std::forward<decltype(elems)>(elems)...}; },
                              std::forward<Tuple>(tuple));
        }

        template <class F, class Tuple, std::size_t... I>
        constexpr decltype(auto) apply_impl(F &&f, TokenList inputs,
                                            Tuple t, std::index_sequence<I...>)
        {
            UNUSED(t); // don't know why this causes unused error?
            return std::invoke(std::forward<F>(f), inputs, std::get<I>(std::forward<Tuple>(t))...);
        }

        template <class F, class Tuple>
        constexpr decltype(auto) apply(F &&f, TokenList inputs, Tuple t)
        {
            return apply_impl(
                std::forward<F>(f), inputs, std::forward<Tuple>(t),
                std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
        }

        template <class T, class Tuple, std::size_t... I>
        constexpr Pointer<T> make_from_tuple_impl(Tuple &&t, std::index_sequence<I...>)
        {
            return create<T>(std::get<I>(std::forward<Tuple>(t))...);
        }

        template <class T, class Tuple>
        constexpr Pointer<T> make_from_tuple(Tuple &&t)
        {
            return make_from_tuple_impl<T>(std::forward<Tuple>(t),
                                           std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
        }

        template <size_t... Indices>
        struct indices
        {
            using next = indices<Indices..., sizeof...(Indices)>;
        };

        template <typename FuncType,
                  typename VecType,
                  size_t... I,
                  typename Traits = function_traits<FuncType>,
                  typename ReturnT = typename Traits::output>
        ReturnT do_call(FuncType func,
                        VecType &args,
                        std::index_sequence<I...>)
        {
            VERIFY(args.size() >= Traits::arity, "Argument size mismatch");
            return func(args[I]...);
        }

        template <typename FuncType,
                  typename VecType,
                  typename Traits = function_traits<FuncType>,
                  typename ReturnT = typename Traits::output>
        ReturnT unpack_caller(FuncType func,
                              VecType &args)
        {
            return do_call(func, args, std::make_index_sequence<Traits::arity>{});
        }

        typedef TokenList::const_iterator ArgIterator;

        template <typename A>
        std::tuple<> extract_args(ArgIterator current, ArgIterator end)
        {
            if (current != end)
                throw TypeException("Number of inputs does not match");

            return std::make_tuple();
        }

        // base template with 1 argument (which will be called from the variadic one).
        template <typename A, typename Arg>
        tuple_with_base_types<Arg> extract_args(ArgIterator current, ArgIterator end)
        {

            if (current == end)
                throw TypeException("Number of inputs does not match");

            return std::tuple<base_type_t<Arg>>(extract<base_type_t<Arg>>(*current));
        }

        template <typename A, typename First, typename Second, typename... Args>
        tuple_with_base_types<First, Second, Args...> extract_args(ArgIterator current, ArgIterator end)
        {
            if (current == end)
                throw TypeException("Number of inputs does not match");

            return std::tuple_cat(extract_args<A, First>(current, end), extract_args<A, Second, Args...>(current + 1, end));
        }

        template <typename A, typename X>
        struct extract_args_tuple {};

        template <class A, typename... Args>
        struct extract_args_tuple<A, std::tuple<Args ...>>
        {
            static tuple_with_base_types<Args...> run(ArgIterator begin, ArgIterator end) {
                return extract_args<A, Args...>(begin, end);
            }
        };

    }

    template <typename Run, Run fn_run, bool unpack>
    class OperationWrapper : public Operation
    {
        using OperationType = OperationWrapper<Run, fn_run, unpack>;
        using ArgTypes = typename details::function_traits<Run>::inputs;
        using OutputType = typename details::function_traits<Run>::output;

    public:
        OperationWrapper() = default;
        ~OperationWrapper() = default;

        virtual TokenReference run(const TokenList& inputs)
        {
            if constexpr (unpack) {
                auto converted = details::extract_args_tuple<OperationType, ArgTypes>::run(inputs.begin(), inputs.end());
                if constexpr (std::is_base_of_v<TokenReference, OutputType>)
                    return wrap(std::apply(std::function(fn_run), converted));
                else {
                    auto result = std::apply(std::function(fn_run), converted);
                    return wrap(result);
                }
            } else {
                return fn_run(inputs);
            }

        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<OperationType>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>(); }

    };

    typedef Sequence<TypeIdentifier> OperationArguments;

    struct OperationDescription
    {
        TypeIdentifier identifier;
        OperationArguments arguments;
    };

    template <typename OperationClass = Operation, typename... Args>
    struct OperationFactory
    {

        static OperationDescription describe()
        {
            return OperationDescription{GetTypeIdentifier<OperationClass>(), OperationArguments{GetTypeIdentifier<Args>()...}};
        }

        static OperationReference new_instance(TokenList inputs)
        {
            auto converted = details::extract_args<OperationClass, Args...>(inputs.begin(), inputs.end());
            auto op = details::make_from_tuple<OperationClass>(converted);
            return op;
        }
    };

    typedef Function<OperationReference(TokenList)> OperationConstructor;
    typedef Function<OperationDescription()> OperationDescriber;

    OperationReference PIXELPIPES_API make_operation(const std::string key, const TokenList& inputs);
    void PIXELPIPES_API register_operation(const std::string key, OperationConstructor constructor, OperationDescriber describer);
    bool PIXELPIPES_API is_operation_registered(const std::string key);

    template <typename OperationClass = Operation, typename... Args>
    void register_operation(const std::string key)
    {
        register_operation(key, OperationFactory<OperationClass, Args...>::new_instance, OperationFactory<OperationClass, Args...>::describe);
    }

    template <typename... Args>
    OperationReference make_operation(const std::string key, Args &&...args)
    {
        return make_operation(key, TokenList({args...}));
    }

    OperationDescription PIXELPIPES_API describe_operation(const std::string key);
    ModuleReference PIXELPIPES_API operation_source(const std::string key);
    OperationReference PIXELPIPES_API create_operation(const std::string key, const TokenList& inputs);
    OperationReference PIXELPIPES_API create_operation(const std::string key, const std::initializer_list<TokenReference>& inputs);
    std::string PIXELPIPES_API operation_name(const OperationReference&);

    template <typename Run, Run fn_run>
    void register_operation_auto(const std::string &name)
    {
        register_operation<OperationWrapper<Run, fn_run, true>>(name);
    }

    template <typename Run, Run fn_run>
    void register_operation_manual(const std::string &name)
    {
        register_operation<OperationWrapper<Run, fn_run, false>>(name);
    }

    template <typename Operation, typename... Args>
    void register_operation_class(const std::string &name)
    {
        register_operation<Operation, Args...>(name);
    }

#define PIXELPIPES_OPERATION_AUTO(NAME, RUN) static AddModuleInitializer CONCAT(__operation_add_, __COUNTER__)([]() { register_operation_auto<decltype(&(RUN)), (RUN)>(NAME); })
#define PIXELPIPES_OPERATION(NAME, RUN) static AddModuleInitializer CONCAT(__operation_add_, __COUNTER__)([]() { register_operation_manual<decltype(&(RUN)), (RUN)>(NAME); })
#define PIXELPIPES_OPERATION_CLASS(N, ...) static AddModuleInitializer CONCAT(__operation_add_, __COUNTER__)([]() { register_operation_class<__VA_ARGS__>( N ); })

}
