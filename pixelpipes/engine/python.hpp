#pragma once
#include <pybind11/pybind11.h>
#include <functional>
#include <list>
#include <tuple>

#include "engine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pixelpipes {


//template<typename F, typename ...Args> 
//using OperationFunction = SharedVariable (*F) (std::vector<SharedVariable>, ContextHandle, Args...);

//template<typename ...Args>
//class OperationFunction<SharedVariable(std::vector<SharedVariable>, ContextHandle, Args...)> { }

template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) apply_impl(F&& f, std::vector<SharedVariable> inputs, ContextHandle context, Tuple&& t, std::index_sequence<I...>)
{
    return std::invoke(std::forward<F>(f), inputs, context, std::get<I>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, std::vector<SharedVariable> inputs, ContextHandle context, Tuple&& t)
{
    return apply_impl(
        std::forward<F>(f), inputs, context, std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

template<typename Fn, Fn fn, typename Base, typename ...Args>
class OperationWrapper: public Base {
public:

    OperationWrapper(Args&&... args) : args(std::forward<Args>(args)...) {
        static_assert(std::is_base_of<Operation, Base>::value, "Base class must inherit from Operation");
    };
    ~OperationWrapper() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) {

        try {

            return apply(std::function(fn), std::tuple_cat(std::make_tuple(inputs, context), args) );
            //return apply(std::forward<Fn>(fn), inputs, context, args);
            //return fn(inputs, context, std::forward<Args>(args)...);

        } catch (VariableException &e) {
            throw OperationException(e.what(), this->shared_from_this());
        }

    }
    
protected:

    std::tuple<Args...> args;

};

template<typename Fn, Fn fn, typename Base, typename ...Args>
void _register_operation(py::module &module, const std::string name) {

    typedef OperationWrapper<Fn, fn, Base, Args...> wrapper_type;

    py::class_<wrapper_type, Operation, std::shared_ptr<wrapper_type> >(module, name.c_str())
        .def(py::init<Args...>());

}

}

/*
typename std::result_of<Fn(Args...)>::type
OperationWrapper(Args&&... args) {
    return fn(std::forward<Args>(args)...);
}
#define WRAPPER(FUNC) wrapper<decltype(&FUNC), &FUNC>
*/
class operation_initializer {
public:
    operation_initializer(std::function<void(py::module &)> initializer);
};

#define REGISTER_OPERATION_FUNCTION_WITH_BASE(FUNC, BASE, ...) operation_initializer FUNC ##_initialization([](py::module &module) { pixelpipes::_register_operation<decltype(&FUNC), &FUNC, BASE, ## __VA_ARGS__>(module, #FUNC); })

#define REGISTER_OPERATION_FUNCTION(FUNC, ...) REGISTER_OPERATION_FUNCTION_WITH_BASE(FUNC, Operation, ## __VA_ARGS__)

#define ADD_OPERATION(O, ...) {py::class_<O, Operation, std::shared_ptr<O> >(m, #O).def(py::init< __VA_ARGS__ >());}

#define REGISTER_OPERATION(O, ...) operation_initializer O ##_initialization([](py::module &m) { ADD_OPERATION(O, ## __VA_ARGS__); });

