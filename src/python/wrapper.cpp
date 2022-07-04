
#include <fstream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <pixelpipes/pipeline.hpp>
#include <pixelpipes/module.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/tensor.hpp>

// Some direct NumPy hacks (TODO: replace with Pybind API when possible)
#include <numpy/arrayobject.h>

#include "../debug.h"

namespace py = pybind11;

using namespace pixelpipes;

PYBIND11_DECLARE_HOLDER_TYPE(T, pixelpipes::Pointer<T>)

// array.cpp
TokenReference wrap_tensor(const py::object &src);
py::object extract_tensor(const TokenReference &src);
//TokenReference wrap_tensor_list(const py::object &src);

/*class PyPipelineCallback : public PipelineCallback
{
public:
    using PipelineCallback::PipelineCallback;

    void done(TokenList result) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            PipelineCallback,
            done,
            result);
    }

    void error(const PipelineException &e) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            PipelineCallback,
            error,
            e);
    }
};*/

// Solution based on: https://www.pierov.org/2020/03/01/python-custom-exceptions-c-extensions/
/*static PyObject *PipelineError_tp_str(PyObject *selfPtr)
{
    py::str ret;
    try {
        py::handle self(selfPtr);
        py::tuple args = self.attr("args");
        ret = py::str(args[0]);
    } catch (py::error_already_set &e) {
        ret = "";
    }

    ret.inc_ref();
    return ret.ptr();
}*/
/*
static PyObject *PipelineError_getoperation(PyObject *selfPtr, void *closure) {
    try {
        py::handle self(selfPtr);
        py::tuple args = self.attr("args");
        py::object code = args[1];
        code.inc_ref();
        return code.ptr();
    } catch (py::error_already_set &e) {
        // We could simply backpropagate the exception with e.restore, but
        // exceptions like OSError return None when an attribute is not set.
        py::none ret;
        ret.inc_ref();
        return ret.ptr();
    }
}

static PyGetSetDef PipelineError_getsetters[] = {
    {"operation", PipelineError_getoperation, NULL, NULL, NULL},
    {NULL}
};*/

// static PyObject *PyPipelineError;

py::array token_to_python(const pixelpipes::TokenReference &variable)
{
    if (!variable)
        throw py::value_error("Undefined result token");

    if (variable->is<String>())
    {
        py::str s = cast<String>(variable)->get();
        return s;
    }

    if (variable->is<IntegerScalar>())
    {
        py::array_t<int> a(1);
        a.mutable_data(0)[0] = cast<IntegerScalar>(variable)->get();
        return a;
    }

    if (variable->is<FloatScalar>())
    {
        py::array_t<float> a(1);
        a.mutable_data(0)[0] = cast<FloatScalar>(variable)->get();
        return a;
    }

    if (variable->is<BooleanScalar>())
    {
        py::array_t<int> a(1);
        a.mutable_data(0)[0] = (int)cast<BooleanScalar>(variable)->get();
        return a;
    }

    if (variable->is<Tensor>())
    {
        return extract_tensor(variable);
    };

    if (variable->is<List>())
    {

        ListReference list = extract<ListReference>(variable);

        if (!list->length())
        {
            return py::array();
        }

        py::list planes(list->length());

        for (size_t i = 0; i < list->length(); i++)
        {
            py::array t = token_to_python(list->get(i));

            std::vector<ssize_t> shape(t.ndim() + 1);
            shape[0] = 1;
            for (int i = 0; i < t.ndim(); i++)
                shape[i + 1] = t.shape(i);
            t.resize(shape);

            planes[i] = t;
        }

        auto handle = py::handle(PyArray_Concatenate(planes.ptr(), 0));
        // return py::array(handle, false);

        return py::reinterpret_steal<py::array>(handle);
    }

    throw py::value_error(Formatter() << "Unable to convert token to Python:"  << variable);
}

TokenReference wrap_dnf_clause(py::object src)
{

    try
    {
        auto original = py::cast<std::vector<std::vector<bool>>>(src);
        // TODO: make this nicer once we figure out how to convert nested types
        std::vector<Sequence<bool>> data;
        std::vector<Span<bool>> data2;

        for (auto x : original)
        {
            data.push_back(Sequence<bool>(x));
            data2.push_back(data.back());
        }

        DNF a(make_span(data2));

        return create<ContainerToken<DNF>>(a);
    }
    catch (const std::exception &exc)
    {
    }

    return empty();
}

template <typename T, typename C>
TokenReference _python_list_convert_strict(const py::list &list)
{

    if (list.size() == 0)
    {
        return empty();
    }

    if (!C::check_(list[0]))
    {
        return empty();
    }

    auto clist = Sequence<T>(list.size());

    for (size_t i = 0; i < list.size(); i++)
    {
        if (!C::check_(list[i]))
            return empty();

        clist[i] = py::cast<T>(list[i]);
    }

    return wrap(clist);
}

#define _CONVERT_VECTOR(src, elem)                                     \
    try                                                                \
    {                                                                  \
        auto list = Sequence<elem>(py::cast < std::vector<elem>>(src)); \
                                                                       \
        return create<Vector<elem>>(make_span(list));                  \
    }                                                                  \
    catch (...)                                                        \
    {                                                                  \
    }

TokenReference python_to_token(py::object src)
{
    if (py::bool_::check_(src))
    {
        py::bool_ value(src);
        return create<BooleanScalar>(value);
    }

    if (py::int_::check_(src))
    {
        py::int_ value(src);
        return create<IntegerScalar>(value);
    }

    if (py::float_::check_(src))
    {
        py::float_ value(src);
        return create<FloatScalar>(value);
    }

    if (py::str::check_(src))
    {
        py::str str(src);
        return create<String>(str);
    }

    if (py::list::check_(src))
    {
        auto pylist = py::list(src);
        TokenReference r = _python_list_convert_strict<bool, py::bool_>(pylist);
        if (r) {
            return r;
        }

        _CONVERT_VECTOR(src, char);
        _CONVERT_VECTOR(src, int);
        _CONVERT_VECTOR(src, float);

        try
        {
            auto list = Sequence<std::string>(py::cast<std::vector<std::string>>(src));
            return create<StringList>(make_span(list));
        }
        catch (...)
        {
        }
    }

    if (py::array::check_(src))
    {
        return wrap_tensor(src);
    }

    // Temporary solution, remove when predictive optimization is moved to C++
    TokenReference dnf = wrap_dnf_clause(src);
    if ((bool)dnf)
        return dnf;

    if (py::list::check_(src))
    {
        auto list = py::list(src);

        Sequence<TokenReference> converted(list.size());

        for (size_t i = 0; i < list.size(); i++)
        {
            converted[i] = python_to_token(list[i]);
        }

        return create<GenericList>(converted);
    }

    throw py::value_error("Unable to convert Python data");
}

template <typename T>
int _add_operation(T &pipeline, std::string &name, py::list args, std::vector<int> inputs)
{

    try
    {

        std::vector<TokenReference> arguments;

        OperationDescription type_hints = describe_operation(name);

        if (type_hints.arguments.size() != args.size())
            throw std::invalid_argument("Argument number mismatch");

        for (size_t i = 0; i < args.size(); i++)
        {
            arguments.push_back(python_to_token(args[i]));
        }

        return pipeline.append(name, make_span(arguments), make_span(inputs));
    }
    catch (BaseException &e)
    {
        throw py::value_error(e.what());
    }
}

PYBIND11_MODULE(pypixelpipes, m)
{

    if (_import_array() < 0)
    {
        throw py::error_already_set();
    }

    m.doc() = "Python Wrapper for PixelPipes Engine";
    /*
        PyPipelineError = PyErr_NewException("PipelineError", NULL, NULL);
        if (PyPipelineError) {
            PyTypeObject *as_type = reinterpret_cast<PyTypeObject *>(PyPipelineError);
            as_type->tp_str = PipelineError_tp_str;
            PyObject *descr = PyDescr_NewGetSet(as_type, PipelineError_getsetters);
            auto dict = py::reinterpret_borrow<py::dict>(as_type->tp_dict);
            dict[py::handle(PyDescr_NAME(descr))] = py::handle(descr);

            Py_XINCREF(PyPipelineError);
            m.add_object("PipelineError", py::handle(PyPipelineError));
        }
    */
    /*
        static py::exception<PipelineException> PyPipelineError(m, "PipelineException", PyExc_RuntimeError);
        py::register_exception_translator([](std::exception_ptr p)
                                          {
            try {
                if (p) {
                    std::rethrow_exception(p);
                }
            } catch (PipelineException &e) {
                py::tuple args(2);
                args[0] = e.what();
                args[1] = e.operation();
                // TODO: also pass some hint about operation?
                PyPipelineError(e.what());
            } });*/
    /*
        static py::exception<TypeException> PyVariableException(m, "VariableException", PyExc_RuntimeError);
        static py::exception<ModuleException> PyModuleException(m, "ModuleException", PyExc_RuntimeError);
        static py::exception<OperationException> PyOperationException(m, "OperationException", PyExc_RuntimeError);
        static py::exception<SerializationException> PySerializationException(m, "SerializationException", PyExc_RuntimeError);
    */
    py::register_exception<SerializationException>(m, "SerializationException");
    py::register_exception<OperationException>(m, "OperationException");
    py::register_exception<TypeException>(m, "TypeException");
    py::register_exception<ModuleException>(m, "ModuleException");
    py::register_exception<IllegalStateException>(m, "IllegalStateException");
    py::register_exception<PipelineException>(m, "PipelineException");

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<>())
        .def("finalize", &Pipeline::finalize, "Finalize pipeline")
        .def(
            "labels", [](Pipeline &p)
            {
                auto labels = p.get_labels();
                return std::vector<std::string>(labels.begin(), labels.end()); },
            "Get output labels as a list")
        .def(
            "append", [](Pipeline &p, std::string &name, py::list args, std::vector<int> inputs)
            { return _add_operation(p, name, args, inputs); },
            "Add operation to pipeline")
        .def(
            "run", [](Pipeline &p, unsigned long index)
            {
                Sequence<TokenReference> result;

                { // release GIL lock when running pure C++, reacquire it when we are converting data
                    py::gil_scoped_release gil;
                    result = p.run(index);
                }

                py::tuple transformed(result.size());
                size_t i = 0;
                for (auto element = result.begin(); element != result.end(); element++, i++) {
                        transformed[i] = (token_to_python(*element));
                }
                return transformed; },
            "Run pipeline", py::arg("index"));

    m.def(
        "read_pipeline", [](std::string &name)
        { return read_pipeline(name); },
        py::arg("filename"));

    m.def(
        "write_pipeline", [](const Pipeline &pipeline, std::string &name, bool compress)
        { return write_pipeline(pipeline, name, compress); },
        py::arg("pipeline"), py::arg("filename"), py::arg("compress") = true);

    /*py::class_<PipelineCallback, PyPipelineCallback, std::shared_ptr<PipelineCallback>>(m, "PipelineCallback")
        .def(py::init());*/

    // py::class_<Operation, OperationReference >(m, "Operation");

    m.def("enum", [](std::string &name)
          { return describe_enumeration(name); });

    m.def("load", [](std::string &name)
          { return Module::load(name); });
}
