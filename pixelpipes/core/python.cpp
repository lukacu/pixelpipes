
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <pixelpipes/engine.hpp>
#include <pixelpipes/module.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/python.hpp>

#include <numpy/arrayobject.h>

namespace py = pybind11;

using namespace pixelpipes;

//const static int initialized = init_conversion();
void register_python_wrappers(Module& m);

class PIXELPIPES_INTERNAL PythonModuleImpl: public pixelpipes::PythonModule {
public:

    PythonModuleImpl() {};
    ~PythonModuleImpl() {};

    virtual py::object extract(const SharedVariable &src) {

        VERIFY((bool) src, "Undefined value");

        TypeIdentifier type_id = src->type();

        auto item = extractors.find(type_id);

        if (item == extractors.end()) {
           throw std::invalid_argument(std::string("No conversion available: ") + src->describe());
        }

        return item->second(src);
    }

    virtual SharedVariable wrap(py::object src, TypeIdentifier type_hint = 0) {

        if (!type_hint) {

            for (auto wrapper : allwrappers) {

                SharedVariable v = wrapper(src);

                if (v) return v;

            }

            throw std::invalid_argument("Unknown input argument");

        } else {

            auto item = wrappers.find(type_hint);

            if (item == wrappers.end()) {
                throw std::invalid_argument("No conversion available");
            }

            SharedVariable variable = item->second(src);

            if (!(bool) variable) {
                throw std::invalid_argument("Unable to convert variable");
            }

            return variable;

        }

    }

    virtual void register_wrapper(TypeIdentifier type_id, const PythonWrapper wrapper, bool implicit = true) {

        auto item = wrappers.find(type_id);

        if (item == wrappers.end()) {
            DEBUGMSG("Adding wrapper %p\n", type_id);
            wrappers.insert(std::pair<TypeIdentifier, const PythonWrapper>(type_id, wrapper));
            if (implicit) allwrappers.push_back(wrapper);
        }

    }

    virtual void register_extractor(TypeIdentifier type_id, const PythonExtractor extractor) {

        auto item = extractors.find(type_id);

        if (item == extractors.end()) {
            DEBUGMSG("Adding extractor %p\n", type_id);
            extractors.insert(std::pair<TypeIdentifier, const PythonExtractor>(type_id, extractor));
        }


    }

    virtual EnumerationMap enumeration(std::string& name) {

        auto item = enumerations.find(name);

        if (item == enumerations.end()) {

            throw std::invalid_argument("Unknown enumeration");
        }

        return item->second;

    }

    template <typename T>
    void register_enumeration(const std::string& name) {

        static_cast<PythonModule*>(this)->register_enumeration<T>(name);

    }

protected:

    virtual void _register_enumeration(const std::string& name, EnumerationMap mapping) {

        auto item = enumerations.find(name);

        if (item == enumerations.end()) {
            DEBUGMSG("Adding enumeration %s\n", name.c_str());
            enumerations.insert(std::pair<std::string, EnumerationMap>(name, mapping));
        }

    }

    std::map<TypeIdentifier, PythonWrapper> wrappers;

    std::vector<PythonWrapper> allwrappers;

    std::map<TypeIdentifier, PythonExtractor> extractors;

    std::map<std::string, PythonModule::EnumerationMap> enumerations;

};

static PythonModuleImpl registry;

class PyPipelineCallback : public PipelineCallback {
  public:
    using PipelineCallback::PipelineCallback;

    void done(std::vector<SharedVariable> result) override {
        PYBIND11_OVERLOAD_PURE(
            void,
            PipelineCallback,
            done,
            result
        );
    }

    void error(const PipelineException &e) override {
        PYBIND11_OVERLOAD_PURE(
            void,
            PipelineCallback,
            error,
            e
        );
    }

};

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

//static PyObject *PyPipelineError;

typedef void (*PythonInitalizerCallback) (PythonModule&);

py::array numpyFromVariable(pixelpipes::SharedVariable variable) {

    if (!variable->is_scalar()) {
        pixelpipes::SharedList list = std::static_pointer_cast<pixelpipes::List>(variable);
        if (!list->size()) {
            return py::array();
        }

        py::list planes(list->size());
      
        for (size_t i = 0; i < list->size(); i++) {
            py::array t = numpyFromVariable(list->get(i));

            std::vector<ssize_t> shape(t.ndim() + 1);
            shape[0] = 1;
            for (int i = 0; i < t.ndim(); i++) shape[i+1] = t.shape(i);
            t.resize(shape);

            planes[i] = t;
        }
 
        auto handle = py::handle(PyArray_Concatenate(planes.ptr(), 0));
        return py::array(handle, false);

    }

    if (variable->type() == pixelpipes::IntegerType) {
        py::array_t<int> a({1});
        a.mutable_data(0)[0] = std::static_pointer_cast<pixelpipes::Integer>(variable)->get();
        return a;
    }

    if (variable->type() == pixelpipes::FloatType) {
        py::array_t<float> a({1});
        a.mutable_data(0)[0] = std::static_pointer_cast<pixelpipes::Float>(variable)->get();
        return a;
    }

    if (variable->type() == pixelpipes::BooleanType) {
        py::array_t<int> a({1});
        a.mutable_data(0)[0] = (int) std::static_pointer_cast<pixelpipes::Boolean>(variable)->get();
        return a;
    }

    return registry.extract(variable);

}

SharedVariable generic_convert(py::object src) {

    return registry.wrap(src);

    return empty();

}

SharedOperation make_operation_python(std::string& name, py::args args) {

    std::vector<SharedVariable> inputs;

    OperationDescription type_hints = describe_operation(name);

    if (type_hints.size() != args.size())
        throw std::invalid_argument("Argument number mismatch");

    for (size_t i = 0; i < args.size(); i++) {
        inputs.push_back(registry.wrap(args[i], type_hints[i]));
    }

    return create_operation(name, inputs);
}

template <typename T>
SharedVariable wrap_list(py::object src) {


    if (py::list::check_(src)) {
        try {

            auto list = py::cast<std::vector<T>>(src);
            return std::make_shared<ContainerList<T>>(list);

        } catch(...) {}

    }
    return empty<List>();

}

template <typename T>
SharedVariable wrap_table(py::object src) {

    if (py::list::check_(src)) {
        try {

            auto data = py::cast<std::vector<std::vector<T>>>(src);
            return std::make_shared<Table<T>>(data);

        } catch(...) {}
    }

    return empty<Table<T>>();

}

SharedVariable wrap_dnf_clause(py::object src) {

    try {

        DNF form;
        form.clauses = py::cast<std::vector<std::vector<bool>>>(src);

        return std::make_shared<ContainerVariable<DNF>>(form);
 
    } catch(const std::exception &exc) {
        //DEBUGMSG("Conversion failed: %s\n", exc.what());
    }

    return empty();

}


template <typename T>
SharedVariable wrap_container(py::object src) {

    try {

        auto object = py::cast<T>(src);

        DEBUGMSG("Conversion type: %s\n", VIEWCHARS(Type<T>::name));
        return std::make_shared<ContainerVariable<T>>(object);
 
    } catch(const std::exception &exc) {
        //DEBUGMSG("Conversion failed: %s\n", exc.what());
    }

    

    return empty();

}


PYBIND11_MODULE(pp_py, m) {

    if (_import_array() < 0) {
        throw py::import_error("Unable to load NumPy");
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

    static py::exception<PipelineException> PyPipelineError(m, "PipelineException", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (PipelineException &e) {
            py::tuple args(2);
            args[0] = e.what();
            args[1] = e.operation();
            // TODO: also pass some hint about operation
            PyPipelineError(e.what());
        }
    });

    static py::exception<VariableException> PyVariableException(m, "VariableException", PyExc_RuntimeError);
    static py::exception<ModuleException> PyModuleException(m, "ModuleException", PyExc_RuntimeError);

    registry.register_enumeration<ComparisonOperation>("comparison");
    registry.register_enumeration<LogicalOperation>("logical");
    registry.register_enumeration<ArithmeticOperation>("arithmetic");
    registry.register_enumeration<ContextData>("context"); 

    py::class_<Pipeline, std::shared_ptr<Pipeline> >(m, "Pipeline")
    .def(py::init<>())
    .def("finalize", &Pipeline::finalize, "Finalize pipeline")
    .def("append", &Pipeline::append, "Add operation to pipeline")
    .def("operation_time", &Pipeline::operation_time, "Get operation statistics")
    .def("run", [](Pipeline& p, unsigned long index) {
        std::vector<SharedVariable> result;

        { // release GIL lock when running pure C++, reacquire it when we are converting data
            py::gil_scoped_release gil; 
            result = p.run(index);
        }

        std::vector<py::object> transformed;
        for (auto element : result) {
                transformed.push_back(numpyFromVariable(element));
        }

        return transformed; 

    }, "Run pipeline", py::arg("index"));



    py::class_<PipelineCallback, PyPipelineCallback, std::shared_ptr<PipelineCallback> >(m, "PipelineCallback")
    .def(py::init());

    py::class_<Operation, SharedOperation >(m, "Operation");

    m.def("make", make_operation_python);

    m.def("enum", [](std::string& name) {

        return registry.enumeration(name);

    });


    m.def("load", [](std::string& name) {

        return Module::load(name);

    });

    py::set_shared_data("_pp_python_registry", &registry);

    registry.register_wrapper(VariableType, generic_convert, false);

    registry.register_wrapper(IntegerType, [](py::object src) {

        if  (py::int_::check_(src)) {
            py::int_ value(src);
            return  std::make_shared<Integer>(value);
        }

        return empty<Integer>();
 
    });

    registry.register_wrapper(FloatType, [](py::object src) {

        if  (py::float_::check_(src)) {
            py::float_ value(src);
            return std::make_shared<Float>(value);
        }

        if  (py::int_::check_(src)) {
            int value = (int) py::int_(src);
            return std::make_shared<Float>(value);
        }

        return empty<Float>();

    }); 

    registry.register_wrapper(StringType, [](py::object src) {

        if  (py::str::check_(src)) {
            py::str str(src);
            return std::make_shared<String>(str);
        }

        return empty<String>();

    });

    registry.register_wrapper(BooleanType, [](py::object src) {

        if  (py::bool_::check_(src)) {
            py::bool_ value(src);
            return std::make_shared<Boolean>(value);
        }

        return empty<Boolean>();

    });

    registry.register_wrapper(IntegerListType, &wrap_list<int>);
    registry.register_wrapper(FloatListType, &wrap_list<float>);
    registry.register_wrapper(StringListType, &wrap_list<std::string>);
    registry.register_wrapper(BooleanListType, &wrap_list<bool>);

    registry.register_wrapper(IntegerTableType, &wrap_table<int>);
    registry.register_wrapper(FloatTableType, &wrap_table<float>);
    registry.register_wrapper(StringTableType, &wrap_table<std::string>);
    registry.register_wrapper(BooleanTableType, &wrap_table<bool>);

    registry.register_wrapper(DNFType, &wrap_dnf_clause, false);

}
