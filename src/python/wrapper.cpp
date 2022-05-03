
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <pixelpipes/pipeline.hpp>
#include <pixelpipes/module.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/image.hpp>
#include <pixelpipes/python.hpp>

// Some direct NumPy hacks (TODO: replace with Pybind API when possible)
#include <numpy/arrayobject.h>

#include "../debug.h"


namespace py = pybind11;

using namespace pixelpipes;

// image.cpp
SharedToken wrap_image(py::object src);
py::object extract_image(SharedToken src);
SharedToken wrap_image_list(py::object src);
py::object extract_image_list(SharedToken src);


//const static int initialized = init_conversion();
void register_python_wrappers(Module& m);

class PIXELPIPES_INTERNAL PythonModuleImpl: public pixelpipes::PythonModule {
public:

    PythonModuleImpl() {};
    ~PythonModuleImpl() {};

    virtual py::object extract(const SharedToken &src) {

        VERIFY((bool) src, "Undefined value");

        TypeIdentifier type_id = src->type_id();

        auto item = extractors.find(type_id);

        if (item == extractors.end()) {
           throw std::invalid_argument(std::string("No conversion available: ") + src->describe());
        }

        return item->second(src);
    }

    virtual SharedToken wrap(py::object src, TypeIdentifier type_hint = 0) {

        if (!type_hint) {

            for (auto wrapper : allwrappers) {

                SharedToken v = wrapper(src);

                if (v) return v;

            }

            throw std::invalid_argument("Illegal input argument, no compatible conversion");

        } else {

            auto item = wrappers.find(type_hint);

            if (item == wrappers.end()) {
                throw std::invalid_argument(Formatter() << "No conversion from Python available for type hint " << type_name(type_hint));
            }

            SharedToken variable = item->second(src);

            if (!(bool) variable) {
                throw std::invalid_argument(Formatter() << "Conversion from Python failed for type hint " << type_name(type_hint));
            }

            return variable;

        }

    }

    virtual void register_wrapper(TypeIdentifier type_id, const PythonWrapper wrapper, bool implicit = true) {

        auto item = wrappers.find(type_id);

        if (item == wrappers.end()) {
            DEBUGMSG("Adding Python wrapper %s (%ld)\n", type_name(type_id).c_str(), type_id);
            wrappers.insert(std::pair<TypeIdentifier, const PythonWrapper>(type_id, wrapper));
            if (implicit) allwrappers.push_back(wrapper);
        }

    }

    virtual void register_extractor(TypeIdentifier type_id, const PythonExtractor extractor) {

        auto item = extractors.find(type_id);

        if (item == extractors.end()) {
            DEBUGMSG("Adding Python extractor %s (%ld)\n", type_name(type_id).c_str(), type_id);
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

    void done(TokenList result) override {
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

py::array numpyFromVariable(pixelpipes::SharedToken variable) {

    if (List::is(variable)) {
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
        //return py::array(handle, false);

        return py::reinterpret_steal<py::array>(handle);
    }

    if (variable->type_id() == pixelpipes::IntegerType) {
        py::array_t<int> a({1});
        a.mutable_data(0)[0] = std::static_pointer_cast<pixelpipes::Integer>(variable)->get();
        return a;
    }

    if (variable->type_id() == pixelpipes::FloatType) {
        py::array_t<float> a({1});
        a.mutable_data(0)[0] = std::static_pointer_cast<pixelpipes::Float>(variable)->get();
        return a;
    }

    if (variable->type_id() == pixelpipes::BooleanType) {
        py::array_t<int> a({1});
        a.mutable_data(0)[0] = (int) std::static_pointer_cast<pixelpipes::Boolean>(variable)->get();
        return a;
    }

    return registry.extract(variable);

}

SharedToken generic_convert(py::object src) {

    return registry.wrap(src);

    return empty();

}

template <typename T>
SharedToken wrap_list(py::object src) {


    if (py::list::check_(src)) {
        try {

            auto list = Sequence<T>(py::cast<std::vector<T>>(src));
            return std::make_shared<ContainerList<T>>(make_span(list));

        } catch(...) {}

    }
    return empty<List>();

}
/*
template <typename T>
SharedToken wrap_table(py::object src) {

    if (py::list::check_(src)) {
        try {

            auto data = py::cast<std::vector<std::vector<T>>>(src);
            return std::make_shared<Table<T>>(data);

        } catch(...) {}
    }

    return empty<Table<T>>();

}*/

SharedToken wrap_dnf_clause(py::object src) {

    try {

        auto original = py::cast<std::vector<std::vector<bool>>>(src);
        // TODO: make this nicer once we figure out how to convert nested types
        std::vector<Sequence<bool>> data;
        std::vector<Span<bool>> data2;

        for (auto x : original) {
            data.push_back(Sequence<bool>(x));
            data2.push_back(data.back());
        }

        DNF a(make_span(data2));

        return std::make_shared<ContainerToken<DNF>>(a);
 
    } catch(const std::exception &exc) {
        //DEBUGMSG("Conversion failed: %s\n", exc.what());
    }

    return empty();

}


template <typename T>
SharedToken wrap_container(py::object src) {

    try {

        auto object = py::cast<T>(src);

        DEBUGMSG("Conversion type: %s\n", type_name(GetTypeIdentifier<T>()));
        return std::make_shared<ContainerToken<T>>(object);
 
    } catch(const std::exception &exc) {
        //DEBUGMSG("Conversion failed: %s\n", exc.what());
    }

    

    return empty();

}

template<typename T>
int _add_operation(T& pipeline, std::string& name, py::list args, std::vector<int> inputs) {

    try {

    std::vector<SharedToken> arguments;

    OperationDescription type_hints = describe_operation(name);

    if (type_hints.arguments.size() != args.size())
        throw std::invalid_argument("Argument number mismatch");

    for (size_t i = 0; i < args.size(); i++) {
        arguments.push_back(registry.wrap(args[i], type_hints.arguments[i]));
    }

    return pipeline.append(name, make_span(arguments), make_span(inputs));

    } catch (BaseException& e) {
        throw py::value_error(e.what());
    }

}


PYBIND11_MODULE(pypixelpipes, m) {

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

    static py::exception<TypeException> PyVariableException(m, "VariableException", PyExc_RuntimeError);
    static py::exception<ModuleException> PyModuleException(m, "ModuleException", PyExc_RuntimeError);
    static py::exception<OperationException> PyOperationException(m, "OperationException", PyExc_RuntimeError);
    static py::exception<SerializationException> PySerializationException(m, "SerializationException", PyExc_RuntimeError);

    registry.register_enumeration<ComparisonOperation>("comparison");
    registry.register_enumeration<LogicalOperation>("logical");
    registry.register_enumeration<ArithmeticOperation>("arithmetic");
    registry.register_enumeration<ContextData>("context"); 

    registry.register_enumeration<ImageDepth>("depth"); 
    registry.register_enumeration<Interpolation>("interpolation"); 
    registry.register_enumeration<BorderStrategy>("border"); 

    py::class_<Pipeline, std::shared_ptr<Pipeline> >(m, "Pipeline")
    .def(py::init<>())
    .def("finalize", &Pipeline::finalize, "Finalize pipeline")
    .def("labels", &Pipeline::get_labels, "Get output labels as a list")
    .def("append", [](Pipeline& p, std::string& name, py::list args, std::vector<int> inputs) {

        return _add_operation(p, name, args, inputs);

    }, "Add operation to pipeline")
    .def("run", [](Pipeline& p, unsigned long index) {
        std::vector<SharedToken> result;

        { // release GIL lock when running pure C++, reacquire it when we are converting data
            py::gil_scoped_release gil; 
            auto tmp = p.run(index);
            result = std::vector(tmp.begin(), tmp.end());
        }

        std::vector<py::object> transformed;
        for (auto element : result) {
                transformed.push_back(numpyFromVariable(element));
        }

        return transformed; 

    }, "Run pipeline", py::arg("index"));


    py::class_<PipelineWriter, std::shared_ptr<PipelineWriter> >(m, "PipelineWriter")
    .def(py::init<>())
    .def("write", [](PipelineWriter& p, std::string filename, bool compress) { p.write(filename, compress); }, "Write the current pipeline", py::arg("filename"), py::arg("compress") = true)
    .def("append", [](PipelineWriter& p, std::string& name, py::list args, std::vector<int> inputs) {

        return _add_operation(p, name, args, inputs);

    }, "Add operation to the pipeline writer");
    
    py::class_<PipelineReader, std::shared_ptr<PipelineReader> >(m, "PipelineReader")
    .def(py::init<>())
    .def("read", [](PipelineReader& p, std::string filename) { return p.read(filename); }, "Read the pipeline");

    py::class_<PipelineCallback, PyPipelineCallback, std::shared_ptr<PipelineCallback> >(m, "PipelineCallback")
    .def(py::init());

    py::class_<Operation, SharedOperation >(m, "Operation");

    m.def("enum", [](std::string& name) {

        return registry.enumeration(name);

    });

    m.def("load", [](std::string& name) {

        return Module::load(name);

    });

    py::set_shared_data("_pixelpipes_python_registry", &registry);

    registry.register_wrapper(TokenType, generic_convert, false);

    registry.register_wrapper(IntegerType, [](py::object src) {

        if  (py::int_::check_(src)) {
            py::int_ value(src);
            return std::make_shared<Integer>(value);
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

    registry.register_wrapper(GetTypeIdentifier<Sequence<int>>(), &wrap_list<int>);
    registry.register_wrapper(GetTypeIdentifier<Sequence<float>>(), &wrap_list<float>);
    registry.register_wrapper(GetTypeIdentifier<Sequence<std::string>>(), &wrap_list<std::string>);
    registry.register_wrapper(GetTypeIdentifier<Sequence<bool>>(), &wrap_list<bool>);

    registry.register_wrapper(DNFType, &wrap_dnf_clause, false);

    registry.register_wrapper(Point2DType, [](py::object src) {

        if  (py::tuple::check_(src)) {
            py::tuple tuple(src);
            if (tuple.size() == 2) {
                py::float_ x(tuple[0]);
                py::float_ y(tuple[1]);

                return MAKE_POINT(x, y);
            }
        }

        return empty<Point2DVariable>();

    });

    registry.register_extractor(Point2DType, [](SharedToken src) {

        Point2D point = extract<Point2D>(src);
        py::array_t<float> result({2});
        *result.mutable_data(0) = point.x;
        *result.mutable_data(1) = point.y;
        return result;

    });

    registry.register_extractor(View2DType, [](SharedToken src) {

        View2D view = View2DVariable::get_value(src);

        py::array_t<float> result({3, 3});

        *result.mutable_data(0, 0) = view.m00;
        *result.mutable_data(0, 1) = view.m01;
        *result.mutable_data(0, 2) = view.m02;
        *result.mutable_data(1, 0) = view.m10;
        *result.mutable_data(1, 1) = view.m11;
        *result.mutable_data(1, 2) = view.m12;
        *result.mutable_data(2, 0) = view.m20;
        *result.mutable_data(2, 1) = view.m21;
        *result.mutable_data(2, 2) = view.m22;

        return result;

    });

    registry.register_wrapper(ImageType, &wrap_image);

    registry.register_extractor(ImageType, &extract_image);

    registry.register_wrapper(GetTypeIdentifier<Span<Image>>(), &wrap_image_list);

    registry.register_extractor(GetTypeIdentifier<Span<Image>>(), &extract_image_list);


}
