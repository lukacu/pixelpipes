#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <numpy/arrayobject.h>

#include <pixelpipes/engine.hpp>
#include <pixelpipes/types.hpp>
#include <pixelpipes/enum.hpp>

namespace py = pybind11;

/*
class PyAllowThreads;
#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}*/

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

int init_conversion();

class NumpyAllocator;

//pixelpipes::SharedVariable variableFromPython(py::handle src);
//py::array numpyFromVariable(pixelpipes::SharedVariable variable);

namespace pixelpipes {

typedef std::function<py::object(SharedVariable)> PythonExtractor;
typedef std::function<SharedVariable(py::object)> PythonWrapper;

class PythonModule {
public:
    typedef std::map<std::string, int> EnumerationMap;

    virtual py::object extract(const SharedVariable &src) = 0;

    virtual SharedVariable wrap(py::object src, TypeIdentifier type_hint = 0) = 0;

    virtual void register_wrapper(TypeIdentifier type_id, const PythonWrapper wrapper, bool implicit = true) = 0;

    virtual void register_extractor(TypeIdentifier type_id, const PythonExtractor extractor) = 0;

    virtual EnumerationMap enumeration(std::string& name) = 0;

    template <typename T>
    void register_enumeration(const std::string& name) {

        auto pairs = detail::enum_entries<T>();

        EnumerationMap mapping;

        for (auto pair: pairs) {
            mapping.insert(EnumerationMap::value_type(pair.second, (int) pair.first));
        }

        _register_enumeration(name, mapping);

        register_wrapper(Type<T>::identifier, enum_wrapper);

    };

protected:

    virtual void _register_enumeration(const std::string& name, EnumerationMap mapping) = 0;

    static SharedVariable enum_wrapper(py::object src) {

        if  (py::int_::check_(src)) {
            py::int_ value(src);
            return std::make_shared<Integer>(value);
        } 
        
        return empty<Integer>();

    }

};

typedef std::function<void(PythonModule&)> ModuleInitializer;

#define PIXELPIPES_PYTHON_MODULE(N) \
class _PythonInitalizerAnchor { public: _PythonInitalizerAnchor(ModuleInitializer initializer); }; \
std::list<ModuleInitializer> & _python_module_initializers() { static std::list<ModuleInitializer> inits;  return inits; } \
_PythonInitalizerAnchor::_PythonInitalizerAnchor(ModuleInitializer initializer) { (_python_module_initializers)().push_back(initializer); } \
PYBIND11_MODULE(N, m) { \
m.doc() = STRINGIFY(N); \
if (_import_array() < 0) { throw py::import_error("Unable to load NumPy"); } \
auto registry = reinterpret_cast<pixelpipes::PythonModule *>(py::get_shared_data("_pp_python_registry")); \
if (!registry) return; \
for (auto initializer : (pixelpipes::_python_module_initializers)()) initializer(*registry); \
}

#define PIXELPIPES_PYTHON_REGISTER_ENUM(N, E) _PythonInitalizerAnchor E ##_enum_reg([](PythonModule& module) { module.register_enumeration<E>(N); })
#define PIXELPIPES_PYTHON_REGISTER_WRAPPER(T, W) _PythonInitalizerAnchor T ##_wrapper_reg([](PythonModule& module) { module.register_wrapper(T, W); })
#define PIXELPIPES_PYTHON_REGISTER_EXTRACTOR(T, E) _PythonInitalizerAnchor T ##_extractor_reg([](PythonModule& module) { module.register_extractor(T, E); })

}

