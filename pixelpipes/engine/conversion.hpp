#pragma once

//#define NUMPY_IMPORT_ARRAY_RETVAL

#include <Python.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "engine.hpp"
#include "types.hpp"

namespace py = pybind11;

#ifdef __PP_PYTORCH
#include <torch/torch.h>

torch::Tensor tensorFromMat(cv::Mat &mat);
py::object torchFromVariable(pixelpipes::SharedVariable variable);

#endif

py::object numpyFromVariable(pixelpipes::SharedVariable variable);

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
}

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

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

class PYBIND11_EXPORT NDArrayConverter
{
public:
    NDArrayConverter();
    cv::Mat toMat(PyObject* o);
    PyObject* toNDArray(const cv::Mat& mat);
};

bool convert_numpy(py::handle src, cv::Mat *value);

namespace pybind11 {
namespace detail {

template <> class type_caster<cv::Mat> {
    typedef cv::Mat type;
public:
    bool load(py::handle src, bool) {
        return convert_numpy(src, &value);
    }
    static py::handle cast(const cv::Mat &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        NDArrayConverter cvt;
        py::handle result(cvt.toNDArray(src));
        return result;
    }
    PYBIND11_TYPE_CASTER(cv::Mat, _("cv::Mat"));
};

template <> class type_caster<cv::Point2f> {
    typedef cv::Point2f type;
public:
    bool load(py::handle src, bool) {
        py::gil_scoped_acquire gil;
        PyObject *v;
        if (!src || src.is_none() || !PyTuple_Check(src.ptr())) return false;
        PyObject *source = src.ptr();
        if (PyTuple_Size(source) != 2) return false;

        v = PyTuple_GetItem(source, 0);
        if (src.is_none() || !PyFloat_Check(source)) return false;
        value.x = PyFloat_AsDouble(v);

        v = PyTuple_GetItem(source, 1);
        if (src.is_none() || !PyFloat_Check(source)) return false;
        value.x = PyFloat_AsDouble(v);

        return true;
    }
    static py::handle cast(const cv::Point2f &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        
        PyObject *result = PyTuple_New(2);
        PyTuple_SetItem(result, 0, PyFloat_FromDouble(src.x));
        PyTuple_SetItem(result, 1, PyFloat_FromDouble(src.y));

        return py::handle(result);
    }
    PYBIND11_TYPE_CASTER(cv::Point2f, _("cv::Point2f"));
};



template <> class type_caster<pixelpipes::SharedVariable> {
    typedef pixelpipes::SharedVariable type;
public:
    bool load(py::handle src, bool) {
        if (!src || src.is_none()) { return false; }

        PyObject *source = src.ptr();
        if  (PyLong_Check(source)) {
            value = std::make_shared<pixelpipes::Integer>(PyLong_AsLong(source));
            return true;
        }

        if  (PyFloat_Check(source)) {
            value = std::make_shared<pixelpipes::Float>(PyFloat_AsDouble(source));
            return true;
        }
        return false;
    }
    static py::handle cast(const pixelpipes::SharedVariable &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        switch (src->type()) {
            case pixelpipes::VariableType::Integer: {
                return PyFloat_FromDouble(std::static_pointer_cast<pixelpipes::Integer>(src)->get());
            }
            case pixelpipes::VariableType::Float: {
                return PyFloat_FromDouble(std::static_pointer_cast<pixelpipes::Float>(src)->get());
            }
            case pixelpipes::VariableType::View: {
                cv::Mat m(std::static_pointer_cast<pixelpipes::View>(src)->get());
                NDArrayConverter cvt;
                py::handle result(cvt.toNDArray(m));
                return result;
            }
            case pixelpipes::VariableType::Image: {
                cv::Mat m(std::static_pointer_cast<pixelpipes::Image>(src)->get());
                NDArrayConverter cvt;
                py::handle result(cvt.toNDArray(m));
                return result;
            }
            default:
                return py::none(); 
        }
    }
    PYBIND11_TYPE_CASTER(pixelpipes::SharedVariable, _("SharedVariable"));
};


}
}

