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
py::object pythonFromVariable(pixelpipes::SharedVariable variable);

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

bool convert_numpy_mat(py::handle src, cv::Mat *dst);
py::object convert_mat_numpy(cv::Mat src);
bool convert_tuple_point(py::handle src, cv::Point2f *dst);
py::object convert_point_tuple(cv::Point2f src);

namespace pybind11 {
namespace detail {

template <> class type_caster<cv::Mat> {
    typedef cv::Mat type;
public:
    bool load(py::handle src, bool) {
        return convert_numpy_mat(src, &value);
    }
    static py::handle cast(const cv::Mat &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        return convert_mat_numpy(src);
    }
    PYBIND11_TYPE_CASTER(cv::Mat, _("cv::Mat"));
};

template <> class type_caster<cv::Point2f> {
    typedef cv::Point2f type;
public:
    bool load(py::handle src, bool) {
        py::gil_scoped_acquire gil;
        return convert_tuple_point(src, &value);
    }
    static py::handle cast(const cv::Point2f &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        return convert_point_tuple(src);
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
        cv::Point2f p;
        if (convert_tuple_point(source, &p)) {
            value = std::make_shared<pixelpipes::Point>(p);
            return true;
        }

        cv::Mat mat;
        if (convert_numpy_mat(source, &mat)) {
            value = std::make_shared<pixelpipes::Image>(mat);
            return true;
        } 
        
        return false;
    }
    static py::handle cast(const pixelpipes::SharedVariable &src, return_value_policy policy, py::handle parent) {
        py::gil_scoped_acquire gil;
        return pythonFromVariable(src);
    }
    PYBIND11_TYPE_CASTER(pixelpipes::SharedVariable, _("SharedVariable"));
};


}
}

