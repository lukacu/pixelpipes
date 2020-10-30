
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

//#include <opencv2/core.hpp>

#include "conversion.hpp"
#include "engine.hpp"
#include "numbers.hpp"
#include "geometry.hpp"
#include "image.hpp"
#include "list.hpp"

namespace py = pybind11;

using namespace pixelpipes;

const static int initialized = init_conversion();

enum class ConvertOutput {None, Numpy, Torch };


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

#define ADD_OPERATION(O) {py::class_<O, Operation, std::shared_ptr<O> >(m, #O).def(py::init());}


// Solution based on: https://www.pierov.org/2020/03/01/python-custom-exceptions-c-extensions/
static PyObject *PipelineError_tp_str(PyObject *selfPtr)
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
}

static PyObject *PipelineError_getoperation(PyObject *selfPtr, void *closure) {
    try {
        py::handle self(selfPtr);
        py::tuple args = self.attr("args");
        py::object code = args[1];
        code.inc_ref();
        return code.ptr();
    } catch (py::error_already_set &e) {
        /* We could simply backpropagate the exception with e.restore, but
        exceptions like OSError return None when an attribute is not set. */
        py::none ret;
        ret.inc_ref();
        return ret.ptr();
    }
}

static PyGetSetDef PipelineError_getsetters[] = {
	{"operation", PipelineError_getoperation, NULL, NULL, NULL},
	{NULL}
};

static PyObject *PyPipelineError;

PYBIND11_MODULE(engine, m) {

    m.doc() = "C++ Core for PixelPipes";

    #ifdef __PP_PYTORCH
    m.attr("torch") = py::bool_(true);
    #else
    m.attr("torch") = py::bool_(false);
    #endif

    PyPipelineError = PyErr_NewException("engine.PipelineError", NULL, NULL);
    if (PyPipelineError) {
        PyTypeObject *as_type = reinterpret_cast<PyTypeObject *>(PyPipelineError);
        as_type->tp_str = PipelineError_tp_str;
        PyObject *descr = PyDescr_NewGetSet(as_type, PipelineError_getsetters);
        auto dict = py::reinterpret_borrow<py::dict>(as_type->tp_dict);
        dict[py::handle(PyDescr_NAME(descr))] = py::handle(descr);

        Py_XINCREF(PyPipelineError);
        m.add_object("PipelineError", py::handle(PyPipelineError));
    }

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (PipelineException &e) {
            py::tuple args(2);
            args[0] = e.what();
            args[1] = e.operation();
            PyErr_SetObject(PyPipelineError, args.ptr());
        }
    });

    py::enum_<ConvertOutput>(m, "Convert")
#ifdef __PP_PYTORCH
    .value("TORCH", ConvertOutput::Torch)
#endif
    .value("NONE", ConvertOutput::None)
    .value("NUMPY", ConvertOutput::Numpy);


    py::class_<Engine, std::shared_ptr<Engine> >(m, "Engine")
    .def(py::init<int>(), py::arg("workers") = 1)
    .def("start", &Engine::start, "Start engine")
    .def("stop", &Engine::stop, "Stop engine")
    .def("running", &Engine::running, "Is engine running")
    .def("add", &Engine::add, "Add pipeline to engine")
    .def("remove", &Engine::remove, "Remove pipeline from engine")
    .def("run", &Engine::run, "Run pipeline");

    py::class_<Pipeline, std::shared_ptr<Pipeline> >(m, "Pipeline")
    .def(py::init<>())
    .def("finalize", &Pipeline::finalize, "Finalize pipeline")
    .def("append", &Pipeline::append, "Add operation to pipeline")
    .def("run", [](Pipeline& p, unsigned long index, ConvertOutput convert) {
        py::gil_scoped_release gil; // release GIL lock
        std::vector<SharedVariable> result = p.run(index);

        std::vector<py::object> transformed;
        for (auto element : result) {
            switch (convert) {
                case ConvertOutput::None: {
                    transformed.push_back(py::cast(element));
                    break;
                }
                case ConvertOutput::Numpy: {
                    transformed.push_back(numpyFromVariable(element));
                    break;
                }
                case ConvertOutput::Torch: {
#ifdef __PP_PYTORCH
                    transformed.push_back(torchFromVariable(element));
                    break;
#else
                    throw PipelineException("Engine not compiled with PyTorch support", p.shared_from_this(), 0);
#endif
                }
            }
            
        }
        return transformed; 
        }, "Run pipeline", py::arg("index"), py::arg("convert") = ConvertOutput::None);

    py::class_<PipelineCallback, PyPipelineCallback, std::shared_ptr<PipelineCallback> >(m, "PipelineCallback")
    .def(py::init());

    py::class_<Operation, std::shared_ptr<Operation> >(m, "Operation");

    py::class_<Output, Operation, std::shared_ptr<Output> >(m, "Output").def(py::init());

    py::class_<Constant, Operation, std::shared_ptr<Constant> >(m, "Constant")
    .def(py::init<SharedVariable>());

    ADD_OPERATION(Copy);

    // Numeric Enums

    py::enum_<ComparisonOperation>(m, "Compare")
    .value("EQUAL", ComparisonOperation::EQUAL)
    .value("LOWER", ComparisonOperation::LOWER)
    .value("LOWER_EQUAL", ComparisonOperation::LOWER_EQUAL)
    .value("GREATER", ComparisonOperation::GREATER)
    .value("GREATER_EQUAL", ComparisonOperation::GREATER_EQUAL);

    py::enum_<LogicalOperation>(m, "Logical")
    .value("AND", LogicalOperation::AND)
    .value("OR", LogicalOperation::OR)
    .value("NOT", LogicalOperation::NOT);

    // Numeric operations
    ADD_OPERATION(NormalDistribution);
    ADD_OPERATION(UniformDistribution);
    ADD_OPERATION(Round);
    ADD_OPERATION(Add);
    ADD_OPERATION(Subtract);
    ADD_OPERATION(Multiply);
    ADD_OPERATION(Divide);
    ADD_OPERATION(Power);

    // View operations
    ADD_OPERATION(TranslateView);
    ADD_OPERATION(RotateView);
    ADD_OPERATION(ScaleView);
    ADD_OPERATION(IdentityView);
    ADD_OPERATION(Chain);
    ADD_OPERATION(ViewPoints);
    ADD_OPERATION(CenterView);
    ADD_OPERATION(FocusView);
    ADD_OPERATION(BoundingBox);

    // Image operations

    py::class_<ViewImage, Operation, std::shared_ptr<ViewImage> >(m, "ViewImage")
    .def(py::init<bool, BorderStrategy>());

    py::class_<ConvertDepth, Operation, std::shared_ptr<ConvertDepth> >(m, "ConvertDepth")
    .def(py::init<ImageDepth>());

    ADD_OPERATION(Equals);
    ADD_OPERATION(Invert);
    ADD_OPERATION(MaskBoundingBox);

    py::enum_<ImageDepth>(m, "ImageDepth")
    .value("BYTE", ImageDepth::Byte)
    .value("SHORT", ImageDepth::Short)
    .value("FLOAT", ImageDepth::Float)
    .value("DOUBLE", ImageDepth::Double);


    py::enum_<BorderStrategy>(m, "BorderStrategy")
    .value("CONSTANT_HIGH", BorderStrategy::ConstantHigh)
    .value("CONSTANT_LOW", BorderStrategy::ConstantLow)
    .value("REPLICATE", BorderStrategy::Replicate)
    .value("REFLECT", BorderStrategy::Reflect)
    .value("WRAP", BorderStrategy::Wrap);

    // Variables 
    py::class_<List, std::shared_ptr<List> >(m, "List");
    py::class_<ImageFileList, List, std::shared_ptr<ImageFileList> >(m, "ImageFileList")
    .def(py::init<std::vector<std::string>, std::string, bool>(), py::arg("list"), py::arg("prefix") = std::string(), py::arg("grayscale") = false);
    py::class_<ImageList, List, std::shared_ptr<ImageList> >(m, "ImageList")
    .def(py::init<std::vector<cv::Mat> >());
    py::class_<PointsList, List, std::shared_ptr<PointsList> >(m, "PointsList")
    .def(py::init<std::vector<std::vector<cv::Point2f> > >())
    .def(py::init<std::vector<cv::Mat> >());

    py::class_<IntegerList, List, std::shared_ptr<IntegerList> >(m, "IntegerList")
    .def(py::init<std::vector<int> >());
    
    py::class_<FloatList, List, std::shared_ptr<FloatList> >(m, "FloatList")
    .def(py::init<std::vector<float> >());

    py::class_<ListSource, Operation, std::shared_ptr<ListSource> >(m, "ListSource")
    .def(py::init<std::shared_ptr<List> >());

    ADD_OPERATION(SublistSelect);
    ADD_OPERATION(FilterSelect);
    ADD_OPERATION(ListElement);
    ADD_OPERATION(ListLength);

    py::class_<ListCompare, Operation, std::shared_ptr<ListCompare> >(m, "ListCompare")
    .def(py::init<ComparisonOperation>());

    py::class_<ListLogical, Operation, std::shared_ptr<ListLogical> >(m, "ListLogical")
    .def(py::init<LogicalOperation>());


}
