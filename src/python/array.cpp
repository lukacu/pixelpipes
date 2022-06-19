
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <pixelpipes/pipeline.hpp>
#include <pixelpipes/module.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/buffer.hpp>
#include <pixelpipes/python.hpp>

#include "../debug.h"

namespace py = pybind11;

using namespace pixelpipes;

typedef Function<void()> DescructorCallback;

#define HIDDEN __attribute__((visibility("hidden")))

class HIDDEN NumpyArray : public Tensor
{
public:
    NumpyArray(const py::array& data) : _array(data)
    {
        _element = AnyType;

        if (py::isinstance<py::array_t<uint8_t>>(data))
        {
            _element = CharIdentifier;
            _bytes = sizeof(uchar);
        }
        else if (py::isinstance<py::array_t<int8_t>>(data))
        {
            _element = CharIdentifier;
            _bytes = sizeof(uchar);
        }
        else if (py::isinstance<py::array_t<uint16_t>>(data))
        {
            _element = ShortIdentifier;
            _bytes = sizeof(ushort);
        }
        else if (py::isinstance<py::array_t<int16_t>>(data))
        {
            _element = ShortIdentifier;
            _bytes = sizeof(ushort);
        }
        else if (py::isinstance<py::array_t<int32_t>>(data))
        {
            _element = IntegerIdentifier;
            _bytes = sizeof(int);
        }
        else if (py::isinstance<py::array_t<float>>(data))
        {
            _element = FloatIdentifier;
            _bytes = sizeof(float);
        }

        if (_element == AnyType)
        {
            const auto pyarray_dtype = data.dtype();
            throw TypeException(Formatter() << "Unsupported depth type " << pyarray_dtype);
        }

        std::vector<size_t> shape;
        std::vector<size_t> strides;

        for (ssize_t i = 0; i < _array.ndim(); i++) {
            shape.push_back(_array.shape(i));
            strides.push_back(_array.strides(i));
        }

        _shape = SizeSequence(shape);
        _strides = SizeSequence(strides);

    }

    virtual ~NumpyArray() = default;

    virtual void describe(std::ostream &os) const
    {
        os << "[NumPy array wrapper]";
    }

    virtual Shape shape() const
    {
        return Shape(_element, _shape);
    }

    virtual size_t length() const
    {
        return _shape[0];
    }

    virtual size_t size() const
    {
        return _array.nbytes();
    }

    virtual TokenReference get(size_t i) const
    {
        // TODO
        UNUSED(i);
        return empty<IntegerScalar>();
    }

    virtual TokenReference get(const Sizes &i) const
    {
        // TODO
        UNUSED(i);
        return empty<IntegerScalar>();
    }

    virtual ReadonlySliceIterator read_slices() const
    {
        return ReadonlySliceIterator((uint8_t *)_array.data(), _shape, _strides, _bytes);
    }

    virtual WriteableSliceIterator write_slices()
    {
        return WriteableSliceIterator((uint8_t *)_array.data(), _shape, _strides, _bytes);
    }

    virtual const uchar *const_data() const
    {
        return (const uchar *) _array.data();
    }

    virtual uchar *data() 
    {
        return (uchar *) _array.mutable_data();
    }

    virtual SizeSequence strides() const 
    {
        return _strides;
    }

    virtual size_t cell_size() const
    {
        return _bytes;
    }

private:
    py::array _array;
    TypeIdentifier _element;
    SizeSequence _shape;
    SizeSequence _strides;
    size_t _bytes;
};

TokenReference wrap_tensor(const py::object& src)
{

    py::array a = py::array::ensure(src);

    VERIFY((bool)a, "Not an array");

    try
    {
        TensorReference src = create<NumpyArray>(a);
        TensorReference dst;
        
        Shape srcshape = src->shape();

        if (srcshape.element() == CharIdentifier) {
            dst = create_tensor<uchar>(src->shape());
        } else if (srcshape.element() == ShortIdentifier) {
            dst = create_tensor<ushort>(src->shape());
        } else if (srcshape.element() == FloatIdentifier) {
            dst = create_tensor<float>(src->shape());
        } else if (srcshape.element() == IntegerIdentifier) {
            dst = create_tensor<int>(src->shape());
        } else {
            throw TypeException("Unable to wrap tensor to Numpy array");
        }
        
        copy_buffer(src, dst);

        return dst;
    }
    catch (TypeException &e)
    {
        throw py::value_error(Formatter() << "Conversion error: " << e.what());
    }
}

struct TensorGuard {
    TensorReference guard;
};

py::object extract_tensor(const TokenReference& src)
{

    if (!src || !src->is<Tensor>())
        return py::none();

    TensorReference tensor = extract<TensorReference>(src);

    auto v = new TensorGuard{tensor.reborrow()};
    // TODO: add capsule name
    auto capsule = py::capsule((void *)v, [](void *v)
                               { delete static_cast<TensorGuard *>(v); });

    std::vector<ssize_t> pydimensions;
    std::vector<ssize_t> pystrides;

    Shape shape = tensor->shape();
    SizeSequence strides = tensor->strides();

    pydimensions.resize(shape.dimensions());
    pystrides.resize(shape.dimensions());

    for (size_t i = 0; i < shape.dimensions(); i++) {
        pydimensions[i] = static_cast<ssize_t>((size_t)shape[i]);
        pystrides[i] = static_cast<ssize_t>(strides[i]);
    }
/*
    if (image->channels() > 1)
    {
        dimensions.push_back(static_cast<ssize_t>(image->channels()));
        strides.push_back(image->element());
    }*/

    if (shape.element() == CharIdentifier)
    {
        return py::array(std::move(pydimensions), std::move(pystrides), (uint8_t *)tensor->data(), capsule);
    }
    if (shape.element() == IntegerIdentifier)
    {
        return py::array(std::move(pydimensions), std::move(pystrides), (int32_t *)tensor->data(), capsule);
    }
    if (shape.element() == ShortIdentifier)
    {
        return py::array(std::move(pydimensions), std::move(pystrides), (uint16_t *)tensor->data(), capsule);
    }
    if (shape.element() == FloatIdentifier)
    {
        return py::array(std::move(pydimensions), std::move(pystrides), (float *)tensor->data(), capsule);
    }

    return py::none();
}

TokenReference wrap_tensor_list(const py::object& src)
{
    if (py::list::check_(src))
    {
        try
        {
            py::list list(src);
            Sequence<TensorReference> tensors;
            for (size_t i = 0; i < list.size(); i++)
            {
                TokenReference tensor = wrap_tensor(py::reinterpret_borrow<py::object>(list[i]));
                if (!tensor)
                {
                    DEBUGMSG("Error during image conversion \n");
                    return empty();
                }
                tensors[i] = (extract<TensorReference>(tensor));
            }
            return wrap(tensors);
        }
        catch (...)
        {
            DEBUGMSG("Error during image list conversion \n");
        }
    }
    return empty<List>();
}
