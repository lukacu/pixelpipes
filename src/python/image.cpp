
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

namespace py = pybind11;

using namespace pixelpipes;

typedef std::function<void()> DescructorCallback;

#define HIDDEN __attribute__((visibility("hidden")))

class HIDDEN NumpyImage : public ImageData
{
public:
    NumpyImage(py::array data) : array(data)
    {

        if (array.ndim() < 2 || array.ndim() > 3)
            throw TypeException("Unsupported number of dimensions");

        const auto pyarray_dtype = data.dtype();
        if (pyarray_dtype.is(pybind11::dtype::of<uint8_t>()))
        {
            image_depth = ImageDepth::Byte;
        }
        else if (pyarray_dtype.is(pybind11::dtype::of<int8_t>()))
        {
            image_depth = ImageDepth::Byte;
        }
        else if (pyarray_dtype.is(pybind11::dtype::of<uint16_t>()))
        {
            image_depth = ImageDepth::Short;
        }
        else if (pyarray_dtype.is(pybind11::dtype::of<int16_t>()))
        {
            image_depth = ImageDepth::Short;
        }
        else if (pyarray_dtype.is(pybind11::dtype::of<float>()))
        {
            image_depth = ImageDepth::Float;
        }
        else if (pyarray_dtype.is(pybind11::dtype::of<double>()))
        {
            image_depth = ImageDepth::Double;
        }
        else
        {
            throw TypeException("Unsupported depth type");
        }

        // for (size_t i = 0; i < array.ndim(); i++) { std::cout << array.strides(i) << " - "; } std::cout << std::endl;
    }

    virtual ~NumpyImage()
    {
    }

    virtual ImageDepth depth() const
    {
        return image_depth;
    }

    virtual size_t width() const
    {
        return array.shape(1);
    }

    virtual size_t height() const
    {
        return array.shape(0);
    }

    virtual size_t channels() const
    {
        return array.ndim() == 2 ? 1 : array.shape(2);
    }

    virtual TypeIdentifier backend() const
    {
        return GetTypeIdentifier<py::array>();
    }

    virtual size_t rowstep() const
    {
        return array.strides(0);
    }

    virtual size_t colstep() const
    {
        return array.strides(1);
    }
/*
    virtual size_t element() const
    {
        return array.ndim() == 2 ? ((size_t)image_depth >> 3) : array.strides(2);
    }
*/
    virtual unsigned char *data() const
    {
        return (uint8_t *)array.data();
    }

private:
    py::array array;

    ImageDepth image_depth;
};

SharedToken wrap_image(py::object src)
{

    py::array a = py::array::ensure(src);

    if (!a)
        return empty<ImageData>();

    if (a.ndim() < 2 || a.ndim() > 3)
        return empty<ImageData>();

    try
    {
        Image src = std::make_shared<NumpyImage>(a);

        Image dst = std::make_shared<BufferImage>(src->width(), src->height(), src->channels(), src->depth());

        copy(src, dst);

        return dst;
    }
    catch (TypeException &e)
    {
        return empty<ImageData>();
    }
}

py::object extract_image(SharedToken src)
{

    if (!ImageData::is(src))
        return py::none();

    Image image = extract<Image>(src);

    auto v = new Image(image);
    // TODO: add capsule name
    auto capsule = py::capsule((void *)v, [](void *v)
                               { delete static_cast<Image *>(v); });

    std::vector<ssize_t> dimensions;
    std::vector<ssize_t> strides;

    dimensions.push_back(static_cast<ssize_t>(image->height()));
    dimensions.push_back(static_cast<ssize_t>(image->width()));

    strides.push_back(static_cast<ssize_t>(image->rowstep()));
    strides.push_back(static_cast<ssize_t>(image->colstep()));

    if (image->channels() > 1)
    {
        dimensions.push_back(static_cast<ssize_t>(image->channels()));
        strides.push_back(image->element());
    }

    // for (auto x : strides) { std::cout << x << " - "; } std::cout << std::endl;

    py::array result;

    switch (image->depth())
    {
    case ImageDepth::Byte:
    {
        result = py::array(std::move(dimensions), std::move(strides), (uint8_t *)image->data(), capsule);
        break;
    }
    case ImageDepth::Short:
    {
        result = py::array(std::move(dimensions), std::move(strides), (uint16_t *)image->data(), capsule);
        break;
    }
    case ImageDepth::Float:
    {
        result = py::array(std::move(dimensions), std::move(strides), (float_t *)image->data(), capsule);
        break;
    }
    case ImageDepth::Double:
    {
        result = py::array(std::move(dimensions), std::move(strides), (double_t *)image->data(), capsule);
        break;
    }
    default:
    {
        return py::none();
    }
    }

    result.inc_ref();
    return result;
}

SharedToken wrap_image_list(py::object src)
{
    if (py::list::check_(src))
    {
        try
        {
            py::list list(src);
            std::vector<Image> images;
            for (auto x : list)
            {
                SharedToken image = wrap_image(py::reinterpret_borrow<py::object>(x));
                if (!image)
                    return empty<List>();
                images.push_back(extract<Image>(image));
            }
            return std::make_shared<ImageList>(images);
        }
        catch (...)
        {
        }
    }
    return empty<List>();
}

py::object extract_image_list(SharedToken src)
{

    if (!List::is_list(src, ImageType))
    {
        return py::none();
    }

    SharedList list = List::cast(src);

    py::list out;

    for (size_t i = 0; i < list->size(); i++)
    {
        out.append(extract_image(list->get(i)));
    }

    return out;
}