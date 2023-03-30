#include <cmath>
#include <limits>

#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/module.hpp>
#include <pixelpipes/tensor.hpp>

#include "common.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

PIXELPIPES_MODULE(opencv)

namespace pixelpipes
{

    PIXELPIPES_REGISTER_ENUM("interpolation", Interpolation);
    PIXELPIPES_REGISTER_ENUM("border", BorderStrategy);
    PIXELPIPES_REGISTER_ENUM("depth", ImageDepth);

    int ocv_border_type(BorderStrategy b, int *value)
    {
        switch (b)
        {
        case BorderStrategy::ConstantHigh:
            *value = 1;
            return cv::BORDER_CONSTANT;
        case BorderStrategy::ConstantLow:
            *value = 0;
            return cv::BORDER_CONSTANT;
        case BorderStrategy::Replicate:
            return cv::BORDER_REPLICATE;
        case BorderStrategy::Reflect:
            return cv::BORDER_REFLECT;
        case BorderStrategy::Wrap:
            return cv::BORDER_WRAP;
        default:
            throw TypeException("Illegal border strategy value");
        }
    }

    TypeIdentifier decode_ocvtype(int cvtype)
    {

        int depth = CV_MAT_DEPTH(cvtype);

        switch (depth)
        {
        case CV_8U:
        case CV_8S:
        {
            return CharIdentifier;
        }
        case CV_16U:
        {
            return UShortIdentifier;
        }
        case CV_16S:
        {
            return ShortIdentifier;
        }
        case CV_32S:
        {
            return IntegerIdentifier;
        }
        case CV_32F:
        {
            return FloatIdentifier;
        }
        default:
        {
            throw TypeException("Unable to convert data type to OpenCV");
        }
        }
    }

    int encode_ocvtype(TypeIdentifier dtype)
    {
        switch (dtype)
        {
        case CharIdentifier:
        {
            return CV_8U;
        }
        case IntegerIdentifier:
        {
            return CV_32S;
        }
        case UShortIdentifier:
        {
            return CV_16U;
        }
        case ShortIdentifier:
        {
            return CV_16S;
        }
        case FloatIdentifier:
        {
            return CV_32F;
        }
        default:
        {
            throw TypeException("Unable to convert data type from OpenCV");
        }
        }
    }

    void MatImage::describe(std::ostream &os) const
    {

        std::string element("Unknown");

        int depth = CV_MAT_DEPTH(_mat.type());

        switch (depth)
        {
        case CV_8U:
        {
            element = "unsigned char";
            break;
        }
        case CV_8S:
        {
            element = "char";
            break;
        }
        case CV_16U:
        {
            element = "ushort";
            break;
        }
        case CV_16S:
        {
            element = "short";
            break;
        }
        case CV_32S:
        {
            element = "int";
            break;
        }
        case CV_32F:
        {
            element = "float";
            break;
        }
        }

        Shape s = shape();

        os << "[Tensor of " << element << " " << (size_t)s[0];
        for (size_t i = 1; i < s.rank(); i++)
        {
            os << " x " << (size_t)s[i];
        }
        os << " - OpenCV]";
    }

    MatImage::~MatImage()
    {
    }

    cv::Mat copy(const TensorReference &tensor)
    {

        Shape shape = tensor->shape();

        int type = encode_ocvtype(shape.element());

        VERIFY(shape.rank() == 2 || shape.rank() == 3, "Only rank 2 or rank 3 tensors accepted");

        int ndims = shape[2] == 1 ? 2 : 3;

        int size[3] = {(int)shape[0], (int)shape[1], (int)shape[2]};

        if (ndims == 3 && size[2] <= CV_CN_MAX)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        cv::Mat mat(ndims, size, type);

        TensorReference destination = create<MatImage>(mat);

        copy_buffer(tensor, destination);

        return mat;
    }

    cv::Mat wrap_tensor(const TensorReference &tensor)
    {

        Shape shape = tensor->shape();

        int type = encode_ocvtype(shape.element());

        VERIFY(shape.rank() == 2 || shape.rank() == 3, "Only rank 2 or rank 3 tensors accepted");

        auto strides = tensor->strides();

        if (shape.rank() == 3)
        {
            int size[2] = {(int)shape[1], (int)shape[2]};
            size_t step[2] = {strides[1], strides[2]};
 
            VERIFY(size[0] <= CV_CN_MAX, "Channel number exceeds OpenCV supported number");

            type |= CV_MAKETYPE(0, ((int)shape[0]));
            if (type_size(shape.element()) != (size_t) strides[0])
            {
                // Channel stride not supported by OpenCV, must copy data
                cv::Mat mat(2, size, type);
                TensorReference destination = create<MatImage>(mat);
                copy_buffer(tensor, destination);
                return mat;
            } else {
                cv::Mat mat(2, size, type, (void *)tensor->data().data(), step);
                return mat;
            }
        } else {
            int size[2] = {(int)shape[0], (int)shape[1]};
            size_t step[2] = {strides[0], strides[1]};
 
            cv::Mat mat(2, size, type, (void *)tensor->data().data(), step);
            return mat;
        }


    }

    TensorReference create_tensor(const cv::Mat &mat)
    {
        auto tensor_type = decode_ocvtype(mat.type());

        if (mat.channels() == 1)
        {
            SizeSequence shape = {(size_t)mat.rows, (size_t)mat.cols};
            return create_tensor(tensor_type, shape);
        }
        else
        {
            SizeSequence shape = {(size_t)mat.channels(), (size_t)mat.rows, (size_t)mat.cols};
            return create_tensor(tensor_type, shape);
        }
    }

    MatImage::MatImage(cv::Mat data) : _mat(data)
    {
        VERIFY(data.dims == 2, "Only two dimensional matrices supported");

        _element = decode_ocvtype(data.type());

        if (data.channels() == 1)
        {
            _strides = {_mat.step[0], _mat.step[1]};
            _shape = {(size_t)_mat.rows, (size_t)_mat.cols};
        }
        else
        {
            _strides = {type_size(_element), _mat.step[0], _mat.step[1]};
            _shape = {(size_t)data.channels(), (size_t)_mat.rows, (size_t)_mat.cols};
        }
    }

    Shape MatImage::shape() const
    {
        return Shape(_element, _shape);
    }

    size_t MatImage::length() const
    {
        return _mat.rows;
    }

    size_t MatImage::size() const
    {
        return _mat.total() * _mat.elemSize();
    }

    TokenReference MatImage::get(size_t i) const
    {
        // TODO
        UNUSED(i);
        return empty<IntegerScalar>();
    }

    TokenReference MatImage::get(const Sizes &i) const
    {
        VERIFY(i.size() == _shape.size(), "Dimension mismatch");

        switch (_element)
        {
        case CharIdentifier:
            return create<CharScalar>(_mat.at<uchar>((const int *)i.data()));
        case ShortIdentifier:
            return create<ShortScalar>(_mat.at<short>((const int *)i.data()));
        case UShortIdentifier:
            return create<UShortScalar>(_mat.at<ushort>((const int *)i.data()));
        case IntegerIdentifier:
            return create<IntegerScalar>(_mat.at<int>((const int *)i.data()));
        case FloatIdentifier:
            return create<FloatScalar>(_mat.at<float>((const int *)i.data()));
        }

        throw TypeException("Cell access error");
    }

    ReadonlySliceIterator MatImage::read_slices() const
    {
        return ReadonlySliceIterator(const_data(), _shape, _strides, cell_size());
    }

    WriteableSliceIterator MatImage::write_slices()
    {
        return WriteableSliceIterator(data(), _shape, _strides, cell_size());
    }

    ByteView MatImage::const_data() const
    {
        return ByteView(_mat.data, size());
    }

    ByteSpan MatImage::data()
    {
        return ByteSpan(_mat.data, size());
    }

    size_t MatImage::cell_size() const
    {
        return _mat.elemSize1();
    }

    TypeIdentifier MatImage::cell_type() const
    {
        return _element;
    }

    SizeSequence MatImage::strides() const
    {
        return _strides;
    }

    cv::Mat MatImage::get() const
    {
        return _mat;
    }

    /**
     * @brief Returns an image with selected values.
     *
     */
    cv::Mat equals(const cv::Mat &image, int value) noexcept(false)
    {

        VERIFY(image.channels() == 1, "Image has more than one channel");
        VERIFY(image.depth() == CV_8U || image.depth() == CV_8S || image.depth() == CV_16S || image.depth() == CV_16U || image.depth() == CV_32S, "Only integer bit types supported");

        cv::Mat result = (image == value);
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("equals", equals);

}
