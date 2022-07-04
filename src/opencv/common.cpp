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
    /*
    #ifndef CV_MAX_DIM
        const int CV_MAX_DIM = 32;
    #endif
    */
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
        for (size_t i = 1; i < s.dimensions(); i++)
        {
            os << " x " << (size_t)s[i];
        }
        os << " - OpenCV]";
    }


    cv::Mat MatImage::copy(const TensorReference &tensor)
    {

        Shape shape = tensor->shape();

        int type = encode_ocvtype(shape.element());

        VERIFY(shape.dimensions() == 2 || shape.dimensions() == 3, "Only rank 2 or rank 3 tensors accepted");

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

    cv::Mat MatImage::wrap(const TensorReference &tensor)
    {

        Shape shape = tensor->shape();

        int type = encode_ocvtype(shape.element());

        VERIFY(shape.dimensions() == 2 || shape.dimensions() == 3, "Only rank 2 or rank 3 tensors accepted");

        auto strides = tensor->strides();

        // This is very likely not correct, should look at stride as well
        int ndims = shape[2] == 1 ? 2 : 3;

        int size[3] = {(int)shape[0], (int)shape[1], (int)shape[2]};
        size_t step[3] = {strides[0], strides[1], 1};

        // bool transposed = false;

        /*if (image->channels() *image->element() != image->colstep())
        {
            size[2] = 1;
            step[2] = image->element();
            ndims++;
        }

        if (ndims >= 2 && step[0] < step[1])
        {
            std::swap(size[0], size[1]);
            std::swap(step[0], step[1]);
            transposed = true;
        }*/

        if (ndims == 3 && size[2] <= CV_CN_MAX)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        cv::Mat mat(ndims, size, type, (void *)tensor->data(), step);

        return mat;
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

            _strides = {_mat.step[0], _mat.step[1], type_size(_element)};
            _shape = {(size_t)_mat.rows, (size_t)_mat.cols, (size_t)data.channels()};
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

    const uchar *MatImage::const_data() const
    {
        return _mat.data;
    }

    uchar *MatImage::data()
    {
        return _mat.data;
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

    cv::Mat image_read(std::string filename) noexcept(false)
    {
        // cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH
        cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read", image_read);

    cv::Mat image_read_color(std::string filename) noexcept(false)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read_color", image_read_color);

    cv::Mat image_read_grayscale(std::string filename) noexcept(false)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read_grayscale", image_read_grayscale);

    /**
     * @brief Converts depth of an image.
     *
     */
    cv::Mat convert_depth(const cv::Mat &image, ImageDepth depth) noexcept(false)
    {
        double maxin = maximum_value(image);
        int dtype;
        double maxout = 1;

        switch (depth)
        {
        case ImageDepth::Byte:
            dtype = CV_8S;
            maxout = std::numeric_limits<char>::max();
            break;
        case ImageDepth::UByte:
            dtype = CV_8U;
            maxout = std::numeric_limits<uchar>::max();
            break;
        case ImageDepth::Short:
            dtype = CV_16S;
            maxout = std::numeric_limits<short>::max();
            break;
        case ImageDepth::UShort:
            dtype = CV_16U;
            maxout = std::numeric_limits<ushort>::max();
            break;
        case ImageDepth::Integer:
            dtype = CV_32S;
            maxout = std::numeric_limits<int>::max();
            break;
        case ImageDepth::Float:
            dtype = CV_32F;
            maxout = 1;
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required
            return image;
        }

        cv::Mat result;
        image.convertTo(result, dtype, maxout / maxin);

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("convert_depth", convert_depth);

    /**
     * @brief Converts color image to grayscale image.
     *
     */
    cv::Mat grayscale(const cv::Mat &image) noexcept(false)
    {
        cv::Mat result;
        cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("grayscale", grayscale);

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

    /**
     * @brief Extracts a single channel from multichannel image.
     *
     */
    cv::Mat extract_channel(const cv::Mat &image, int index) noexcept(false)
    {
        VERIFY(index >= 0 && index < image.channels(), "Wrong channel index, out of bounds");

        cv::Mat result;
        cv::extractChannel(image, result, index);

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("extract_channel", extract_channel);

    /**
     * @brief Combines 3 single channel images into color image.
     * TODO: make number of channels variable
     */
    cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1, const cv::Mat &c2) noexcept(false)
    {

        VERIFY(c0.depth() == c1.depth() && c1.depth() == c2.depth(), "Image types do not match");
        VERIFY(c0.rows == c1.rows && c1.rows == c2.rows, "Image sizes do not match");
        VERIFY(c0.cols == c1.cols && c1.cols == c2.cols, "Image sizes do not match");

        std::vector<cv::Mat> channels;
        cv::Mat result;

        channels.push_back(c0);
        channels.push_back(c1);
        channels.push_back(c2);
        CV_EX_WRAP(cv::merge(channels, result));

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("merge_channels", merge);

    /**
     * @brief Calculates image moments.
     *
     */
    Sequence<float> moments(const cv::Mat &image, bool binary) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");

        cv::Moments m = cv::moments(image, binary);

        Sequence<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

        return data;
    }

    PIXELPIPES_OPERATION_AUTO("moments", moments);

    /**
     * @brief Tabulates a function into a matrix of a given size
     *
     * TODO: reorganize into spearate methods
     *
     */
    /*cv::Mat map_gaussian(int size_x, int size_y, int function, bool normalize) noexcept(false)
    {
        cv::Mat result(size_y, size_x, CV_32F);

        switch (function)
        {
        case 0:
        {
            VERIFY(inputs.size() == 6, "Incorrect number of parameters");

            float mean_x = FloatScalar::get_value(inputs[2]);
            float mean_y = FloatScalar::get_value(inputs[3]);
            float sigma_x = FloatScalar::get_value(inputs[4]);
            float sigma_y = FloatScalar::get_value(inputs[5]);

            // intialising standard deviation to 1.0
            float r;
            float sum = 0.0;

            // generating 5x5 kernel
            for (int x = 0; x < size_x; x++)
            {
                for (int y = 0; y < size_y; y++)
                {
                    float px = x - mean_x;
                    float py = y - mean_y;
                    r = (px * px) / (2 * sigma_x * sigma_x) + (py * py) / (2 * sigma_y * sigma_y);
                    float v = exp(-r);
                    sum += v;
                    result.at<float>(y, x) = v;
                }
            }

            if (normalize)
                result /= sum;
        }
        }

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("map_function", map_function);*/

    /**
     * @brief Thresholds an image.
     *
     */
    cv::Mat image_threshold(const cv::Mat &image, float threshold) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");
        float maxval = maximum_value(image);
        cv::Mat result;
        cv::threshold(image, result, threshold, maxval, cv::THRESH_BINARY);
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("threshold", image_threshold);

    /**
     * @brief Inverts image pixel values.
     *
     */
    cv::Mat Invert(const cv::Mat &image) noexcept(false)
    {
        float maxval = maximum_value(image);
        cv::Mat result = maxval - image;
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("invert", Invert);

}
