#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <pixelpipes/operation.hpp>
#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

#include "common.hpp"
namespace pixelpipes
{

    PIXELPIPES_MODULE(image)

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

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

    ImageDepth decode_ocvtype(int cvtype)
    {

        int depth = CV_MAT_DEPTH(cvtype);

        ImageDepth dtype;

        switch (depth)
        {
        case CV_8U:
        {
            dtype = ImageDepth::Byte;
            break;
        }
        case CV_8S:
        {
            dtype = ImageDepth::Byte;
            break;
        }
        case CV_16U:
        {
            dtype = ImageDepth::Short;
            break;
        }
        case CV_16S:
        {
            dtype = ImageDepth::Short;
            break;
        }
        case CV_32F:
        {
            dtype = ImageDepth::Float;
            break;
        }
        case CV_64F:
        {
            dtype = ImageDepth::Double;
            break;
        }
        default:
        {
            throw TypeException("Unsupported data type");
        }
        }

        return dtype;
    }

    int encode_ocvtype(ImageDepth dtype)
    {

        int type = 0;

        switch (dtype)
        {
        case ImageDepth::Byte:
        {
            type = CV_8U;
            break;
        }
        case ImageDepth::Short:
        {
            type = CV_16U;
            break;
        }
        case ImageDepth::Float:
        {
            type = CV_32F;
            break;
        }
        case ImageDepth::Double:
        {
            type = CV_64F;
            break;
        }
        default:
        {
            throw TypeException("Unsupported data type");
        }
        }

        return type;
    }

    cv::Mat MatImage::make(const Image image)
    {

        int type = encode_ocvtype(image->depth());

        int ndims = image->channels() == 1 ? 2 : 3;

        int size[3] = {(int)image->height(), (int)image->width(), (int)image->channels()};

        if (ndims == 3 && size[2] <= CV_CN_MAX)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        cv::Mat mat(ndims, size, type);

        Image destination = std::make_shared<MatImage>(mat);

        copy(image, destination);

        return mat;
    }

    cv::Mat MatImage::wrap(const Image image)
    {

        int type = encode_ocvtype(image->depth());

        int ndims = image->channels() == 1 ? 2 : 3;

        int size[3] = {(int)image->height(), (int)image->width(), (int)image->channels()};
        size_t step[3] = {(size_t)image->rowstep(), (size_t)image->colstep(), 1};

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

        cv::Mat mat(ndims, size, type, (void *)image->data(), step);

        return mat;
    }

    MatImage::MatImage(cv::Mat data) : mat(data)
    {

        VERIFY(data.dims == 2, "Only two dimension matrices supported");

        decode_ocvtype(data.type());
    }

    MatImage::~MatImage()
    {
    }

    ImageDepth MatImage::depth() const
    {
        return decode_ocvtype(mat.type());
    }

    size_t MatImage::element() const
    {
        return ((size_t)depth() >> 3);
    }

    size_t MatImage::width() const
    {
        return mat.cols;
    }

    size_t MatImage::height() const
    {
        return mat.rows;
    }

    size_t MatImage::channels() const
    {
        return CV_MAT_CN(mat.type());
    }

    TypeIdentifier MatImage::backend() const
    {
        return GetTypeIdentifier<cv::Mat>();
    }

    size_t MatImage::rowstep() const
    {
        return mat.step[0];
    }

    size_t MatImage::colstep() const
    {
        return mat.step[1];
    }

    unsigned char *MatImage::data() const
    {
        return mat.data;
    }

    SharedToken ImageRead(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        std::string filename = extract<std::string>(inputs[0]);
//cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH
        cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        return wrap(image);
    }

    REGISTER_OPERATION_FUNCTION("read", ImageRead);

    SharedToken ImageReadColor(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        std::string filename = extract<std::string>(inputs[0]);

        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        return wrap(image);
    }

    REGISTER_OPERATION_FUNCTION("read_color", ImageReadColor);

    SharedToken ImageReadGrayscale(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        std::string filename = extract<std::string>(inputs[0]);

        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        if (image.empty())
            throw TypeException("Image not found or IO error: " + filename);

        return wrap(image);
    }

    REGISTER_OPERATION_FUNCTION("read_grayscale", ImageReadGrayscale);

    /**
     * @brief Converts depth of an image.
     *
     */
    SharedToken ConvertDepth(TokenList inputs, ImageDepth depth) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        double maxin = maximum_value(image);
        int dtype;
        double maxout = 1;

        switch (depth)
        {
        case ImageDepth::Byte:
            dtype = CV_8U;
            maxout = 255;
            break;
        case ImageDepth::Short:
            dtype = CV_16S;
            maxout = 255 * 255;
            break;
        case ImageDepth::Float:
            dtype = CV_32F;
            maxout = 1;
            break;
        case ImageDepth::Double:
            dtype = CV_64F;
            maxout = 1;
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required
            return inputs[0];
        }

        cv::Mat result;
        image.convertTo(result, dtype, maxout / maxin);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("convert", ConvertDepth, ImageDepth);

    /**
     * @brief Converts color image to grayscale image.
     *
     */
    SharedToken Grayscale(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        cv::Mat result;
        cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("grayscale", Grayscale);

    /**
     * @brief Returns an image with selected values.
     *
     */
    SharedToken Equals(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        int value = Integer::get_value(inputs[1]);

        VERIFY(image.channels() == 1, "Image has more than one channel");
        VERIFY(image.depth() == CV_8U || image.depth() == CV_16S, "Only integer bit types supported");

        cv::Mat result = (image == value);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("equals", Equals);

    /**
     * @brief Extracts a single channel from multichannel image.
     *
     */
    SharedToken Channel(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        int index = Integer::get_value(inputs[1]);

        VERIFY(index >= 0 && index < image.channels(), "Wrong channel index, out of bounds");

        cv::Mat result;
        cv::extractChannel(image, result, index);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("channel", Channel);

    /**
     * @brief Combines 3 single channel images into color image.
     *
     */
    SharedToken Merge(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 3, "Incorrect number of parameters");

        cv::Mat image_0 = extract<cv::Mat>(inputs[0]);
        cv::Mat image_1 = extract<cv::Mat>(inputs[1]);
        cv::Mat image_2 = extract<cv::Mat>(inputs[2]);

        VERIFY(image_0.depth() == image_1.depth() && image_1.depth() == image_2.depth(), "Image types do not match");
        VERIFY(image_0.rows == image_1.rows && image_1.rows == image_2.rows, "Image sizes do not match");
        VERIFY(image_0.cols == image_1.cols && image_1.cols == image_2.cols, "Image sizes do not match");

        std::vector<cv::Mat> channels;
        cv::Mat result;

        channels.push_back(image_0);
        channels.push_back(image_1);
        channels.push_back(image_2);
        CV_EX_WRAP(cv::merge(channels, result));

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("merge", Merge);

    /**
     * @brief Calculates image moments.
     *
     */
    SharedToken Moments(TokenList inputs, bool binary) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        VERIFY(image.channels() == 1, "Image has more than one channel");

        cv::Moments m = cv::moments(image, binary);

        std::vector<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

        return std::make_shared<FloatList>(make_span(data));
    }

    REGISTER_OPERATION_FUNCTION("moments", Moments, bool);

    /**
     * @brief Tabulates a function into a matrix of a given size
     *
     * TODO: reorganize into spearate methods
     *
     */
    SharedToken MapFunction(TokenList inputs, int function, bool normalize) noexcept(false)
    {

        VERIFY(inputs.size() > 1, "Incorrect number of parameters");

        int size_x = Integer::get_value(inputs[0]);
        int size_y = Integer::get_value(inputs[1]);

        cv::Mat result(size_y, size_x, CV_32F);

        switch (function)
        {
        case 0:
        {
            VERIFY(inputs.size() == 6, "Incorrect number of parameters");

            float mean_x = Float::get_value(inputs[2]);
            float mean_y = Float::get_value(inputs[3]);
            float sigma_x = Float::get_value(inputs[4]);
            float sigma_y = Float::get_value(inputs[5]);

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

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("map", MapFunction, int, bool);

    /**
     * @brief Thresholds an image.
     *
     */
    SharedToken ImageThreshold(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        float threshold = Float::get_value(inputs[1]);

        VERIFY(image.channels() == 1, "Image has more than one channel");

        float maxval = maximum_value(image);

        cv::Mat result;
        cv::threshold(image, result, threshold, maxval, cv::THRESH_BINARY);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("threshold", ImageThreshold);

    /**
     * @brief Inverts image pixel values.
     *
     */
    SharedToken Invert(TokenList inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        float maxval = maximum_value(image);

        cv::Mat result = maxval - image;

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("invert", Invert);

}
