
#include "image.hpp"
#include "geometry.hpp"
#include "numbers.hpp"
#include "list.hpp"

#include <opencv2/imgproc.hpp>

namespace pixelpipes {

inline int maximum_value(cv::Mat image) {

    switch (image.depth()) {
    case CV_8U:
        return 255;
    case CV_16S:
        return 255 * 255;
    case CV_32F:
    case CV_64F:
        return 1;
    default:
        throw VariableException("Unsupported image depth");
    }
}

ViewImage::ViewImage(bool interpolate, BorderStrategy border) : interpolate(interpolate) {

        border_value = 0;

        switch (border) {
        case BorderStrategy::ConstantHigh:
            border_const = cv::BORDER_CONSTANT;
            border_value = 1;
            break;
        case BorderStrategy::ConstantLow:
            border_const = cv::BORDER_CONSTANT;
            border_value = 0;
            break;
        case BorderStrategy::Replicate:
            border_const = cv::BORDER_REPLICATE;
            break;
        case BorderStrategy::Reflect:
            border_const = cv::BORDER_REFLECT;
            break;
        case BorderStrategy::Wrap:
            border_const = cv::BORDER_WRAP;
            break;
        default:
            throw OperationException("Illegal border strategy value", shared_from_this());
    }

}

SharedVariable ViewImage::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 4) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);
    cv::Matx33f transform = View::get_value(inputs[1]);

    int width = Integer::get_value(inputs[2]);
    int height = Integer::get_value(inputs[3]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate ? cv::INTER_LINEAR : cv::INTER_NEAREST, border_const, bvalue);
    } catch (cv::Exception cve) {
        throw OperationException(cve.what(), shared_from_this());
    }

    return std::make_shared<Image>(output);

}

ConvertDepth::ConvertDepth(ImageDepth depth) : depth(depth) {};


SharedVariable ConvertDepth::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);

    double maxin = maximum_value(image);
    int dtype;
    double maxout;

    switch (depth) {
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

    if (image.depth() == dtype) {
        // No conversion required
        return inputs[0];
    }

    cv::Mat result;
    image.convertTo(result, dtype, maxout / maxin);

    return std::make_shared<Image>(result);

}

SharedVariable Grayscale::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);

    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);

    return std::make_shared<Image>(result);

}

SharedVariable Threshold::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);
    float threshold = Float::get_value(inputs[1]);

    if (image.channels() != 1)
        throw OperationException("Image has more than one channel", shared_from_this());

    float maxval = maximum_value(image);

    cv::Mat result;
    cv::threshold(image, result, threshold, maxval, cv::THRESH_BINARY);

    return std::make_shared<Image>(result);

}

SharedVariable Invert::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);

    float maxval = maximum_value(image);

    cv::Mat result = maxval - image;

    return std::make_shared<Image>(result);
   
}

SharedVariable Equals::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);
    int value = Integer::get_value(inputs[1]);

    if (image.channels() != 1)
        throw OperationException("Image has more than one channel", shared_from_this());

    if (image.depth() != CV_8U && image.depth() != CV_16S)
            throw OperationException("Only integer bit types supported", shared_from_this());

    cv::Mat result = (image == value);

    return std::make_shared<Image>(result);

}

SharedVariable Channel::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);
    int index = Integer::get_value(inputs[1]);

    if (index < 0 || image.channels() >= index)
        throw OperationException("Wrong channel index, out of bounds", shared_from_this());


    cv::Mat result;
    cv::extractChannel(image, result, index);

    return std::make_shared<Image>(result);

}

SharedVariable Moments::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    cv::Mat image = Image::get_value(inputs[0]);

    if (image.channels() != 1)
        throw OperationException("Image has more than one channel", shared_from_this());


    cv::Moments m = cv::moments(image);

    std::vector<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

    return std::make_shared<FloatList>(data);
   
}


SharedVariable Polygon::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    
}


SharedVariable Noise::run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    
}

template<typename T>
std::vector<float> bounds(cv::Mat image) {

    int top = std::numeric_limits<int>::max();
    int bottom = std::numeric_limits<int>::lowest();
    int left = std::numeric_limits<int>::max();
    int right = std::numeric_limits<int>::lowest();

    for (int y = 0; y < image.rows; y++) {
        T *p = image.ptr<T>(y);
        for (int x = 0; x < image.cols; x++) {
            if (p[x]) {
                top = std::min(top, y);
                left = std::min(left, x);
                bottom = std::max(bottom, y);
                right = std::max(right , x);
            }
        }

    }

    if (top > bottom)
        return {(float)0, (float)0, (float)image.cols, (float)image.rows};
    else
        return {(float)left, (float)top, (float)right, (float)bottom};

}

SharedVariable MaskBoundingBox::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    cv::Mat image = Image::get_value(inputs[0]);

    if (image.channels() != 1)
        throw OperationException("Image has more than one channel", shared_from_this());



    switch (image.depth()) {
        case CV_8U:
            return std::make_shared<FloatList>(bounds<uint8_t>(image));
        case CV_16S:
            return std::make_shared<FloatList>(bounds<uint16_t>(image));
        case CV_32F:
            return std::make_shared<FloatList>(bounds<float>(image));
        case CV_64F:
            return std::make_shared<FloatList>(bounds<double>(image));
        default:
            throw OperationException("Unsupported image depth", shared_from_this());
    }

}

}