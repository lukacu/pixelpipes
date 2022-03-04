
#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

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

inline int interpolate_convert(Interpolation interpolation) {

    switch (interpolation) {
    case Interpolation::Linear:
        return cv::INTER_LINEAR;
    case Interpolation::Area:
        return cv::INTER_AREA;
    case Interpolation::Cubic:
        return cv::INTER_CUBIC;
    case Interpolation::Lanczos:
        return cv::INTER_LANCZOS4;
    default:
        return cv::INTER_NEAREST;
    }
}


/**
 * @brief Apply view linear transformation to an image.
 * 
 */
SharedVariable ViewImage(std::vector<SharedVariable> inputs, Interpolation interpolation, BorderStrategy border) {

    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int border_value = 0;
    int border_const = 0;

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
        throw VariableException("Illegal border strategy value");
    }

    cv::Mat image = Image::get_value(inputs[0]);
    cv::Matx33f transform = View2D::get_value(inputs[1]);

    int width = Integer::get_value(inputs[2]);
    int height = Integer::get_value(inputs[3]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate_convert(interpolation), border_const, bvalue);
    } catch (cv::Exception& cve) {
        throw VariableException(cve.what());
    }

    return std::make_shared<Image>(output);

}

REGISTER_OPERATION_FUNCTION("view", ViewImage, Interpolation, BorderStrategy);

/**
 * @brief Apply view linear transformation to an image.
 * 
 */
SharedVariable RemapImage(std::vector<SharedVariable> inputs, Interpolation interpolation, BorderStrategy border) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    int border_value = 0;
    int border_const = 0;

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
        throw VariableException("Illegal border strategy value");
    }

    cv::Mat image = Image::get_value(inputs[0]);
    cv::Mat x = Image::get_value(inputs[1]);
    cv::Mat y = Image::get_value(inputs[2]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::remap(image, output, x, y, interpolate_convert(interpolation), border_const, bvalue);
    } catch (cv::Exception& cve) {
        throw VariableException(cve.what());
    }

    return std::make_shared<Image>(output);

}

REGISTER_OPERATION_FUNCTION("remap", RemapImage, Interpolation, BorderStrategy);

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
        return {(float)left, (float)top, (float)right + 1, (float)bottom + 1};

}

SharedVariable MaskBounds(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);

    VERIFY(image.channels() == 1, "Image has more than one channel");

    float maxval = maximum_value(image);

    switch (image.depth()) {
        case CV_8U: 
            return std::make_shared<FloatList>(bounds<uchar>(image));
        case CV_16S: 
            return std::make_shared<FloatList>(bounds<short>(image));
        case CV_32F: 
            return std::make_shared<FloatList>(bounds<float>(image));
        case CV_64F: 
            return std::make_shared<FloatList>(bounds<double>(image));
        default:
            throw VariableException("Unsupported depth");
    }

}

REGISTER_OPERATION_FUNCTION("bounds", MaskBounds);

/**
 * @brief Performs image resize.
 * 
 */
SharedVariable ImageResize(std::vector<SharedVariable> inputs, Interpolation interpolation) {

    if (inputs.size() == 3) {

        cv::Mat image = Image::get_value(inputs[0]);

        int width = Integer::get_value(inputs[1]);
        int height = Integer::get_value(inputs[2]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(height, width), 0, 0, interpolate_convert(interpolation));

        return std::make_shared<Image>(result);

    } else if (inputs.size() == 2) {

        cv::Mat image = Image::get_value(inputs[0]);

        int scale = Float::get_value(inputs[1]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(), scale, scale, interpolate_convert(interpolation));

        return std::make_shared<Image>(result);

    } else {
        throw VariableException("Incorrect number of parameters");
    }
}

REGISTER_OPERATION_FUNCTION("resize", ImageResize, Interpolation);

/**
 * @brief Rotates an image without cropping.
 * 
 */
SharedVariable Rotate(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int clockwise = Integer::get_value(inputs[1]);
    
    cv::Mat result;

    if (clockwise == 1){
        cv::transpose(image, result);  
        cv::flip(result, result, 1);
    }else if(clockwise == -1){
        cv::transpose(image, result);  
        cv::flip(result, result, 0);
    }else{    
        cv::flip(image, result, -1);
    }

    return std::make_shared<Image>(result);
}  

REGISTER_OPERATION_FUNCTION("rotate", Rotate);


/**
 * @brief Flips a 2D array around vertical, horizontal, or both axes.
 * 
 */
SharedVariable Flip(std::vector<SharedVariable> inputs) noexcept(false) {
    
    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int flip_code = Integer::get_value(inputs[1]);
    
    cv::Mat result;
    cv::flip(image, result, flip_code); 

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("flip", Flip);

/**
 * @brief Returns a bounding box of custom size.
 * 
 */
SharedVariable ImageCrop(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[1], FloatType), "Not a float list");

    cv::Mat image = Image::get_value(inputs[0]);
    auto bbox = std::static_pointer_cast<List>(inputs[1]);

    float x1 = Float::get_value(bbox->get(0));
    float y1 = Float::get_value(bbox->get(1));
    float x2 = Float::get_value(bbox->get(2));
    float y2 = Float::get_value(bbox->get(3));
    
    cv::Mat result = image(cv::Rect(x1,y1,x2-x1,y2-y1)).clone();

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("crop", ImageCrop);


}
