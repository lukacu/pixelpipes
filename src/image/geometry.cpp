
#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

#include "common.hpp"

namespace pixelpipes {


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

SharedToken Transpose(TokenList inputs) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);

    cv::Mat result;
    
    cv::transpose(image, result);  
       
    return wrap(result);
}  

REGISTER_OPERATION_FUNCTION("transpose", Transpose);


/**
 * @brief Apply view linear transformation to an image.
 * 
 */
SharedToken ViewImage(TokenList inputs, Interpolation interpolation, BorderStrategy border) {

    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int border_value = 0;
    int border_const = 0;

    border_const = ocv_border_type(border, &border_value);

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    cv::Matx33f transform = extract<cv::Matx33f>(inputs[1]);

    int width = Integer::get_value(inputs[2]);
    int height = Integer::get_value(inputs[3]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate_convert(interpolation), border_const, bvalue);
    } catch (cv::Exception& cve) {
        throw TypeException(cve.what());
    }

    return wrap(output);

}

REGISTER_OPERATION_FUNCTION("view", ViewImage, Interpolation, BorderStrategy);

/**
 * @brief Apply view linear transformation to an image.
 * 
 */
SharedToken RemapImage(TokenList inputs, Interpolation interpolation, BorderStrategy border) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    int border_value = 0;
    int border_const = 0;

    border_const = ocv_border_type(border, &border_value);

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    cv::Mat x = extract<cv::Mat>(inputs[1]);
    cv::Mat y = extract<cv::Mat>(inputs[2]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::remap(image, output, x, y, interpolate_convert(interpolation), border_const, bvalue);
    } catch (cv::Exception& cve) {
        throw TypeException(cve.what());
    }

    return wrap(output);

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

SharedToken MaskBounds(TokenList inputs) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);

    VERIFY(image.channels() == 1, "Image has more than one channel");

    //float maxval = maximum_value(image); TOD: what to do with this?

    switch (image.depth()) {
        case CV_8U: 
            return std::make_shared<FloatList>(make_span(bounds<uchar>(image)));
        case CV_16S: 
            return std::make_shared<FloatList>(make_span(bounds<short>(image)));
        case CV_32F: 
            return std::make_shared<FloatList>(make_span(bounds<float>(image)));
        case CV_64F: 
            return std::make_shared<FloatList>(make_span(bounds<double>(image)));
        default:
            throw TypeException("Unsupported depth");
    }

}

REGISTER_OPERATION_FUNCTION("bounds", MaskBounds);

/**
 * @brief Performs image resize.
 * 
 * TODO: split into two operations
 * 
 */
SharedToken ImageResize(TokenList inputs, Interpolation interpolation) {

    if (inputs.size() == 3) {

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        int width = Integer::get_value(inputs[1]);
        int height = Integer::get_value(inputs[2]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(width, height), 0, 0, interpolate_convert(interpolation));

        return wrap(result);

    } else if (inputs.size() == 2) {

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        float scale = Float::get_value(inputs[1]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(), scale, scale, interpolate_convert(interpolation));

        return wrap(result);

    } else {
        throw TypeException("Incorrect number of parameters");
    }
}

REGISTER_OPERATION_FUNCTION("resize", ImageResize, Interpolation);

/**
 * @brief Rotates an image without cropping.
 * 
 */
SharedToken Rotate(TokenList inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
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

    return wrap(result);
}  

REGISTER_OPERATION_FUNCTION("rotate", Rotate);


/**
 * @brief Flips a 2D array around vertical, horizontal, or both axes.
 * 
 */
SharedToken Flip(TokenList inputs, bool horizontal, bool vertical) noexcept(false) {
    
    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);

    cv::Mat result;

    if (horizontal) {
        if (vertical) {
            cv::flip(image, result, -1); 
        } else {
            cv::flip(image, result, 1); 
        }
    } else {
        if (!vertical) {
            result = image;
        } else {
            cv::flip(image, result, 0); 
        }
    }

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("flip", Flip, bool, bool);

/**
 * @brief Returns a bounding box of custom size.
 * 
 */
SharedToken ImageCrop(TokenList inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[1], FloatIdentifier), "Not a float list");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    auto bbox = std::static_pointer_cast<List>(inputs[1]);

    float x1 = Float::get_value(bbox->get(0));
    float y1 = Float::get_value(bbox->get(1));
    float x2 = Float::get_value(bbox->get(2));
    float y2 = Float::get_value(bbox->get(3));
    
    cv::Mat result = image(cv::Rect(x1,y1,x2-x1,y2-y1)).clone();

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("crop", ImageCrop);


}
