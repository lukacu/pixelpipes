#include <utility>

#define OPERATION_NAMESPACE "image"

#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>

#include "common.hpp"


namespace pixelpipes {

cv::Mat replicate_channels(cv::Mat &image, int channels) {

    cv::Mat out;
    std::vector<cv::Mat> in;
    for (int i = 0; i < channels; i++) in.push_back(image);
    cv::merge(in, out);

    return out;

}

cv::Scalar uniform_scalar(float value, int channels) {

    if (channels == 2) return cv::Scalar(value, value);
    if (channels == 3) return cv::Scalar(value, value, value);
    if (channels == 4) return cv::Scalar(value, value, value, value);
    
    return cv::Scalar(value);

}

std::pair<cv::Mat, cv::Mat> ensure_channels(cv::Mat &image1, cv::Mat &image2) {

    int channels1 = image1.channels();
    int channels2 = image2.channels();

    if (channels1 != channels2) {
        if (channels1 == 1) {
            image1 = replicate_channels(image1, channels2);
        } else if (channels2 == 1) {
            image2 = replicate_channels(image2, channels1);
        } else {
            throw VariableException("Channel count mismatch");
        }
    }

    return std::pair<cv::Mat, cv::Mat>(image1, image2);

}

/**
 * @brief Combines two images of same size (pixel-wise).
 * 
 */
SharedVariable ImageAdd(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(ImageData::is(inputs[0]) || ImageData::is(inputs[1]), "At least one input should be an image");

    cv::Mat result;

    // Both inputs are images
    if (ImageData::is(inputs[0]) && ImageData::is(inputs[1])) {

        cv::Mat image0 = extract<cv::Mat>(inputs[0]);
        cv::Mat image1 = extract<cv::Mat>(inputs[1]);

        VERIFY(image0.rows == image1.rows && image0.cols == image1.cols, "Image sizes do not match");

        auto image_pair = ensure_channels(image0, image1);
        image0 = image_pair.first;
        image1 = image_pair.second;

        cv::add(image0, image1, result); // cv::add potentially saturates data type

    } else {

        if (ImageData::is(inputs[0])) {

            cv::Mat image = extract<cv::Mat>(inputs[0]);
            float value = Float::get_value(inputs[1]);

            result = image + uniform_scalar((value), image.channels()); // TODO: scaling based on input

        } else {

            cv::Mat image = extract<cv::Mat>(inputs[1]);
            float value = Float::get_value(inputs[0]);

            result = image + uniform_scalar((value), image.channels());

        }
    }

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("add", ImageAdd);


/**
 * @brief Subtracts two images.
 * 
 */
SharedVariable ImageSubtract(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(ImageData::is(inputs[0]) || ImageData::is(inputs[1]), "At least one input should be an image");

    cv::Mat result;

    // Both inputs are images
    if (ImageData::is(inputs[0]) && ImageData::is(inputs[1])) {

        cv::Mat image0 = extract<cv::Mat>(inputs[0]);
        cv::Mat image1 = extract<cv::Mat>(inputs[1]);

        VERIFY(image0.rows == image1.rows && image0.cols == image1.cols, "Image sizes do not match");

        auto image_pair = ensure_channels(image0, image1);
        image0 = image_pair.first;
        image1 = image_pair.second;

        cv::subtract(image0, image1, result); // cv::subtract potentially saturates data type

    } else {

        if (ImageData::is(inputs[0])) {

            cv::Mat image = extract<cv::Mat>(inputs[0]);
            float value = Float::get_value(inputs[1]);

            result = image - uniform_scalar((value), image.channels()); // TODO: scaling based on input
        } else {

            cv::Mat image = extract<cv::Mat>(inputs[1]);
            float value = Float::get_value(inputs[0]);
            
            result = image - uniform_scalar((value), image.channels()); // TODO: scaling based on input
        }
    }

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("subtract", ImageSubtract);

/**
 * @brief Multiplies image with a multiplier (number).
 * 
 */
SharedVariable ImageMultiply(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(ImageData::is(inputs[0]) || ImageData::is(inputs[1]), "At least one input should be an image");

    cv::Mat result;

    if (ImageData::is(inputs[0]) && ImageData::is(inputs[1])) {
        // Both inputs are images     
        cv::Mat image0 = extract<cv::Mat>(inputs[0]);
        cv::Mat image1 = extract<cv::Mat>(inputs[1]);

        VERIFY(image0.rows == image1.rows && image0.cols == image1.cols, "Image sizes do not match");

        auto image_pair = ensure_channels(image0, image1);
        image0 = image_pair.first;
        image1 = image_pair.second;

        cv::multiply(image0, image1, result, 1.0);
    } else {
        // One input is image, other input is scalar
        if (ImageData::is(inputs[0])) {
            cv::Mat image = extract<cv::Mat>(inputs[0]);
            float value = Float::get_value(inputs[1]);           
            result = value * image;
        } else {      
            cv::Mat image = extract<cv::Mat>(inputs[1]);
            float value = Float::get_value(inputs[0]);
            result = value * image;
        }
    }

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("multiply", ImageMultiply);

}
