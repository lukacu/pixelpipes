
#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>

#include "common.hpp"

/*
FILTER AND BLURING OPERATIONS
*/

namespace pixelpipes {

/**
 * @brief Blurs an image using a median filter.
 * 
 */
SharedToken MedianBlur(std::vector<SharedToken> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    CV_EX_WRAP(cv::medianBlur(image, result, size));

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("median_blur", MedianBlur);


/**
 * @brief Convolving an image with a normalized box filter.
 * 
 */
SharedToken AverageBlur(std::vector<SharedToken> inputs) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    CV_EX_WRAP(cv::blur(image, result, cv::Size(size, size)) );

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("average_blur", AverageBlur);

/**
 * @brief Applies the bilateral filter to an image.
 * 
 */
SharedToken BilateralFilter(std::vector<SharedToken> inputs) noexcept(false) {
    
    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    int d = Integer::get_value(inputs[1]);
    float sigma_color = Float::get_value(inputs[2]);
    float sigma_space = Float::get_value(inputs[3]);

    cv::Mat result;
    CV_EX_WRAP(cv::bilateralFilter(image, result, d, (double)sigma_color, (double)sigma_space));

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("bilateral_filter", BilateralFilter);


/**
 * @brief Convolves an image with custom kernel.
 * 
 */
SharedToken LinearFilter(std::vector<SharedToken> inputs, BorderStrategy border) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = extract<cv::Mat>(inputs[0]);
    cv::Mat kernel = extract<cv::Mat>(inputs[1]);

    int border_value = 0;
    int border_const = 0;

    border_const = ocv_border_type(border, &border_value);

    VERIFY(kernel.channels() == 1, "Kernel has more than one channel");

    cv::Mat result;
    CV_EX_WRAP(cv::filter2D(image, result, image.depth(), kernel, cv::Point(-1, -1), 0, border_const));

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("linear_filter", LinearFilter, BorderStrategy);


/**
 * @brief Convolves an image with custom kernel.
 * 
 */
SharedToken GaussianKernel(std::vector<SharedToken> inputs) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    int size = Integer::get_value(inputs[0]);

    VERIFY(size > 0 && size % 2 == 1, "Incorrect kernel size");

    cv::Mat result;
    CV_EX_WRAP(result = cv::getGaussianKernel(size, 0););

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("gaussian_kernel", GaussianKernel);


SharedToken UniformKernel(std::vector<SharedToken> inputs) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    int size = Integer::get_value(inputs[0]);

    VERIFY(size > 0 && size % 2 == 1, "Incorrect kernel size");

    cv::Mat result = cv::Mat::ones(cv::Size(size, 1), CV_32F) / (float) size;
    
    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("uniform_kernel", UniformKernel);

}

