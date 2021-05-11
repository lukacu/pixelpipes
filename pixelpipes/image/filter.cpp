
#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>

/*
FILTER AND BLURING OPERATIONS
*/

namespace pixelpipes {

/**
 * @brief Blurs an image using a gaussian filter.
 * 
 */
SharedVariable GaussianBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 5, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size_x = Integer::get_value(inputs[1]);
    int size_y = Integer::get_value(inputs[2]);
    float sigma_x = Float::get_value(inputs[3]);
    float sigma_y = Float::get_value(inputs[4]);

    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(size_y, size_x), sigma_x, sigma_y, cv::BORDER_REPLICATE);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("gaussian_blur", GaussianBlur);

/**
 * @brief Blurs an image using a median filter.
 * 
 */
SharedVariable MedianBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    cv::medianBlur(image, result, size);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("median_blur", MedianBlur);


/**
 * @brief Convolving an image with a normalized box filter.
 * 
 */
SharedVariable AverageBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    cv::blur(image, result, cv::Size(size, size));

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("average_blur", AverageBlur);

/**
 * @brief Applies the bilateral filter to an image.
 * 
 */
SharedVariable BilateralFilter(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {
    
    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int d = Integer::get_value(inputs[1]);
    float sigma_color = Float::get_value(inputs[2]);
    float sigma_space = Float::get_value(inputs[3]);

    cv::Mat result;
    cv::bilateralFilter(image, result, d, (double)sigma_color, (double)sigma_space);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("bilateral_filter", BilateralFilter);


/**
 * @brief Convolves an image with custom kernel.
 * 
 */
SharedVariable LinearFilter(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    //cv::Mat kernel = Image::get_value(inputs[1]);
    // TEMP TEST FIX
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_32F)/(float)(25);

    VERIFY(kernel.channels() == 1, "Kernel has more than one channel");

    cv::Mat result;
    cv::filter2D(image, result, image.depth(), kernel);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("linear_filter", LinearFilter);


}

