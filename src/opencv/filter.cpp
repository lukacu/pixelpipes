#include "common.hpp"

#include <opencv2/imgproc.hpp>

#include <pixelpipes/tensor.hpp>

/*
FILTER AND BLURING OPERATIONS
*/

namespace pixelpipes
{

    /**
     * @brief Blurs an image using a median filter.
     *
     */
    cv::Mat median_blur(const cv::Mat &image, int size) noexcept(false)
    {
        cv::Mat result;
        CV_EX_WRAP(cv::medianBlur(image, result, size));
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("median_blur", median_blur);

    /**
     * @brief Convolving an image with a normalized box filter.
     *
     */
    cv::Mat average_blur(const cv::Mat &image, int size) noexcept(false)
    {
        cv::Mat result;
        CV_EX_WRAP(cv::blur(image, result, cv::Size(size, size)));
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("average_blur", average_blur);

    /**
     * @brief Applies the bilateral filter to an image.
     *
     */
    cv::Mat bilateral_filter(const cv::Mat &image, int d, float sigma_color, float sigma_space) noexcept(false)
    {
        cv::Mat result;
        CV_EX_WRAP(cv::bilateralFilter(image, result, d, (double)sigma_color, (double)sigma_space));
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("bilateral_filter", bilateral_filter);

    /**
     * @brief Convolves an image with custom kernel.
     *
     */
    cv::Mat linear_filter(const cv::Mat &image, const cv::Mat &kernel, BorderStrategy border) noexcept(false)
    {

        int border_value = 0;
        int border_const = 0;
        border_const = ocv_border_type(border, &border_value);

        VERIFY(kernel.channels() == 1, "Kernel has more than one channel");

        cv::Mat result;
        CV_EX_WRAP(cv::filter2D(image, result, image.depth(), kernel, cv::Point(-1, -1), 0, border_const));
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("linear_filter", linear_filter);

    /**
     * @brief Convolves an image with custom kernel.
     *
     */
    cv::Mat gaussian_kernel(int size) noexcept(false)
    {
        VERIFY(size > 0 && size % 2 == 1, "Incorrect kernel size");

        cv::Mat result;
        CV_EX_WRAP(result = cv::getGaussianKernel(size, 0, CV_32F););

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("gaussian_kernel", gaussian_kernel);

    cv::Mat uniform_kernel(int size) noexcept(false)
    {
        VERIFY(size > 0 && size % 2 == 1, "Incorrect kernel size");

        cv::Mat result = cv::Mat::ones(cv::Size(size, 1), CV_32F) / (float)size;

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("uniform_kernel", uniform_kernel);

}
