
#include "common.hpp"

#include <opencv2/imgproc.hpp>

namespace pixelpipes
{
    /**
     * @brief Draw a polygon to a canvas of a given size.
     *
     */
    TokenReference polygon_mask(const std::vector<cv::Point2f>& points, int width, int height) noexcept(false)
    {

        try
        {

            std::vector<cv::Point> v(points.begin(), points.end());

            cv::Mat mat = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

            // TODO: expose value?
            cv::fillPoly(mat, std::vector<std::vector<cv::Point>>({v}), cv::Scalar(255, 255, 255));

            return wrap(mat);
        }
        catch (cv::Exception &cve)
        {
            throw TypeException(cve.what());
        }
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("polygon_mask", polygon_mask, (given_shape<1, 2, CharType>));

    /*
    NOISE GENERATION
    */

    /**
     * @brief Creates a single channel image with values sampled from normal distribution.
     *
     */
    TokenReference gaussian_noise(int width, int height, float mean, float std, int seed) noexcept(false)
    {

        cv::RNG generator(seed);
        cv::Mat noise(height, width, CV_32F);
        generator.fill(noise, cv::RNG::NORMAL, mean, std);

        return wrap(noise);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("gaussian_noise", gaussian_noise, (given_shape<0, 1, FloatType>));

    /**
     * @brief Creates a single channel image with values sampled from uniform distribution.
     *
     */
    TokenReference uniform_noise(int width, int height, float min, float max, int seed) noexcept(false)
    {

        cv::RNG generator(seed);
        cv::Mat noise(height, width, CV_32F);
        generator.fill(noise, cv::RNG::UNIFORM, min, max);

        return wrap(noise);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("uniform_noise", uniform_noise, (given_shape<0, 1, FloatType>));

    cv::Mat linear_image(int width, int height, float min, float max, bool flip) noexcept(false)
    {

        cv::Mat result(height, width, CV_32F);

        float *data = &result.at<float>(0, 0);

        if (flip)
        {
            for (int y = 0; y < result.rows; y++)
            {
                float v = ((float)y / (float)(result.rows - 1)) * (max - min) + min;
                for (int x = 0; x < result.cols; x++)
                {
                    *data = v;
                    data++;
                }
            }
        }
        else
        {

            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    *data = ((float)x / (float)(result.cols - 1)) * (max - min) + min;
                    data++;
                }
            }
        }

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("linear_image", linear_image, (given_shape<0, 1, FloatType>));



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


}
