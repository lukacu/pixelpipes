
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

    PIXELPIPES_OPERATION_AUTO("polygon_mask", polygon_mask);

    /*
    NOISE GENERATION
    */

    /**
     * @brief Creates a single channel image with values sampled from normal distribution.
     *
     */
    TokenReference normal_noise(int width, int height, float mean, float std, int seed) noexcept(false)
    {

        cv::RNG generator(seed);
        cv::Mat noise(height, width, CV_32F);
        generator.fill(noise, cv::RNG::NORMAL, mean, std);

        return wrap(noise);
    }

    PIXELPIPES_OPERATION_AUTO("normal_noise", normal_noise);

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

    PIXELPIPES_OPERATION_AUTO("uniform_noise", uniform_noise);

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

    PIXELPIPES_OPERATION_AUTO("linear_image", linear_image);

}
