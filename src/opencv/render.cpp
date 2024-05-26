
#include "common.hpp"

#include <opencv2/imgproc.hpp>

namespace pixelpipes
{

    /**
     * @brief Draw a polygon to a canvas of a given size.
     *
     */
    TokenReference polygon_mask(const std::vector<cv::Point2f>& points, int width, int height, int thickness) noexcept(false)
    {

        try
        {

            std::vector<cv::Point> v(points.begin(), points.end());

            VERIFY(v.size() > 2, "Polygon must have at least 3 points");
            
            cv::Mat mat = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

            if (thickness > 0)
            {
                cv::polylines(mat, std::vector<std::vector<cv::Point>>({v}), true, cv::Scalar(255, 255, 255), thickness);
            } else {
                cv::fillPoly(mat, std::vector<std::vector<cv::Point>>({v}), cv::Scalar(255, 255, 255));
            }

            return wrap(mat);
        }
        catch (cv::Exception &cve)
        {
            throw TypeException(cve.what());
        }
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("polygon_mask", polygon_mask, (given_shape<1, 2, CharType, 1>));

    /**
     * @brief Draw points on a canvas of a given size.
     *
     */
    TokenReference points_mask(const std::vector<cv::Point2f>& points, int width, int height, int size) noexcept(false)
    {

        try
        {

            std::vector<cv::Point> v(points.begin(), points.end());

            cv::Mat mat = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

            // Draw points to a canvas
            for (auto p : v)
            {
                cv::circle(mat, p, size, cv::Scalar(255, 255, 255), -1);
            }

            return wrap(mat);
        }
        catch (cv::Exception &cve)
        {
            throw TypeException(cve.what());
        }
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("points_mask", points_mask, (given_shape<1, 2, CharType, 1>));



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

    PIXELPIPES_COMPUTE_OPERATION_AUTO("gaussian_noise", gaussian_noise, (given_shape<0, 1, FloatType, 1>));

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

    PIXELPIPES_COMPUTE_OPERATION_AUTO("uniform_noise", uniform_noise, (given_shape<0, 1, FloatType, 1>));


    /**
     * @brief Creates a single channel image with binary values
     *
     */
    TokenReference binary_noise(int width, int height, float positive, int seed) noexcept(false)
    {

        VERIFY(positive >= 0.0 && positive <= 1.0, "Positives percentage value must be between 0 and 1");

        RandomGenerator generator = create_generator(seed);
        cv::Mat noise(height, width, CV_8U);

        // Generate a vector of increasing 1D image indices
        std::vector<size_t> indices(width * height);
        for (size_t i = 0; i < noise.total(); i++)
        {
            indices[i] = i;
        }

        // Shuffle the indices
        std::shuffle(indices.begin(), indices.end(), generator);

        // Set the first positive% of the indices to 1
        for (int i = 0; i < positive * noise.total(); i++)
        {
            noise.at<uchar>(indices[i]) = 255;
        }

        return wrap(noise);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("binary_noise", binary_noise, (given_shape<0, 1, FloatType, 1>));


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

    PIXELPIPES_COMPUTE_OPERATION_AUTO("linear_image", linear_image, (given_shape<0, 1, FloatType, 1>));



    /*
     * @brief Tabulates a mixture of Gaussian functions.
     *
     * 
     *
     */
    cv::Mat map_mixture(int width, int height) noexcept(false)
    {
        cv::Mat result(height, width, CV_32F);
        result.setTo(0);



        return result;
    }   

    PIXELPIPES_COMPUTE_OPERATION_AUTO("map_mixture", map_mixture, (given_shape<0, 1, FloatType, 1>));
}
