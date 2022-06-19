
#include <opencv2/imgproc.hpp>

#include "common.hpp"

namespace pixelpipes
{

    inline int interpolate_convert(Interpolation interpolation)
    {

        switch (interpolation)
        {
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

    cv::Mat transpose(const cv::Mat &image) noexcept(false)
    {

        cv::Mat result;

        cv::transpose(image, result);

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("transpose", transpose);

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    cv::Mat view(const cv::Mat& image, const cv::Matx33f transform, int width, int height, Interpolation interpolation, BorderStrategy border)
    {
        int border_value = 0;
        int border_const = 0;

        border_const = ocv_border_type(border, &border_value);

        cv::Mat output;
        int bvalue = maximum_value(image) * border_value;

        CV_EX_WRAP(cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate_convert(interpolation), border_const, bvalue));
        
        return output;
    }

    PIXELPIPES_OPERATION_AUTO("view", view);

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    cv::Mat remap(const cv::Mat& image, const cv::Mat& x, const cv::Mat& y, Interpolation interpolation, BorderStrategy border)
    {

        int border_value = 0;
        int border_const = 0;

        border_const = ocv_border_type(border, &border_value);

        cv::Mat output;

        int bvalue = maximum_value(image) * border_value;

        CV_EX_WRAP(cv::remap(image, output, x, y, interpolate_convert(interpolation), border_const, bvalue));
        
        return output;
    }

    PIXELPIPES_OPERATION_AUTO("remap", remap);

    template <typename T>
    Rectangle bounds(const cv::Mat& image)
    {

        int top = std::numeric_limits<int>::max();
        int bottom = std::numeric_limits<int>::lowest();
        int left = std::numeric_limits<int>::max();
        int right = std::numeric_limits<int>::lowest();

        for (int y = 0; y < image.rows; y++)
        {
            const T *p = image.ptr<T>(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (p[x])
                {
                    top = std::min(top, y);
                    left = std::min(left, x);
                    bottom = std::max(bottom, y);
                    right = std::max(right, x);
                }
            }
        }

        if (top > bottom)
            return {(float)0, (float)0, (float)image.cols, (float)image.rows};
        else
            return {(float)left, (float)top, (float)right + 1, (float)bottom + 1};
    }

    Rectangle mask_bounds(const cv::Mat& image) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");

        // float maxval = maximum_value(image); TOD: what to do with this?

        switch (image.depth())
        {
        case CV_8U:
            return bounds<uchar>(image);
        case CV_8S:
            return bounds<uchar>(image);
        case CV_16S:
            return bounds<short>(image);
        case CV_16U:
            return bounds<ushort>(image);
        case CV_32F:
            return bounds<float>(image);
        case CV_64F:
            return bounds<double>(image);
        default:
            throw TypeException("Unsupported depth");
        }
    }

    PIXELPIPES_OPERATION_AUTO("mask_bounds", mask_bounds);

    /**
     * @brief Performs image resize.
     *
     * TODO: split into two operations
     *
     */
    cv::Mat resize(const cv::Mat &image, int width, int height, Interpolation interpolation)
    {

        cv::Mat result;
        cv::resize(image, result, cv::Size(width, height), 0, 0, interpolate_convert(interpolation));

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("resize", resize);

    cv::Mat rescale(const cv::Mat &image, float scale, Interpolation interpolation)
    {

        cv::Mat result;
        cv::resize(image, result, cv::Size(), scale, scale, interpolate_convert(interpolation));

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("rescale", rescale);

    /**
     * @brief Rotates an image without cropping.
     *
     */
    cv::Mat rotate90(const cv::Mat &image, int clockwise) noexcept(false)
    {

        cv::Mat result;

        if (clockwise == 1)
        {
            cv::transpose(image, result);
            cv::flip(result, result, 1);
        }
        else if (clockwise == -1)
        {
            cv::transpose(image, result);
            cv::flip(result, result, 0);
        }
        else
        {
            cv::flip(image, result, -1);
        }

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("rotate90", rotate90);

    /**
     * @brief Flips a 2D array around vertical, horizontal, or both axes.
     *
     */
    cv::Mat flip(const cv::Mat& image, bool horizontal, bool vertical) noexcept(false)
    {

        cv::Mat result;

        if (horizontal)
        {
            if (vertical)
            {
                cv::flip(image, result, -1);
            }
            else
            {
                cv::flip(image, result, 1);
            }
        }
        else
        {
            if (!vertical)
            {
                result = image;
            }
            else
            {
                cv::flip(image, result, 0);
            }
        }

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("flip", flip);

    /**
     * @brief Returns a bounding box of custom size.
     *
     */
    cv::Mat crop(const cv::Mat &image, cv::Rect bbox)
    {
        try                                            
        {                                              
            return image(bbox).clone();                                  
        }                                              
        catch (cv::Exception & e)                      
        {                                              
            throw pixelpipes::TypeException(e.what()); 
        }

    }

    PIXELPIPES_OPERATION_AUTO("crop", crop);

}
