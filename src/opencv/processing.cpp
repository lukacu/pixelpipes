
#include "common.hpp"

#include <opencv2/imgproc.hpp>

namespace pixelpipes
{

    // TODO: make all element type specific loops templated

    /**
     * @brief Converts images between color spaces
     *
     */
    cv::Mat convert_color_run(const cv::Mat &image, ColorConversion convert) noexcept(false)
    {
        int code = -1;
        switch (convert)
        {
        case ColorConversion::RGB_GRAY:
            code = cv::COLOR_RGB2GRAY;
            break;
        case ColorConversion::GRAY_RGB:
            code = cv::COLOR_GRAY2RGB;
            break;
        case ColorConversion::RGB_HSV:
            code = cv::COLOR_RGB2HSV;
            break;
        case ColorConversion::HSV_RGB:
            code = cv::COLOR_HSV2RGB;
            break;
        case ColorConversion::RGB_YCRCB:
            code = cv::COLOR_RGB2YCrCb;
            break;
        case ColorConversion::YCRCB_RGB:
            code = cv::COLOR_YCrCb2RGB;
            break;
        }

        cv::Mat result;
        cv::cvtColor(image, result, code);
        return result;
    }

    TokenReference convert_color_eval(const TokenList& inputs) noexcept(false)
    {
        VERIFY(inputs.size() == 2, "Color conversion requires two inputs");

        auto shape = inputs[0]->shape();
        auto convert = extract<ColorConversion>(inputs[1]);

        switch (convert)
        {
        case ColorConversion::RGB_GRAY:
            return create<Placeholder>(Shape(shape.element(), {shape[0], shape[1]}));
        case ColorConversion::GRAY_RGB:
            return create<Placeholder>(Shape(shape.element(), {3, shape[0], shape[1]}));
        case ColorConversion::RGB_HSV:
            return create<Placeholder>(shape);
        case ColorConversion::HSV_RGB:
            return create<Placeholder>(shape);
        case ColorConversion::RGB_YCRCB:
            return create<Placeholder>(shape);
        case ColorConversion::YCRCB_RGB:
            return create<Placeholder>(shape);
        }

        throw TypeException("Unknown color space conversion");
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("color", convert_color_run, convert_color_eval);

    /**
     * @brief Converts depth of an image, scaling values in the process
     *
     */
    TokenReference convert_depth_run(const TensorReference &input, ImageDepth depth) noexcept(false)
    {

        cv::Mat image = wrap_tensor(input);

        double maxin = maximum_value(image);
        int dtype = -1;
        Type ti;
        double maxout = 1;

        switch (depth)
        {
        case ImageDepth::Char:
            dtype = CV_8U;
            maxout = std::numeric_limits<uchar>::max();
            ti = CharType;
            break;
        case ImageDepth::Short:
            dtype = CV_16S;
            maxout = std::numeric_limits<short>::max();
            ti = ShortType;
            break;
        case ImageDepth::UShort:
            dtype = CV_16U;
            maxout = std::numeric_limits<ushort>::max();
            ti = UnsignedShortType;
            break;
        case ImageDepth::Integer:
            dtype = CV_32S;
            maxout = std::numeric_limits<int>::max();
            ti = IntegerType;
            break;
        case ImageDepth::Float:
            dtype = CV_32F;
            maxout = 1;
            ti = FloatType;
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required
            return input.reborrow();
        }

        TensorReference result = create_tensor(input->shape().cast(ti));

        cv::Mat out = wrap_tensor(result);

        image.convertTo(out, dtype, maxout / maxin);

        return result;
    }

    TokenReference convert_depth_eval(const TokenList& inputs) noexcept(false)
    {
        VERIFY(inputs.size() == 2, "Depth conversion requires two inputs");

        auto shape = inputs[0]->shape();
        auto depth = extract<ImageDepth>(inputs[1]);

        switch (depth)
        {
        case ImageDepth::Char:
            return create<Placeholder>(shape.cast(CharType));
        case ImageDepth::Short:
            return create<Placeholder>(shape.cast(ShortType));
        case ImageDepth::UShort:
            return create<Placeholder>(shape.cast(UnsignedShortType));
        case ImageDepth::Integer:
            return create<Placeholder>(shape.cast(IntegerType));
        case ImageDepth::Float:
            return create<Placeholder>(shape.cast(FloatType));
        }

        throw TypeException("Unknown depth type conversion");
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("convert_depth", convert_depth_run, convert_depth_eval);

    /**
     * @brief Thresholds an image.
     *
     */
    TokenReference threshold(const cv::Mat &image, float threshold) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");
        float maxval = maximum_value(image);

        TensorReference result = create_tensor(image);

        cv::Mat out = wrap_tensor(result);

        cv::threshold(image, out, threshold, maxval, cv::THRESH_BINARY);
        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("threshold", threshold, (forward_shape<0>));

    /**
     * @brief Inverts image pixel values.
     *
     */
    TokenReference invert(const cv::Mat &image) noexcept(false)
    {
        float maxval = maximum_value(image);

        TensorReference result = create_tensor(image);

        cv::Mat out = wrap_tensor(result);

        cv::subtract(maxval, image, out);

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("invert", invert, forward_shape<0>);

    /**
     * @brief Calculates image moments.
     *
     */
    Sequence<float> moments(const cv::Mat &image, bool binary) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");

        cv::Moments m = cv::moments(image, binary);

        Sequence<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

        return data;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("moments", moments, (constant_shape<float, 4>));

    /**
     * @brief Blends two images using alpha.
     *
     */
    cv::Mat blend(const cv::Mat &a, const cv::Mat &b, float alpha)
    {

        float beta = (1 - alpha);
        cv::Mat result;
        cv::addWeighted(a, alpha, b, beta, 0.0, result);
        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("blend", blend, (forward_shape<0>));

    TokenReference normalize(const cv::Mat &image)
    {

        VERIFY(image.channels() == 1, "Only single channel images accepted");

        int maxv = maximum_value(image);

        TensorReference result = create_tensor(image);

        cv::Mat out = wrap_tensor(result);

        cv::normalize(image, out, 0, maxv, cv::NORM_MINMAX);

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("normalize", normalize, (forward_shape<0>));

    /**
     * @brief Sets image pixels to zero with probability P.
     *
     */
    cv::Mat dropout(const cv::Mat &image, float dropout_p, int seed) noexcept(false)
    {

        cv::RNG generator(seed);
        cv::Mat result = image.clone();

        if (result.channels() == 1)
        {
            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    if (generator.uniform(0.0, 1.0) < dropout_p)
                    {
                        if (result.depth() == CV_8U)
                        {
                            result.at<uchar>(y, x) = 0;
                        }
                        else if (result.depth() == CV_32F)
                        {
                            result.at<float>(y, x) = 0.0;
                        }
                        else if (result.depth() == CV_64F)
                        {
                            result.at<double>(y, x) = 0.0;
                        }
                    }
                }
            }
        }
        else if (result.channels() == 3)
        {
            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    if (generator.uniform(0.0, 1.0) < dropout_p)
                    {
                        if (result.depth() == CV_8U)
                        {
                            cv::Vec3b &color = result.at<cv::Vec3b>(y, x);
                            color[0] = 0;
                            color[1] = 0;
                            color[2] = 0;
                        }
                        else if (result.depth() == CV_32F)
                        {
                            cv::Vec3f &color = result.at<cv::Vec3f>(y, x);
                            color[0] = 0.0;
                            color[1] = 0.0;
                            color[2] = 0.0;
                        }
                        else if (result.depth() == CV_64F)
                        {
                            cv::Vec3d &color = result.at<cv::Vec3d>(y, x);
                            color[0] = 0.0;
                            color[1] = 0.0;
                            color[2] = 0.0;
                        }
                    }
                }
            }
        }

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("dropout", dropout, (forward_shape<0>));

    /**
     * @brief Divides image to pacthes and sets patch pixels to zero with probability P.
     *
     */
    cv::Mat coarse_dropout(const cv::Mat &image, float dropout_p, float dropout_size, int seed) noexcept(false)
    {

        cv::Mat result = image.clone();

        int patch_size_x = (int)(result.cols * dropout_size);
        int patch_size_y = (int)(result.rows * dropout_size);
        int num_patches_x = (int)(1 / dropout_size);
        int num_patches_y = (int)(1 / dropout_size);

        cv::RNG generator(seed);

        if (result.channels() == 1)
        {
            for (int yp = 0; yp < num_patches_y; yp++)
            {
                for (int xp = 0; xp < num_patches_x; xp++)
                {
                    if (generator.uniform(0.0, 1.0) < dropout_p)
                    {
                        for (int y = 0; y < patch_size_y; y++)
                        {
                            for (int x = 0; x < patch_size_x; x++)
                            {

                                int iy = y + yp * patch_size_y;
                                int ix = x + xp * patch_size_x;

                                if (result.depth() == CV_8U)
                                {
                                    result.at<uchar>(iy, ix) = 0;
                                }
                                else if (result.depth() == CV_32F)
                                {
                                    result.at<float>(iy, ix) = 0.0;
                                }
                                else if (result.depth() == CV_64F)
                                {
                                    result.at<double>(iy, ix) = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        else if (result.channels() == 3)
        {
            for (int yp = 0; yp < num_patches_y; yp++)
            {
                for (int xp = 0; xp < num_patches_x; xp++)
                {
                    if (generator.uniform(0.0, 1.0) < dropout_p)
                    {
                        for (int y = 0; y < patch_size_y; y++)
                        {
                            for (int x = 0; x < patch_size_x; x++)
                            {

                                int iy = y + yp * patch_size_y;
                                int ix = x + xp * patch_size_x;

                                if (result.depth() == CV_8U)
                                {
                                    cv::Vec3b &color = result.at<cv::Vec3b>(iy, ix);
                                    color[0] = 0;
                                    color[1] = 0;
                                    color[2] = 0;
                                }
                                else if (result.depth() == CV_32F)
                                {
                                    cv::Vec3f &color = result.at<cv::Vec3f>(iy, ix);
                                    color[0] = 0.0;
                                    color[1] = 0.0;
                                    color[2] = 0.0;
                                }
                                else if (result.depth() == CV_64F)
                                {
                                    cv::Vec3d &color = result.at<cv::Vec3d>(iy, ix);
                                    color[0] = 0.0;
                                    color[1] = 0.0;
                                    color[2] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("coarse_dropout", coarse_dropout, forward_shape<0>);

    /**
     * @brief Cuts region form an image defined by the bounding box.
     *
     */
    cv::Mat cut(const cv::Mat &image, const Rectangle &region)
    {

        cv::Mat result = image.clone();

        if (result.channels() == 1)
        {
            for (int y = (int)region.top; y < (int)region.bottom; y++)
            {
                for (int x = (int)region.left; x < (int)region.right; x++)
                {
                    if (result.depth() == CV_8U)
                    {
                        result.at<uchar>(y, x) = 0;
                    }
                    else if (result.depth() == CV_32F)
                    {
                        result.at<float>(y, x) = 0.0;
                    }
                    else if (result.depth() == CV_64F)
                    {
                        result.at<double>(y, x) = 0.0;
                    }
                }
            }
        }

        else if (result.channels() == 3)
        {
            for (int y = (int)region.top; y < (int)region.bottom; y++)
            {
                for (int x = (int)region.left; x < (int)region.right; x++)
                {
                    if (result.depth() == CV_8U)
                    {
                        cv::Vec3b &color = result.at<cv::Vec3b>(y, x);
                        color[0] = 0;
                        color[1] = 0;
                        color[2] = 0;
                    }
                    else if (result.depth() == CV_32F)
                    {
                        cv::Vec3f &color = result.at<cv::Vec3f>(y, x);
                        color[0] = 0.0;
                        color[1] = 0.0;
                        color[2] = 0.0;
                    }
                    else if (result.depth() == CV_64F)
                    {
                        cv::Vec3d &color = result.at<cv::Vec3d>(y, x);
                        color[0] = 0.0;
                        color[1] = 0.0;
                        color[2] = 0.0;
                    }
                }
            }
        }

        return result;
    }

    //PIXELPIPES_OPERATION_AUTO("cut", cut);

    /**
     * @brief Inverts all values above a threshold in image.
     *
     */
    cv::Mat solarize(const cv::Mat &image, float threshold)
    {

        VERIFY(image.channels() == 1, "Image has more than one channel");

        float max = (float) maximum_value(image);

        cv::Mat result = image.clone();

        if (result.channels() == 1)
        {
            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    if (result.depth() == CV_8U)
                    {
                        if (result.at<uchar>(y, x) > (int)threshold)
                        {
                            result.at<uchar>(y, x) = max - result.at<uchar>(y, x);
                        }
                    }
                    else if (result.depth() == CV_32F)
                    {
                        if (result.at<float>(y, x) > threshold)
                        {
                            result.at<float>(y, x) = max - result.at<float>(y, x);
                        }
                    }
                    else if (result.depth() == CV_64F)
                    {
                        if (result.at<double>(y, x) > threshold)
                        {
                            result.at<double>(y, x) = max - result.at<double>(y, x);
                        }
                    }
                }
            }
        }

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("solarize", solarize, forward_shape<0>);

}