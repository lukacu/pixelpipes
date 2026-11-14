
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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
    cv::Mat convert_depth_run(const cv::Mat &image, DataType depth) noexcept(false)
    {

        double maxin = maximum_value(image);
        int dtype = -1;
        //Type ti;
        double maxout = 1;

        VERIFY(depth != DataType::Boolean, "Boolean depth conversion not supported");

        switch (depth)
        {
        case DataType::Char:
            dtype = CV_8U;
            maxout = std::numeric_limits<uchar>::max();
           // ti = CharType;
            break;
        case DataType::Short:
            dtype = CV_16S;
            maxout = std::numeric_limits<short>::max();
          //  ti = ShortType;
            break;
        case DataType::UnsignedShort:
            dtype = CV_16U;
            maxout = std::numeric_limits<ushort>::max();
          //  ti = UnsignedShortType;
            break;
        case DataType::Integer:
            dtype = CV_32S;
            maxout = std::numeric_limits<int>::max();
         //   ti = IntegerType;
            break;
        case DataType::Float:
            dtype = CV_32F;
            maxout = 1;
         //   ti = FloatType;
            break;
        case DataType::Boolean:
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required, should we return a copy?
            return image;
        }

        //TensorReference result = create_tensor(input->shape().cast(ti));

        cv::Mat out;

        image.convertTo(out, dtype, maxout / maxin);

        return out;
    }

    TokenReference convert_depth_eval(const TokenList& inputs) noexcept(false)
    {
        VERIFY(inputs.size() == 2, "Depth conversion requires two inputs");

        auto shape = inputs[0]->shape();
        auto depth = extract<DataType>(inputs[1]);

        VERIFY(depth != DataType::Boolean, "Boolean depth conversion not supported");

        switch (depth)
        {
        case DataType::Char:
            return create<Placeholder>(shape.cast(CharType));
        case DataType::Short:
            return create<Placeholder>(shape.cast(ShortType));
        case DataType::UnsignedShort:
            return create<Placeholder>(shape.cast(UnsignedShortType));
        case DataType::Integer:
            return create<Placeholder>(shape.cast(IntegerType));
        case DataType::Float:
            return create<Placeholder>(shape.cast(FloatType));
        case DataType::Boolean:
            break;
        }

        throw TypeException("Unknown depth type conversion");
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("convert_depth", convert_depth_run, convert_depth_eval);

    /**
     * @brief Thresholds an image.
     *
     */
    cv::Mat threshold(const cv::Mat &image, float threshold) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");
        float maxval = maximum_value(image);

        cv::Mat out;
        cv::threshold(image, out, threshold, maxval, cv::THRESH_BINARY);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("threshold", threshold, (forward_shape<0>));

    /**
     * @brief Inverts image pixel values.
     *
     */
    cv::Mat invert(const cv::Mat &image) noexcept(false)
    {
        float maxval = maximum_value(image);

        cv::Mat out;
        cv::subtract(maxval, image, out);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("invert", invert, forward_shape<0>);

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

    cv::Mat normalize(const cv::Mat &image)
    {

        VERIFY(image.channels() == 1, "Only single channel images accepted");

        int maxv = maximum_value(image);

        cv::Mat out;
        cv::normalize(image, out, 0, maxv, cv::NORM_MINMAX);
        return out;
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

    namespace
    {
        float sample_range(cv::RNG &generator, const Sequence<float> &range, const char *name)
        {
            VERIFY(range.size() == 2, Formatter() << name << " must contain two values");
            VERIFY(range[0] <= range[1], Formatter() << name << " lower bound exceeds upper bound");
            if (range[0] == range[1])
                return range[0];
            return generator.uniform(range[0], range[1]);
        }

        int sample_range(cv::RNG &generator, const Sequence<int> &range, const char *name)
        {
            VERIFY(range.size() == 2, Formatter() << name << " must contain two values");
            VERIFY(range[0] <= range[1], Formatter() << name << " lower bound exceeds upper bound");
            if (range[0] == range[1])
                return range[0];
            return generator.uniform(range[0], range[1] + 1);
        }

        cv::Mat to_unit(const cv::Mat &image)
        {
            cv::Mat result;
            image.convertTo(result, CV_MAKETYPE(CV_32F, image.channels()), 1.0 / maximum_value(image));
            return result;
        }

        cv::Mat from_unit(const cv::Mat &image, int depth, double maximum)
        {
            cv::Mat result;
            image.convertTo(result, CV_MAKETYPE(depth, image.channels()), maximum);
            return result;
        }

        void clip_unit(cv::Mat &image)
        {
            cv::min(image, 1.0, image);
            cv::max(image, 0.0, image);
        }

        void apply_contrast(cv::Mat &image, float factor)
        {
            const cv::Scalar mean = cv::mean(image);
            const double average = std::accumulate(mean.val, mean.val + image.channels(), 0.0) / image.channels();
            image = image * factor + (1.0 - factor) * average;
        }

        void apply_saturation_hue(cv::Mat &image, float saturation, float hue)
        {
            VERIFY(image.channels() == 3, "Saturation and hue require an RGB image");

            cv::Mat hsv;
            cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);

            channels[1] *= saturation;
            cv::min(channels[1], 1.0, channels[1]);
            cv::max(channels[1], 0.0, channels[1]);

            for (int y = 0; y < channels[0].rows; y++)
            {
                float *row = channels[0].ptr<float>(y);
                for (int x = 0; x < channels[0].cols; x++)
                {
                    row[x] = std::fmod(row[x] + hue * 360.0f + 360.0f, 360.0f);
                }
            }

            cv::merge(channels, hsv);
            cv::cvtColor(hsv, image, cv::COLOR_HSV2RGB);
        }

        cv::Mat random_noise(cv::RNG &generator, cv::Size size, int channels, float low, float high,
                             const std::string &noise_type, bool per_channel)
        {
            const int noise_channels = per_channel ? channels : 1;
            cv::Mat noise(size, CV_MAKETYPE(CV_32F, noise_channels));

            if (noise_type == "uniform")
            {
                generator.fill(noise, cv::RNG::UNIFORM, low, high);
            }
            else if (noise_type == "gaussian")
            {
                const float mean = (low + high) * 0.5f;
                const float sigma = std::abs(high - low) * 0.5f;
                generator.fill(noise, cv::RNG::NORMAL, mean, sigma);
            }
            else if (noise_type == "laplace")
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                std::mt19937 engine((uint32_t)generator.next());
                const float mean = (low + high) * 0.5f;
                const float scale = std::abs(high - low) / std::sqrt(2.0f);
                float *data = noise.ptr<float>();
                for (size_t i = 0; i < noise.total() * noise.channels(); i++)
                {
                    const float u = uniform(engine) - 0.5f;
                    data[i] = mean - scale * (u < 0 ? -1.0f : 1.0f) * std::log(1.0f - 2.0f * std::abs(u));
                }
            }
            else if (noise_type == "beta")
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                std::mt19937 engine((uint32_t)generator.next());
                float *data = noise.ptr<float>();
                for (size_t i = 0; i < noise.total() * noise.channels(); i++)
                {
                    const float a = uniform(engine);
                    const float b = uniform(engine);
                    data[i] = low + (high - low) * (a / std::max(a + b, 1e-7f));
                }
            }
            else
            {
                throw TypeException("Unknown additive noise type");
            }

            if (!per_channel && channels > 1)
            {
                std::vector<cv::Mat> split_channels(channels, noise);
                cv::merge(split_channels, noise);
            }
            return noise;
        }
    }

    cv::Mat random_brightness_contrast(const cv::Mat &image, const Sequence<float> &brightness_range,
                                       const Sequence<float> &contrast_range, bool brightness_by_max,
                                       bool ensure_safe_output, int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        const double maximum = maximum_value(image);
        const float brightness_delta = sample_range(generator, brightness_range, "brightness_range");
        const float contrast_delta = sample_range(generator, contrast_range, "contrast_range");
        const float contrast = 1.0f + contrast_delta;
        const float brightness = 1.0f + brightness_delta;
        cv::Mat result = to_unit(image);

        if (brightness_by_max)
        {
            result = result * contrast + brightness_delta;
        }
        else
        {
            apply_contrast(result, contrast);
            result *= brightness;
        }

        // The safe mode has the same visible result for already bounded inputs,
        // while the final clamp also protects integer conversion from overflow.
        UNUSED(ensure_safe_output);
        clip_unit(result);
        return from_unit(result, image.depth(), maximum);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("random_brightness_contrast", random_brightness_contrast, (forward_shape<0>));

    cv::Mat color_jitter(const cv::Mat &image, const Sequence<float> &brightness_range,
                         const Sequence<float> &contrast_range, const Sequence<float> &saturation_range,
                         const Sequence<float> &hue_range, int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        cv::Mat result = to_unit(image);
        std::vector<int> order{0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), std::mt19937((uint32_t)generator.next()));

        for (int operation : order)
        {
            switch (operation)
            {
            case 0:
                result *= sample_range(generator, brightness_range, "brightness_range");
                break;
            case 1:
                apply_contrast(result, sample_range(generator, contrast_range, "contrast_range"));
                break;
            case 2:
                if (result.channels() == 3)
                    apply_saturation_hue(result, sample_range(generator, saturation_range, "saturation_range"), 0);
                break;
            case 3:
                if (result.channels() == 3)
                    apply_saturation_hue(result, 1, sample_range(generator, hue_range, "hue_range"));
                break;
            }
        }

        clip_unit(result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("color_jitter", color_jitter, (forward_shape<0>));

    cv::Mat hue_saturation_value(const cv::Mat &image, const Sequence<float> &hue_range,
                                 const Sequence<float> &saturation_range, const Sequence<float> &value_range,
                                 int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        cv::Mat result = to_unit(image);
        const float hue = sample_range(generator, hue_range, "hue_shift_range") / 360.0f;
        const float saturation = sample_range(generator, saturation_range, "sat_shift_range") / maximum_value(image);
        const float value = sample_range(generator, value_range, "val_shift_range") / maximum_value(image);

        if (result.channels() == 3)
        {
            apply_saturation_hue(result, 1.0f, hue);
            cv::Mat hsv;
            cv::cvtColor(result, hsv, cv::COLOR_RGB2HSV);
            std::vector<cv::Mat> channels;
            cv::split(hsv, channels);
            channels[1] += saturation;
            channels[2] += value;
            cv::min(channels[1], 1.0, channels[1]);
            cv::max(channels[1], 0.0, channels[1]);
            cv::min(channels[2], 1.0, channels[2]);
            cv::max(channels[2], 0.0, channels[2]);
            cv::merge(channels, hsv);
            cv::cvtColor(hsv, result, cv::COLOR_HSV2RGB);
        }
        else
        {
            result += value;
        }

        clip_unit(result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("hue_saturation_value", hue_saturation_value, (forward_shape<0>));

    cv::Mat rgb_shift(const cv::Mat &image, const Sequence<float> &red_range,
                      const Sequence<float> &green_range, const Sequence<float> &blue_range,
                      int seed) noexcept(false)
    {
        VERIFY(image.channels() == 3, "RGBShift requires an RGB image");
        cv::RNG generator(seed);
        cv::Mat result = to_unit(image);
        std::vector<cv::Mat> channels;
        cv::split(result, channels);
        channels[0] += sample_range(generator, red_range, "r_shift_range") / maximum_value(image);
        channels[1] += sample_range(generator, green_range, "g_shift_range") / maximum_value(image);
        channels[2] += sample_range(generator, blue_range, "b_shift_range") / maximum_value(image);
        for (cv::Mat &channel : channels)
        {
            cv::min(channel, 1.0, channel);
            cv::max(channel, 0.0, channel);
        }
        cv::merge(channels, result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("rgb_shift", rgb_shift, (forward_shape<0>));

    cv::Mat random_gamma(const cv::Mat &image, const Sequence<float> &gamma_range, int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        const float gamma = sample_range(generator, gamma_range, "gamma_range") / 100.0f;
        cv::Mat result;
        cv::pow(to_unit(image), gamma, result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("random_gamma", random_gamma, (forward_shape<0>));

    cv::Mat clahe(const cv::Mat &image, const Sequence<float> &clip_range,
                  const Sequence<int> &tile_grid_size, int seed) noexcept(false)
    {
        VERIFY(tile_grid_size.size() == 2, "tile_grid_size must contain two values");
        VERIFY(tile_grid_size[0] > 0 && tile_grid_size[1] > 0, "tile_grid_size values must be positive");
        cv::RNG generator(seed);
        const double clip_limit = sample_range(generator, clip_range, "clip_range");
        const int original_depth = image.depth();
        const double maximum = maximum_value(image);
        cv::Mat working;
        image.convertTo(working, CV_MAKETYPE(CV_8U, image.channels()), 255.0 / maximum);
        cv::Ptr<cv::CLAHE> equalizer = cv::createCLAHE(clip_limit, cv::Size(tile_grid_size[1], tile_grid_size[0]));

        if (working.channels() == 1)
        {
            cv::Mat result;
            equalizer->apply(working, result);
            result.convertTo(result, CV_MAKETYPE(original_depth, 1), maximum / 255.0);
            return result;
        }

        VERIFY(working.channels() == 3, "CLAHE supports grayscale or RGB images");
        cv::Mat lab;
        cv::cvtColor(working, lab, cv::COLOR_RGB2Lab);
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        equalizer->apply(channels[0], channels[0]);
        cv::merge(channels, lab);
        cv::cvtColor(lab, working, cv::COLOR_Lab2RGB);
        working.convertTo(working, CV_MAKETYPE(original_depth, working.channels()), maximum / 255.0);
        return working;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("clahe", clahe, (forward_shape<0>));

    cv::Mat channel_dropout(const cv::Mat &image, const Sequence<int> &drop_range,
                            float fill, int seed) noexcept(false)
    {
        if (image.channels() == 1)
            return image.clone();
        VERIFY(drop_range.size() == 2, "channel_drop_range must contain two values");
        VERIFY(drop_range[0] >= 0 && drop_range[1] < image.channels(), "Invalid channel_drop_range");
        cv::RNG generator(seed);
        const int count = sample_range(generator, drop_range, "channel_drop_range");
        VERIFY(count < image.channels(), "ChannelDropout must retain one channel");
        cv::Mat result = image.clone();
        std::vector<cv::Mat> channels;
        cv::split(result, channels);
        std::vector<int> indices(image.channels());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937((uint32_t)generator.next()));
        for (int i = 0; i < count; i++)
            channels[indices[i]].setTo(fill);
        cv::merge(channels, result);
        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("channel_dropout", channel_dropout, (forward_shape<0>));

    cv::Mat channel_shuffle(const cv::Mat &image, int seed) noexcept(false)
    {
        if (image.channels() == 1)
            return image.clone();
        cv::RNG generator(seed);
        std::vector<int> indices(image.channels());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937((uint32_t)generator.next()));
        std::vector<cv::Mat> input;
        cv::split(image, input);
        std::vector<cv::Mat> output;
        output.reserve(input.size());
        for (int index : indices)
            output.push_back(input[index]);
        cv::Mat result;
        cv::merge(output, result);
        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("channel_shuffle", channel_shuffle, (forward_shape<0>));

    cv::Mat additive_noise(const cv::Mat &image, const std::string &noise_type,
                           const Sequence<float> &noise_range, bool per_channel, int seed) noexcept(false)
    {
        VERIFY(noise_range.size() == 2, "noise_range must contain two values");
        cv::RNG generator(seed);
        cv::Mat result = to_unit(image);
        result += random_noise(generator, image.size(), image.channels(), noise_range[0], noise_range[1], noise_type, per_channel);
        clip_unit(result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("additive_noise", additive_noise, (forward_shape<0>));

    cv::Mat multiplicative_noise(const cv::Mat &image, const Sequence<float> &multiplier_range,
                                 bool per_channel, bool elementwise, int seed) noexcept(false)
    {
        VERIFY(multiplier_range.size() == 2, "multiplier must contain two values");
        cv::RNG generator(seed);
        const int channels = per_channel ? image.channels() : 1;
        const int rows = elementwise ? image.rows : 1;
        const int cols = elementwise ? image.cols : 1;
        cv::Mat multiplier(rows, cols, CV_MAKETYPE(CV_32F, channels));
        generator.fill(multiplier, cv::RNG::UNIFORM, multiplier_range[0], multiplier_range[1]);
        if (!elementwise)
            multiplier = cv::repeat(multiplier, image.rows, image.cols);
        if (!per_channel && image.channels() > 1)
        {
            cv::Mat shared = multiplier;
            std::vector<cv::Mat> shared_channels(image.channels(), shared);
            cv::merge(shared_channels, multiplier);
        }
        cv::Mat result = to_unit(image);
        result = result.mul(multiplier);
        clip_unit(result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("multiplicative_noise", multiplicative_noise, (forward_shape<0>));

    cv::Mat shot_noise(const cv::Mat &image, const Sequence<float> &scale_range, int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        const float scale = sample_range(generator, scale_range, "scale_range");
        VERIFY(scale > 0, "scale_range must be positive");
        cv::Mat linear;
        cv::pow(to_unit(image), 2.2, linear);
        linear *= 1.0 / scale;
        cv::Mat noisy;
        noisy = linear.clone();
        std::mt19937 engine((uint32_t)generator.next());
        for (int y = 0; y < noisy.rows; y++)
        {
            float *row = noisy.ptr<float>(y);
            for (int x = 0; x < noisy.cols * noisy.channels(); x++)
            {
                std::poisson_distribution<int> poisson(std::max(0.0f, row[x]));
                row[x] = static_cast<float>(poisson(engine));
            }
        }
        noisy *= scale;
        cv::pow(noisy, 1.0 / 2.2, noisy);
        clip_unit(noisy);
        return from_unit(noisy, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("shot_noise", shot_noise, (forward_shape<0>));

    cv::Mat salt_and_pepper(const cv::Mat &image, const Sequence<float> &amount_range,
                            const Sequence<float> &salt_range, int seed) noexcept(false)
    {
        cv::RNG generator(seed);
        const float amount = sample_range(generator, amount_range, "amount_range");
        const float salt_ratio = sample_range(generator, salt_range, "salt_vs_pepper_range");
        VERIFY(amount >= 0 && amount <= 1, "amount_range must be between 0 and 1");
        VERIFY(salt_ratio >= 0 && salt_ratio <= 1, "salt_vs_pepper_range must be between 0 and 1");
        cv::Mat result = to_unit(image);
        const int pixels = image.rows * image.cols;
        std::vector<int> indices(pixels);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937((uint32_t)generator.next()));
        const int noisy_pixels = static_cast<int>(amount * pixels);
        const int salt_pixels = static_cast<int>(salt_ratio * noisy_pixels);
        std::vector<cv::Mat> channels;
        cv::split(result, channels);
        for (int i = 0; i < noisy_pixels; i++)
        {
            const int y = indices[i] / image.cols;
            const int x = indices[i] % image.cols;
            const float value = i < salt_pixels ? 1.0f : 0.0f;
            for (cv::Mat &channel : channels)
                channel.at<float>(y, x) = value;
        }
        cv::merge(channels, result);
        return from_unit(result, image.depth(), maximum_value(image));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("salt_and_pepper", salt_and_pepper, (forward_shape<0>));

    /**
     * @brief Compute X partial derivative of an image.
     *
     */

    cv::Mat derivative_x(const cv::Mat &image) 
    {
        cv::Mat out;
        cv::Sobel(image, out, CV_32F, 1, 0);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("derivative_x", derivative_x, (forward_shape_new_type<0, float>) );

    /**
     * @brief Compute Y partial derivative of an image.
     *
     */

    cv::Mat derivative_y(const cv::Mat &image)
    {
        cv::Mat out;
        cv::Sobel(image, out, CV_32F, 0, 1);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("derivative_y", derivative_y, (forward_shape_new_type<0, float>) );

    /**
     * @brief Compute Canny edge detection.
     *
     */

    cv::Mat edges(const cv::Mat &image, float threshold1, float threshold2)
    {
        cv::Mat out;
        cv::Canny(image, out, threshold1, threshold2);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("edges", edges, ensure_single_channel);

    /**
     * @brief Compute Laplacian of an image.
     *
     */

    cv::Mat laplacian(const cv::Mat &image)
    {
        cv::Mat out;
        cv::Laplacian(image, out, CV_32F);
        return out;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("laplacian", laplacian, (forward_shape_new_type<0, float>) );

   
}
