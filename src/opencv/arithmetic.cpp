#include <utility>

#define OPERATION_NAMESPACE "image"

#include <opencv2/imgproc.hpp>

#include <pixelpipes/tensor.hpp>

#include "common.hpp"

namespace pixelpipes
{

    cv::Mat replicate_channels(cv::Mat &image, int channels)
    {

        cv::Mat out;
        std::vector<cv::Mat> in;
        for (int i = 0; i < channels; i++)
            in.push_back(image);
        cv::merge(in, out);

        return out;
    }

    cv::Scalar uniform_scalar(float value, int channels)
    {

        if (channels == 2)
            return cv::Scalar(value, value);
        if (channels == 3)
            return cv::Scalar(value, value, value);
        if (channels == 4)
            return cv::Scalar(value, value, value, value);

        return cv::Scalar(value);
    }

    std::pair<cv::Mat, cv::Mat> ensure_channels(cv::Mat &image1, cv::Mat &image2)
    {

        int channels1 = image1.channels();
        int channels2 = image2.channels();

        if (channels1 != channels2)
        {
            if (channels1 == 1)
            {
                image1 = replicate_channels(image1, channels2);
            }
            else if (channels2 == 1)
            {
                image2 = replicate_channels(image2, channels1);
            }
            else
            {
                throw TypeException("Channel count mismatch");
            }
        }

        return std::pair<cv::Mat, cv::Mat>(image1, image2);
    }

    template <void (*F)(cv::InputArray, cv::InputArray, cv::OutputArray, int)>
    TokenReference image_elementwise_binary(const TokenReference &a, const TokenReference &b)
    {
        cv::Mat result;

        // Both inputs are images
        if (a->is<Tensor>() && b->is<Tensor>())
        {
            cv::Mat image0 = extract<cv::Mat>(a);
            cv::Mat image1 = extract<cv::Mat>(b);

            VERIFY(image0.rows == image1.rows && image0.cols == image1.cols, "Image sizes do not match");

            auto image_pair = ensure_channels(image0, image1);
            image0 = image_pair.first;
            image1 = image_pair.second;

            F(image0, image1, result, -1);
        }
        else
        {
            if (a->is<Tensor>())
            {
                cv::Mat image = extract<cv::Mat>(a);
                float value = extract<float>(b);
                F(image, uniform_scalar((value), image.channels()), result, image.type()); // TODO: scaling based on input
            }
            else
            {
                float value = extract<float>(a);
                cv::Mat image = extract<cv::Mat>(b);
                F(uniform_scalar((value), image.channels()), image, result, image.type());
            }
        }

        return wrap(result);
    }

    void _add(cv::InputArray a, cv::InputArray b, cv::OutputArray c, int dtype)
    {
        cv::add(a, b, c, cv::noArray(), dtype);
    }

#define image_add image_elementwise_binary<_add>
    PIXELPIPES_OPERATION_AUTO("image_add", image_add);

    void _subtract(cv::InputArray a, cv::InputArray b, cv::OutputArray c, int dtype)
    {
        cv::subtract(a, b, c, cv::noArray(), dtype);
    }

#define image_subtract image_elementwise_binary<_subtract>
    PIXELPIPES_OPERATION_AUTO("image_subtract", image_subtract);

    void _multiply(cv::InputArray a, cv::InputArray b, cv::OutputArray c, int dtype)
    {
        cv::multiply(a, b, c, 1.0, dtype);
    }

#define image_multiply image_elementwise_binary<_multiply>
    PIXELPIPES_OPERATION_AUTO("image_multiply", image_multiply);

    void _divide(cv::InputArray a, cv::InputArray b, cv::OutputArray c, int dtype)
    {
        cv::divide(a, b, c, dtype);
    }

#define image_divide image_elementwise_binary<_divide>
    PIXELPIPES_OPERATION_AUTO("image_divide", image_divide);

}
