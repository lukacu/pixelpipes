
#include <set>

#include "common.hpp"

#include <opencv2/imgproc.hpp>

namespace pixelpipes
{

    /**
     * @brief Calculates image moments.
     *
     */
    TensorReference moments(const TokenList& inputs) noexcept(false)
    {
        VERIFY(inputs.size() == 1, "Moments requires one input");

        auto image = extract<cv::Mat>(inputs[0]);

        VERIFY(image.channels() == 1, "Image has more than one channel");
        VERIFY(image.depth() == CV_8U || image.depth() == CV_16U, "Image has more than one channel");

        // Get unique values in the image by iteration
        
        std::set<uchar> values;
        for (int i = 0; i < image.rows; ++i)
        {
            for (int j = 0; j < image.cols; ++j)
            {
                values.insert(image.at<uchar>(i, j));
            }
        }

        size_t n = values.size();

        auto moments = create<ArrayTensor<float, 2>>(SizeSequence({n, 10}));

        auto data = moments->data().reinterpret<float>();

        size_t j = 0;
        for (auto i = values.begin(); i != values.end(); ++i)
        {
            cv::Moments m = cv::moments(image == *i, true);

            data[j++] = (float)m.m00;
            data[j++] = (float)m.m01;
            data[j++] = (float)m.m10;
            data[j++] = (float)m.m11;
            data[j++] = (float)m.m20;
            data[j++] = (float)m.m02;
            data[j++] = (float)m.m12;
            data[j++] = (float)m.m21;
            data[j++] = (float)m.m30;
            data[j++] = (float)m.m03;

        }

        return moments;
    }

    PIXELPIPES_COMPUTE_OPERATION("moments", moments, (constant_shape<float, unknown, 10>));

    /**
     * @brief Calculates connected components.
     *
     */

    cv::Mat connected_components(const cv::Mat &image) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");
        VERIFY(image.depth() == CV_8U, "Image has more than one channel");

        cv::Mat labels;
        cv::connectedComponents(image, labels, 8, CV_16U);

        return labels;
    }

    TokenReference connected_components_eval(const TokenList &inputs)
    {
        auto shape = image_shape(inputs[0]);

        VERIFY(shape.channels == 1, "Image has more than one channel");
        VERIFY(shape.type == UnsignedShortType || shape.type  == CharType, "Image is not of type unsigned char");

        return create<Placeholder>(Shape(UnsignedShortType, {shape.height, shape.width}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("connected_components", connected_components, connected_components_eval);

    /**
     * @brief Compute distance transform of an image.
     *
     */

    cv::Mat distance_transform(const cv::Mat &image)
    {
        cv::Mat out;
        cv::distanceTransform(image, out, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        return out;
    }

    TokenReference distance_transform_evaluate(const TokenList& inputs) noexcept(false)
    {
        VERIFY(inputs.size() == 1, "Distance transform requires one input");
        VERIFY(image_shape(inputs[0]).channels == 1, "Distance transform requires single channel image");

        auto shape = inputs[0]->shape();

        return create<Placeholder>(shape.cast(FloatType));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("distance_transform", distance_transform, distance_transform_evaluate);
}