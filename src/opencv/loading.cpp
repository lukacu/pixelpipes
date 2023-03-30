
#include "common.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pixelpipes {

    cv::Mat image_read(const BufferReference& buffer) noexcept(false)
    {
        cv::Mat wrapper(1, buffer->size(), CV_8UC1, buffer->data().data());

        // cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH
        cv::Mat image = cv::imdecode(wrapper, cv::IMREAD_UNCHANGED);

        VERIFY(!image.empty(), "Image decode error");

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read", image_read);

    cv::Mat image_read_color(const BufferReference& buffer) noexcept(false)
    {
        cv::Mat wrapper(1, buffer->size(), CV_8UC1, buffer->data().data());

        cv::Mat image = cv::imdecode(wrapper, cv::IMREAD_COLOR);

        VERIFY(!image.empty(), "Image decode error");

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read_color", image_read_color);

    cv::Mat image_read_grayscale(const BufferReference& buffer) noexcept(false)
    {
        cv::Mat wrapper(1, buffer->size(), CV_8UC1, buffer->data().data());

        cv::Mat image = cv::imdecode(wrapper, cv::IMREAD_GRAYSCALE);

        VERIFY(!image.empty(), "Image decode error");

        return image;
    }

    PIXELPIPES_OPERATION_AUTO("image_read_grayscale", image_read_grayscale);


}
