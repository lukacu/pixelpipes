

#include <pixelpipes/types.hpp>
#include <pixelpipes/image.hpp>

#include <opencv2/core.hpp>

namespace pixelpipes {

template<>
inline cv::Mat extract(const SharedVariable v) {
    if (!Image::is(v))
        throw VariableException("Not an image value");

    return std::static_pointer_cast<Image>(v)->get();
}

template<>
inline SharedVariable wrap(const cv::Mat v) {
    return std::make_shared<Image>(v);
}

inline int maximum_value(cv::Mat image) {

    switch (image.depth()) {
    case CV_8U:
        return 255;
    case CV_16S:
        return 255 * 255;
    case CV_32F:
    case CV_64F:
        return 1;
    default:
        throw VariableException("Unsupported image depth");
    }
}

}