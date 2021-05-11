

#include <pixelpipes/types.hpp>
#include <opencv2/core.hpp>

namespace pixelpipes {

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