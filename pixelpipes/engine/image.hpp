
#ifndef __PP_IMAGE_H__
#define __PP_IMAGE_H__

#include <opencv2/core.hpp>

#include "engine.hpp"

namespace pixelpipes {

enum class ImageDepth {Byte, Short, Float, Double};

enum class BorderStrategy {ConstantHigh, ConstantLow, Replicate, Reflect, Wrap};


/**
 * @brief Variable container for an image. It contains an OpenCV Mat object that represents an image.
 * 
 */
class Image: public Variable {
public:
    Image(const cv::Mat value): value(value) {};

    ~Image() = default;

    cv::Mat get() { return value; };

    virtual VariableType type() { return VariableType::Image; };

    /**
     * @brief Get the internal image representation for a variable or throw a VariableException.
     * 
     * @param v Variable container
     * @return cv::Mat Internal image representation
     */
    static cv::Mat get_value(SharedVariable v) {
        if (v->type() != VariableType::Image)
            throw VariableException("Not an image value");

        return std::static_pointer_cast<Image>(v)->get();
    }

private:

    cv::Mat value;

};

/**
 * @brief Apply view linear transformation to an image.
 * 
 */
class ViewImage: public Operation {
public:

    /**
     * @brief Construct a new View Image operation
     * 
     * @param interpolate use linear interpolation or not
     * @param border what kind of border handing to use
     */
    ViewImage(bool interpolate, BorderStrategy border);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

private:

    bool interpolate;

    int border_const;
    int border_value;

};

/**
 * @brief Convert image from one depth to another, this operation also applies scaling, assuming that integer depths
 * span the full numeric range and floating point depths span from 0 to 1.
 * 
 * Accepts a single input, image to be converted.
 */
class ConvertDepth: public Operation {
public:

    ConvertDepth(ImageDepth depth);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

private:

    ImageDepth depth;
};


/**
 * @brief Convert image to grayscale single channel image. Applys cvtColor from OpenCV.
 * 
 * Accepts a single variable input.
 */
class Grayscale: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

/**
 * @brief Applies threshold to a single channel image, returns a two-value image.
 * 
 */
class Threshold: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

class Invert: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};


class Equals: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

class Channel: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

class Moments: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

/**
 * @brief Convert a list of points (3 or more) to a binary mask by rendering a polygon. Requires also width
 * and height of the output image.
 * 
 */
class Polygon: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

/**
 * @brief Generates a random noise image for a given 
 * 
 */
class Noise: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

/**
 * @brief Returns a bounding of all non-zero elements in an image. Returns a list of four numbers: left, top, right, bottom.
 * 
 */
class MaskBoundingBox: public Operation {
public:
    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);
};

}

#endif