
#ifndef __PP_GEOMETRY_H__
#define __PP_GEOMETRY_H__

#include <opencv2/core.hpp>

#include "engine.hpp"

namespace pixelpipes {

class View: public Variable {
public:
    View(const cv::Matx33f value): value(value) {};

    ~View() = default;

    cv::Matx33f get() { return value; };

    virtual VariableType type() { return VariableType::View; };

    static cv::Matx33f get_value(SharedVariable v) {
        if (v->type() != VariableType::View)
            throw VariableException("Not a view value");

        return std::static_pointer_cast<View>(v)->get();
    }

private:

    cv::Matx33f value;

};

class Points: public Variable {
public:
    Points(const std::vector<cv::Point2f> value): value(value) {};

    ~Points() = default;

    std::vector<cv::Point2f> get() { return value; };

    virtual VariableType type() { return VariableType::Points; };

    static std::vector<cv::Point2f> get_value(SharedVariable v) {
        if (v->type() != VariableType::Points)
            return std::vector<cv::Point2f>();

        return std::static_pointer_cast<Points>(v)->get();
    }

private:

    std::vector<cv::Point2f> value;

};

class TranslateView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

class RotateView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

class ScaleView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

class IdentityView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

class Chain: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

class ViewPoints: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Calculates bounding box of a list of points and returns it as a list of four numbers: left, top, right, bottom.
 * 
 */
class BoundingBox: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Returns a view that centers to a bounding box.
 * 
 */
class CenterView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Returns a view that scales space in a way that to a bounding box
 * 
 */
class FocusView: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

}

#endif