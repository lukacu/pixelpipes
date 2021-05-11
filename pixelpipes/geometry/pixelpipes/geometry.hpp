#pragma once

#include <pixelpipes/types.hpp>

#include <opencv2/core.hpp>

namespace pixelpipes {

#define MAKE_POINT(X, Y) std::make_shared<Point2D>(cv::Point2f(X, Y))

constexpr static TypeIdentifier Point2DType = GetTypeIdentifier<cv::Point2f>();
constexpr static TypeIdentifier View2DType = GetTypeIdentifier<cv::Matx33f>();

typedef ScalarVariable<cv::Point2f> Point2D;
typedef ContainerList<cv::Point2f> Point2DList;
typedef ScalarVariable<cv::Matx33f> View2D;

template<typename T>
struct Conversion <T, typename std::enable_if<std::is_same<T, cv::Point2f>::value, T >::type> {

    static T extract(const SharedVariable v) {
        VERIFY((bool) v, "Uninitialized variable");

        if (v->type() == FloatType) {
            float value = Float::get_value(v);
            return cv::Point2f(value, value);
        }

        if (v->type() == IntegerType) {
            int value = Integer::get_value(v);
            return cv::Point2f(value, value);
        }

        if (v->type() != Point2DType)
            throw VariableException("Not a point value");

        return std::static_pointer_cast<Point2D>(v)->get();
    }

    static SharedVariable wrap(const T v) {
        return std::make_shared<Point2D>(v);
    }

};

template<typename T>
struct Conversion <T, typename std::enable_if<std::is_same<T, cv::Matx33f>::value, T >::type> {

    static T extract(const SharedVariable v) {
        VERIFY((bool) v, "Uninitialized variable");

        if (v->type() != View2DType)
            throw VariableException("Not a view value");

        return std::static_pointer_cast<View2D>(v)->get();
    }

    static SharedVariable wrap(const T v) {
        return std::make_shared<View2D>(v);
    }

};

}