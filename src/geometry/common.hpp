

#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>

#include <opencv2/core.hpp>
namespace pixelpipes
{

    template <>
    inline cv::Point2f extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (v->type_id() == FloatType)
        {
            float value = Float::get_value(v);
            return cv::Point2f(value, value);
        }

        if (v->type_id() == IntegerType)
        {
            int value = Integer::get_value(v);
            return cv::Point2f((float)value, (float)value);
        }

        if (v->type_id() != Point2DType)
            throw TypeException("Not a point value");

        Point2D p = std::static_pointer_cast<Point2DVariable>(v)->get();

        return cv::Point2f(p.x, p.y);
    }

    template <>
    inline SharedToken wrap(const cv::Point2f v)
    {
        return std::make_shared<Point2DVariable>(Point2D{v.x, v.y});
    }

    template <>
    inline std::vector<cv::Point2f> extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (!Point2DList::is(v))
            throw TypeException("Not a point list");

        std::vector<Point2D> original = extract<std::vector<Point2D>>(v);

        std::vector<cv::Point2f> convert; convert.reserve(original.size());
        for (auto p : original) {
            convert.push_back(cv::Point2f{p.x, p.y});
        }

        return convert;
    }

    template <>
    inline SharedToken wrap(const std::vector<cv::Point2f> v)
    {

        std::vector<Point2D> convert; convert.reserve(v.size());
        for (auto p : v) {
            convert.push_back(Point2D{p.x, p.y});
        }

        return wrap(convert);
    }

    template <>
    inline cv::Matx33f extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (v->type_id() != View2DType)
            throw TypeException("Not a view value");

        View2D d = std::static_pointer_cast<View2DVariable>(v)->get();

        cv::Matx33f m{d.m00, d.m01, d.m02, d.m10, d.m11, d.m12, d.m20, d.m21, d.m22};
        return m;
    }

    template <>
    inline SharedToken wrap(const cv::Matx33f v)
    {
        View2D d{v(0, 0), v(0, 1), v(0, 2), v(1, 0), v(1, 1), v(1, 2), v(2, 0), v(2, 1), v(2, 2)};

        return std::make_shared<View2DVariable>(d);
    }

}