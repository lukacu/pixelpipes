#pragma once

#include <pixelpipes/tensor.hpp>

namespace pixelpipes
{

    typedef struct PIXELPIPES_API Point2D
    {

        float x = 0;
        float y = 0;

    } Point2D;

    inline std::ostream &operator<<(std::ostream &os, const Point2D &p)
    {
        os << "Point2D x=" << p.x << ", y=" << p.y;
        return os;
    }

    typedef struct PIXELPIPES_API Rectangle
    {

        float left = 0;
        float top = 0;
        float right = 0;
        float bottom = 0;

    } Rectangle;

    inline std::ostream &operator<<(std::ostream &os, const Rectangle &p)
    {
        os << "Rectangle l=" << p.left << ", t=" << p.top << ", r=" << p.right << ", b=" << p.bottom;
        return os;
    }

    typedef struct PIXELPIPES_API Point3D
    {

        float x = 0;
        float y = 0;
        float z = 0;

    } Point3D;

    inline std::ostream &operator<<(std::ostream &os, const Point3D &p)
    {
        os << "Point3D x=" << p.x << ", y=" << p.y << ", z=" << p.z;
        return os;
    }

    typedef struct PIXELPIPES_API View2D
    {

        float m00 = 1;
        float m01 = 0;
        float m02 = 0;
        float m10 = 0;
        float m11 = 1;
        float m12 = 0;
        float m20 = 0;
        float m21 = 0;
        float m22 = 1;

    } View2D;

    inline std::ostream &operator<<(std::ostream &os, const View2D &p)
    {
        UNUSED(p);
        os << "View2D";
        return os;
    }

    typedef struct View3D
    {

        float m00;
        float m01;
        float m02;
        float m03;
        float m10;
        float m11;
        float m12;
        float m13;
        float m20;
        float m21;
        float m22;
        float m23;
        float m30;
        float m31;
        float m32;
        float m33;

    } View3D;

    inline std::ostream &operator<<(std::ostream &os, const View3D &p)
    {
        UNUSED(p);
        os << "View3D";
        return os;
    }

    template <>
    inline Point2D extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            float value = extract<float>(v);
            return Point2D{value, value};
        }

        if (shape.dimensions() == 1 && shape[0] == 2)
        {
            Sequence<float> value = extract<Sequence<float>>(v);
            return Point2D{value[0], value[1]};
        }

        throw TypeException("Unable to convert to Point2D");
    }

    template <>
    inline TokenReference wrap(const Point2D v)
    {
        return create<FloatVector>(make_view(std::vector<float>{v.x, v.y}));
    }

    template <>
    inline Rectangle extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.dimensions() == 1 && shape[0] == 4)
        {
            Sequence<float> value = extract<Sequence<float>>(v);
            return Rectangle{value[0], value[1], value[2], value[3]};
        }

        throw TypeException("Unable to convert to Rectangle");
    }

    template <>
    inline TokenReference wrap(const Rectangle v)
    {
        return create<FloatVector>(make_view(std::vector<float>{v.left, v.top, v.right, v.bottom}));
    }

    template <>
    inline Point3D extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            float value = extract<float>(v);
            return Point3D{value, value, value};
        }

        if (shape.dimensions() == 1 && shape[0] == 3)
        {
            Sequence<float> value = extract<Sequence<float>>(v);
            return Point3D{value[0], value[1], value[2]};
        }

        throw TypeException("Unable to convert to Point3D");
    }

    template <>
    inline TokenReference wrap(const Point3D v)
    {
        return create<FloatVector>(make_view(std::vector<float>{v.x, v.y, v.z}));
    }

    template <>
    inline View2D extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.dimensions() == 1 && shape[0] == 9)
        {
            auto value = extract<Sequence<float>>(v);
            return View2D{value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8]};
        }

        throw TypeException("Unable to convert to View2D");
    }

    template <>
    inline TokenReference wrap(const View2D v)
    {
        return create<FloatVector>(make_view<float>((float *)&v, 9));
    }

    template <>
    inline View3D extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.dimensions() == 1 && shape[0] == 16)
        {
            auto value = extract<Sequence<float>>(v);
            return View3D{value[0], value[1], value[2], value[3],
                          value[4], value[5], value[6], value[7],
                          value[8], value[9], value[10], value[11],
                          value[12], value[13], value[14], value[15]};
        }

        throw TypeException("Unable to convert to View3D");
    }

    template <>
    inline TokenReference wrap(const View3D v)
    {
        return create<FloatVector>(make_view<float>((float *)&v, 16));
    }

    template <>
    inline Sequence<Point2D> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.dimensions() == 1 && (shape[0] % 2 == 0))
        {
            return extract<Sequence<float>>(v).convert<Point2D>();
        }

        if (shape.dimensions() == 2)
        {
            TensorReference tensor = extract<TensorReference>(v);
            Sequence<Point2D> points(tensor->length());
            auto view = points.reinterpret<uchar>();
            copy_buffer(tensor, view);
            return points;
        }

        try
        {
            Point2D p = extract<Point2D>(v);
            return Sequence<Point2D>({p});
        }
        catch (...)
        {
            throw TypeException("Unable to convert to sequence of Point2D");
        }
    }

    template <>
    inline std::vector<Point2D> extract(const TokenReference &v)
    {
        auto s = extract<Sequence<Point2D>>(v);
        return std::vector<Point2D>(s.begin(), s.end());
    }

    template <>
    inline TokenReference wrap(const std::vector<Point2D> &v)
    {
        return create<FloatMatrix>(v.size(), 2, make_view<float>((float *)v.data(), 2 * v.size()));
    }

    template <>
    inline TokenReference wrap(const Sequence<Point2D> &v)
    {
        return create<FloatMatrix>(v.size(), 2, make_view<float>((float *)v.data(), 2 * v.size()));
    }

    template <>
    inline TokenReference wrap(Sequence<Point2D> v)
    {
        return create<FloatMatrix>(v.size(), 2, make_view<float>((float*)v.data(), 2 * v.size()));
    }

    template <>
    inline Sequence<Point3D> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        Shape shape = v->shape();

        if (shape.dimensions() == 1 && (shape[0] % 3 == 0))
        {
            return extract<Sequence<float>>(v).convert<Point3D>();
        }

        if (shape.dimensions() == 3)
        {
            TensorReference tensor = extract<TensorReference>(v);
            Sequence<Point3D> points(tensor->length());
            auto view = points.reinterpret<uchar>();
            copy_buffer(tensor, view);
            return points;
        }

        try
        {
            Point3D p = extract<Point3D>(v);
            return Sequence<Point3D>({p});
        }
        catch (...)
        {
            throw TypeException("Unable to convert to sequence of Point3D");
        }
    }

    template <>
    inline std::vector<Point3D> extract(const TokenReference &v)
    {
        auto s = extract<Sequence<Point3D>>(v);
        return std::vector<Point3D>(s.begin(), s.end());
    }

    template <>
    inline TokenReference wrap(const std::vector<Point3D> &v)
    {
        return create<FloatMatrix>(v.size(), 3, make_view<float>((float *)v.data(), 3 * v.size()));
    }

    inline bool is_numeric_list(const ListReference &v)
    {
        Shape s = v->shape();
        return (s.element() == IntegerIdentifier || s.element() == FloatIdentifier) && (s.dimensions() == 1);
    }

    inline bool is_rectangle(const ListReference &v)
    {
        return is_numeric_list(v) && v->shape()[0] == 4;
    }

}