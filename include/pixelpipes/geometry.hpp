#pragma once

#include <pixelpipes/token.hpp>

namespace pixelpipes {

typedef struct Point2D {

    float x; float y;

} Point2D;

inline std::ostream& operator<<(std::ostream& os, const Point2D& p) {
    os << "Point2D x=" << p.x << ", y=" << p.y;
    return os;
}

typedef struct point3D {

    float x; float y; float z;

} Point3D;

inline std::ostream& operator<<(std::ostream& os, const Point3D& p) {
    os << "Point3D x=" << p.x << ", y=" << p.y << ", z=" << p.z;
    return os;
}

typedef struct View2D {

    float m00; float m01; float m02;
    float m10; float m11; float m12;
    float m20; float m21; float m22;

} View2D;

inline std::ostream& operator<<(std::ostream& os, const View2D& p) {
    os << "View2D";
    return os;
}

typedef struct View3D {

    float m00; float m01; float m02; float m03;
    float m10; float m11; float m12; float m13;
    float m20; float m21; float m22; float m23;
    float m30; float m31; float m32; float m33;

} View3D;

inline std::ostream& operator<<(std::ostream& os, const View3D& p) {
    os << "View3D";
    return os;
}

#define Point2DType GetTypeIdentifier<Point2D>()
#define Point3DType GetTypeIdentifier<Point3D>()

#define View2DType GetTypeIdentifier<View2D>()
#define View3DType GetTypeIdentifier<View3D>()

typedef ContainerToken<Point2D> Point2DVariable;
typedef ContainerToken<Point3D> Point3DVariable;

typedef ContainerToken<View2D> View2DVariable;
typedef ContainerToken<View3D> View3DVariable;

typedef ContainerList<Point2D> Point2DList;
typedef ContainerList<Point3D> Point3DList;

#define Point2DListType GetListIdentifier<std::vector<Point2D>>()
#define Point3DListType GetListIdentifier<std::vector<Point3D>>()

template<>
inline Point2D extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type_id() == FloatType) {
        float value = Float::get_value(v);
        return Point2D{value, value};
    }

    if (v->type_id() == IntegerType) {
        int value = Integer::get_value(v);
        return Point2D{(float)value, (float)value};
    }

    if (v->type_id() != Point2DType)
        throw TypeException("Not a point value");

    return std::static_pointer_cast<Point2DVariable>(v)->get();
}

template<>
inline SharedToken wrap(const Point2D v) {
    return std::make_shared<Point2DVariable>(v);
}


template<>
inline Point3D extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type_id() == FloatType) {
        float value = Float::get_value(v);
        return Point3D{value, value, value};
    }

    if (v->type_id() == IntegerType) {
        int value = Integer::get_value(v);
        return Point3D{(float)value, (float)value, (float)value};
    }

    if (v->type_id() != Point3DType)
        throw TypeException("Not a point value");

    return std::static_pointer_cast<Point3DVariable>(v)->get();
}

template<>
inline SharedToken wrap(const Point3D v) {
    return std::make_shared<Point3DVariable>(v);
}

template<>
inline View2D extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type_id() != View2DType)
        throw TypeException("Not a point value");

    return std::static_pointer_cast<View2DVariable>(v)->get();
}

template<>
inline SharedToken wrap(const View2D v) {
    return std::make_shared<View2DVariable>(v);
}


template<>
inline View3D extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type_id() != View3DType)
        throw TypeException("Not a point value");

    return std::static_pointer_cast<View3DVariable>(v)->get();
}

template<>
inline SharedToken wrap(const View3D v) {
    return std::make_shared<View3DVariable>(v);
}

template<>
inline std::vector<Point2D> extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (Point2DVariable::is(v)) {
        auto p = extract<Point2D>(v);
        return std::vector<Point2D>{p};
    } else {
        return Point2DList::cast(v)->elements<Point2D>();
    }
}

template<>
inline SharedToken wrap(const std::vector<Point2D> v) {
    return std::make_shared<Point2DList>(v);
}


#define MAKE_POINT(X, Y) std::make_shared<Point2DVariable>(Point2D{X, Y})

#define IS_NUMERIC_LIST(V) ((List::is_list(V, FloatType) || List::is_list(V, IntegerType)))
#define IS_RECTANGLE(V) ((List::is_list(V, FloatType) || List::is_list(V, IntegerType)) && List::length(V) == 4)


}