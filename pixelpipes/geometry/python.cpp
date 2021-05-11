#include <pixelpipes/geometry.hpp>
#include <pixelpipes/python.hpp>

namespace pixelpipes {

PIXELPIPES_PYTHON_MODULE(pp_geometry_py);

PIXELPIPES_PYTHON_REGISTER_WRAPPER(Point2DType, [](py::object src) {

        if  (py::tuple::check_(src)) {
            py::tuple tuple(src);
            if (tuple.size() == 2) {
                py::float_ x(tuple[0]);
                py::float_ y(tuple[1]);

                return MAKE_POINT(x, y);
            }
        }

        return empty<Point2D>();

    } );

PIXELPIPES_PYTHON_REGISTER_EXTRACTOR(Point2DType, [](SharedVariable src) {

        cv::Point2f point = Point2D::get_value(src);

        py::array_t<float> result({2});

        *result.mutable_data(0) = point.x;
        *result.mutable_data(1) = point.y;

        return result;

    } );

PIXELPIPES_PYTHON_REGISTER_WRAPPER(View2DType, [](py::object src) {

        if  (py::tuple::check_(src)) {
            py::tuple tuple(src);
            if (tuple.size() == 2) {
                py::float_ x(tuple[0]);
                py::float_ y(tuple[1]);

                return MAKE_POINT(x, y);
            }
        }

        return empty<Point2D>();

    } );

PIXELPIPES_PYTHON_REGISTER_EXTRACTOR(View2DType, [](SharedVariable src) {

        cv::Matx33f mat = View2D::get_value(src);

        py::array_t<float> result({3, 3});

        for (int i = 0; i < 3; i++) {
            *result.mutable_data(i, 0) = mat(i, 0);
            *result.mutable_data(i, 1) = mat(i, 1);
            *result.mutable_data(i, 2) = mat(i, 2);
        }

        return result;

    } );

}
