
#include <pixelpipes/geometry.hpp>

namespace pixelpipes {


    PIXELPIPES_REGISTER_TYPE(Point2DType, "point2d", DEFAULT_TYPE_CONSTRUCTOR(Point2DType), default_type_resolve);
    PIXELPIPES_REGISTER_TYPE(Point3DType, "point3d", DEFAULT_TYPE_CONSTRUCTOR(Point3DType), default_type_resolve);
    PIXELPIPES_REGISTER_TYPE(View2DType, "view2d", DEFAULT_TYPE_CONSTRUCTOR(View2DType), default_type_resolve);
    PIXELPIPES_REGISTER_TYPE(View3DType, "view3d", DEFAULT_TYPE_CONSTRUCTOR(View3DType), default_type_resolve);

    PIXELPIPES_REGISTER_TYPE(Point2DListType, "point2d_list", DEFAULT_TYPE_CONSTRUCTOR(Point2DListType), default_type_resolve);
    PIXELPIPES_REGISTER_TYPE(Point3DListType, "point3d_list", DEFAULT_TYPE_CONSTRUCTOR(Point3DListType), default_type_resolve);


}