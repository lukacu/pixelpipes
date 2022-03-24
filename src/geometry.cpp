
#include <pixelpipes/geometry.hpp>

namespace pixelpipes {


    PIXELPIPES_REGISTER_TYPE(Point2DType, "point2d", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(Point3DType, "point3d", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(View2DType, "view2d", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(View3DType, "view3d", do_not_create, do_not_resolve);

    PIXELPIPES_REGISTER_TYPE(Point2DListType, "point2d_list", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(Point3DListType, "point3d_list", do_not_create, do_not_resolve);


}