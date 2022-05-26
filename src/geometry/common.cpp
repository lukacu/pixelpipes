#include <cmath>
#include <limits>

#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/module.hpp>

#include "common.hpp"

PIXELPIPES_MODULE(geometry)

namespace pixelpipes {

/**
 * @brief Calculates bounding box of a list of points and returns it as a list of four numbers: left, top, right, bottom.
 * 
 */

SharedToken BoundingBox(TokenList inputs) {

    verify(inputs.size() == 1, "Incorrect number of parameters");
    verify(List::is_list(inputs[0], Point2DIdentifier), "Not a list of points");

    std::vector<Point2D> points = extract<std::vector<Point2D>>(inputs[0]);

    float top = std::numeric_limits<float>::max();
    float bottom = std::numeric_limits<float>::lowest();
    float left = std::numeric_limits<float>::max();
    float right = std::numeric_limits<float>::lowest();

    for (auto p : points) {
        top = std::min(top, p.y);
        left = std::min(left, p.x);
        bottom = std::max(bottom, p.y);
        right = std::max(right, p.x);
    }

    std::vector<float> bb = {left, top, right, bottom};

    return std::make_shared<FloatList>(make_span(bb));

}


REGISTER_OPERATION_FUNCTION("bounding_box", BoundingBox);

/**
 * @brief Returns a bounding box of custom size, defined by four inputs.
 * 
 */
SharedToken MakeRectangle(TokenList inputs) {

    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int x1 = Integer::get_value(inputs[0]);    
    int x2 = Integer::get_value(inputs[1]);  
    int y1 = Integer::get_value(inputs[2]);    
    int y2 = Integer::get_value(inputs[3]);

    std::vector<float> bbox = {(float)x1, (float)y1, (float)x2, (float)y2};

    return std::make_shared<FloatList>(make_span(bbox));
}

REGISTER_OPERATION_FUNCTION("make_rectangle", MakeRectangle);

}

