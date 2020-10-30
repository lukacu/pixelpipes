#include <cmath>
#include <limits>

#include "numbers.hpp"
#include "geometry.hpp"
#include "list.hpp"

namespace pixelpipes {

SharedVariable TranslateView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    float x = Float::get_value(inputs[0]);
    float y = Float::get_value(inputs[1]);

    cv::Matx33f m(1, 0, x,
          0, 1, y,
          0, 0, 1);

    return std::make_shared<View>(m);

}

SharedVariable RotateView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    float r = Float::get_value(inputs[0]);

    cv::Matx33f m(std::cos(r), -std::sin(r), 0,
          std::sin(r), std::cos(r), 0,
          0, 0, 1);

    return std::make_shared<View>(m);

}

SharedVariable ScaleView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    float x = Float::get_value(inputs[0]);
    float y = Float::get_value(inputs[1]);

    cv::Matx33f m(x, 0, 0,
          0, y, 0,
          0, 0, 1);

    return std::make_shared<View>(m);

}

SharedVariable Chain::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() < 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    cv::Matx33f m(1, 0, 0,
          0, 1, 0,
          0, 0, 1);

    for (auto input : inputs) {
        cv::Matx33f view = View::get_value(input);

        m = m * view;

    }

    return std::make_shared<View>(m);

}

SharedVariable IdentityView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 0) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    cv::Matx33f m(1, 0, 0,
          0, 1, 0,
          0, 0, 1);

    return std::make_shared<View>(m);

}

SharedVariable ViewPoints::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    std::vector<cv::Point2f> points = Points::get_value(inputs[0]);
    cv::Matx33f transform = View::get_value(inputs[1]);

    std::vector<cv::Point2f> points2;

    try {

        cv::perspectiveTransform(points, points2, transform);

    } catch (cv::Exception cve) {
        throw (OperationException(cve.what(), shared_from_this()));
    }


    return std::make_shared<Points>(points2);

}

SharedVariable BoundingBox::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    std::vector<cv::Point2f> points = Points::get_value(inputs[0]);

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

    return std::make_shared<FloatList>(bb);

}


SharedVariable CenterView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    if (!List::is_list(inputs[0], VariableType::Float)) {
        throw OperationException("Not a float list", shared_from_this());
    }

    auto bbox = std::static_pointer_cast<List>(inputs[0]);

    float left = Float::get_value(bbox->get(0));
    float top = Float::get_value(bbox->get(1));
    float right = Float::get_value(bbox->get(2));
    float bottom = Float::get_value(bbox->get(3));

    cv::Matx33f m(1, 0, -(left + right) / 2,
          0, 1, -(top + bottom) / 2,
          0, 0, 1);

    return std::make_shared<View>(m);


}

SharedVariable FocusView::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    if (!List::is_list(inputs[0], VariableType::Float)) {
        throw OperationException("Not a float list", shared_from_this());
    }

    float scale = Float::get_value(inputs[1]);

    auto bbox = std::static_pointer_cast<List>(inputs[0]);

    float left = Float::get_value(bbox->get(0));
    float top = Float::get_value(bbox->get(1));
    float right = Float::get_value(bbox->get(2));
    float bottom = Float::get_value(bbox->get(3));

    scale = sqrt(scale / ( (bottom - top) * (right - left) ));

    cv::Matx33f m(scale, 0, 0,
          0, scale, 0,
          0, 0, 1);

    return std::make_shared<View>(m);


}

}

