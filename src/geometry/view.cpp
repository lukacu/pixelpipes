#include <cmath>
#include <limits>

#include "common.hpp"

namespace pixelpipes {

class TranslateView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        if (inputs.size() != 2) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        float x = Float::get_value(inputs[0]);
        float y = Float::get_value(inputs[1]);
        
        cv::Matx33f m(1, 0, x,
            0, 1, y,
            0, 0, 1);

        return wrap(m);

    }

};

REGISTER_OPERATION("translate", TranslateView);

class RotateView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        if (inputs.size() != 1) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        float r = Float::get_value(inputs[0]);

        cv::Matx33f m(std::cos(r), -std::sin(r), 0,
            std::sin(r), std::cos(r), 0,
            0, 0, 1);

        return wrap(m);

    }

};

REGISTER_OPERATION("rotate", RotateView);

class ScaleView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        if (inputs.size() != 2) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        float x = Float::get_value(inputs[0]);
        float y = Float::get_value(inputs[1]);

        cv::Matx33f m(x, 0, 0,
            0, y, 0,
            0, 0, 1);

        return wrap(m);

    }

};

REGISTER_OPERATION("scale", ScaleView);

class IdentityView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        if (inputs.size() != 0) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        cv::Matx33f m(1, 0, 0,
            0, 1, 0,
            0, 0, 1);

        return wrap(m);

    }

};

REGISTER_OPERATION("identiry", IdentityView);


class Chain: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        if (inputs.size() < 1) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        cv::Matx33f m(1, 0, 0,
            0, 1, 0,
            0, 0, 1);

        for (auto input : inputs) {
            cv::Matx33f view = extract<cv::Matx33f>(input);

            m = m * view;

        }

        return wrap(m);

    }

};

REGISTER_OPERATION("chain", Chain);

class ViewPoints: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        verify(inputs.size() == 2, "Incorrect number of parameters");
        verify(List::is_list(inputs[0], Point2DType), "Not a list of points");

        std::vector<cv::Point2f> points = List::cast(inputs[0])->elements<cv::Point2f>();
        cv::Matx33f transform = extract<cv::Matx33f>(inputs[1]);

        std::vector<cv::Point2f> points2;

        try {

            cv::perspectiveTransform(points, points2, transform);

        } catch (cv::Exception &cve) {
            throw (OperationException(cve.what(), shared_from_this()));
        }


        return wrap(points2);

    }

};

REGISTER_OPERATION("view_points", ViewPoints);

/**
 * @brief Returns a view that centers to a bounding box.
 * 
 */
class CenterView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs)  {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");
        VERIFY(IS_RECTANGLE(inputs[0]), "Not a float list with four values");

        auto bbox = std::static_pointer_cast<List>(inputs[0]);

        float left = Float::get_value(bbox->get(0));
        float top = Float::get_value(bbox->get(1));
        float right = Float::get_value(bbox->get(2));
        float bottom = Float::get_value(bbox->get(3));

        cv::Matx33f m(1, 0, -(left + right) / 2,
            0, 1, -(top + bottom) / 2,
            0, 0, 1);

        return wrap(m);


    }

};

REGISTER_OPERATION("center_view", CenterView);


/**
 * @brief Returns a view that scales space in a way that to a bounding box
 * 
 */
class FocusView: public Operation {
public:

    virtual SharedToken run(std::vector<SharedToken> inputs) {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");
        VERIFY(IS_RECTANGLE(inputs[0]), "Not a float list with four values");

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

        return wrap(m);


    }

};

REGISTER_OPERATION("focus_view", FocusView);


}

