
#include "common.hpp"

namespace pixelpipes {

SharedToken PointsCenter(std::vector<SharedToken> inputs) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], Point2DType), "Not a point list");

    SharedList list = List::cast(inputs[0]);
    cv::Point2f accumulator;
    for (size_t i = 0; i < list->size(); i++) {
        accumulator += extract<cv::Point2f>(list->get(i));
    }
    accumulator /= (float) list->size();

    return wrap(accumulator);

}

REGISTER_OPERATION_FUNCTION("points_center", PointsCenter);

SharedToken PointsFromRectangle(std::vector<SharedToken> inputs) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(IS_RECTANGLE(inputs[0]), "Not a float list of four elements");

    SharedList list = List::cast(inputs[0]);
    float left = Float::get_value(list->get(0));
    float top = Float::get_value(list->get(1));
    float right = Float::get_value(list->get(2));
    float bottom = Float::get_value(list->get(3));

    return std::make_shared<Point2DList>(std::vector<Point2D>({Point2D{left, top}, Point2D{right, top}, Point2D{right, bottom}, Point2D{left, bottom}}));

}

REGISTER_OPERATION_FUNCTION("points_from_rectangle", PointsFromRectangle);

SharedToken PointFromInputs(std::vector<SharedToken> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters, only two required");

    return MAKE_POINT(Float::get_value(inputs[0]), Float::get_value(inputs[1]));
}

REGISTER_OPERATION_FUNCTION("make_point", PointFromInputs);

SharedToken PointsFromInputs(std::vector<SharedToken> inputs) {

    VERIFY(inputs.size() % 2 == 0, "Incorrect number of parameters, number should be even");

    std::vector<Point2D> result;

    for (size_t i = 0; i < inputs.size(); i+=2) {
        result.push_back(Point2D{Float::get_value(inputs[i]), Float::get_value(inputs[i+1])});
    }

    return std::make_shared<Point2DList>(result);
}

REGISTER_OPERATION_FUNCTION("make_points", PointsFromInputs);

SharedToken PointsFromList(std::vector<SharedToken> inputs) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(IS_NUMERIC_LIST(inputs[0]) && List::length(inputs[0]) % 2 == 0, "Not a float list with even element count");

    std::vector<Point2D> result;

    SharedList list = List::cast(inputs[0]);
    for (size_t i = 0; i < list->size(); i+=2) {
        result.push_back(Point2D{Float::get_value(list->get(i)), Float::get_value(list->get(i+1))});
    }

    return std::make_shared<Point2DList>(result);

}

REGISTER_OPERATION_FUNCTION("list_to_points", PointsFromList);

SharedToken RandomPoints(std::vector<SharedToken> inputs, int count) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    std::vector<Point2D> data(count);

    RandomGenerator generator = StohasticOperation::create_generator(inputs[0]);

    std::uniform_real_distribution<float> distribution(-1, 1);

    for (int i = 0; i < count; i++) {
        data.push_back(Point2D{distribution(generator), distribution(generator)});
    }

    return std::make_shared<Point2DList>(data);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("random", RandomPoints, StohasticOperation, int);

inline void execute_operation(ArithmeticOperation op, std::vector<Point2D>& points0, std::vector<Point2D>& points1, std::vector<Point2D>& result) {

        result.resize(points0.size());

        switch (op) {
            case ArithmeticOperation::ADD: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i].x = points0[i].x + points1[i].x;
                    result[i].y = points0[i].y + points1[i].y;
                }
                break;
            }
            case ArithmeticOperation::SUBTRACT: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i].x = points0[i].x - points1[i].x;
                    result[i].y = points0[i].y - points1[i].y;
                }
                break;
            }
            case ArithmeticOperation::MULTIPLY: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i].x = points0[i].x * points1[i].x;
                    result[i].y = points0[i].y * points1[i].y;
                }
                break;
            }
            case ArithmeticOperation::DIVIDE: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i].x = points0[i].x / points1[i].x;
                    result[i].y = points0[i].y / points1[i].y;
                }
                break;
            }
            case ArithmeticOperation::POWER: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i].x = pow(points0[i].x, points1[i].x);
                    result[i].y = pow(points0[i].y, points1[i].y);
                }
                break;
            }
            default: {
                throw TypeException("Unsupported operation");
            }
        }

}

SharedToken PointArithmeticOperation(std::vector<SharedToken> inputs, ArithmeticOperation op) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(Point2DVariable::is(inputs[0]) || Point2DVariable::is(inputs[1]), "At least one input should be a point");

    Point2D result;

    Point2D point0 = extract<Point2D>(inputs[0]);
    Point2D point1 = extract<Point2D>(inputs[1]);

    switch (op) {
        case ArithmeticOperation::ADD: {
            result.x = point0.x + point1.x;
            result.y = point0.y + point1.y;
            break;
        }
        case ArithmeticOperation::SUBTRACT: {
            result.x = point0.x - point1.x;
            result.y = point0.y - point1.y;
            break;
        }
        case ArithmeticOperation::MULTIPLY: {
            result.x = point0.x * point1.x;
            result.y = point0.y * point1.y;
            break;
        }
        case ArithmeticOperation::DIVIDE: {
            result.x = point0.x / point1.x;
            result.y = point0.y / point1.y;
            break;
        }
        case ArithmeticOperation::POWER: {
            result.x = pow(point0.x, point1.x);
            result.y = pow(point0.y, point1.y);
            break;
        }
        default: {
            throw TypeException("Unsupported operation");
        }
    }

    return wrap(result);
}

REGISTER_OPERATION_FUNCTION("point_arithmetic", PointArithmeticOperation, ArithmeticOperation);

SharedToken PointsArithmeticOperation(std::vector<SharedToken> inputs, ArithmeticOperation op) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    VERIFY(Point2DList::is_list(inputs[0]) || Point2DList::is_list(inputs[1]), "At least one input should be a list of points");

    std::vector<Point2D> result;

    // Both inputs are lists
    if (List::is(inputs[0]) && List::is_list(inputs[1], Point2DType)) {

        SharedList list0 = List::cast(inputs[0]);
        SharedList list1 = List::cast(inputs[1]);

        VERIFY(list0->size() == list1->size(), "List sizes do not match");

        auto points0 = list0->elements<Point2D>();
        auto points1 = list1->elements<Point2D>();

        execute_operation(op, points0, points1, result);

    } else {

        if (Point2DList::is(inputs[0])) {

            SharedList list0 = List::cast(inputs[0]);
            result.resize(list0->size());
            std::vector<Point2D> points0 = list0->elements<Point2D>();

            Point2D value = extract<Point2D>(inputs[1]);
            std::vector<Point2D> points1(points0.size(), value);

            execute_operation(op, points0, points1, result);

        } else {

            SharedList list1 = List::cast(inputs[1]);
            result.resize(list1->size());
            std::vector<Point2D> points1 = list1->elements<Point2D>();

            Point2D value = extract<Point2D>(inputs[0]);
            std::vector<Point2D> points0(points1.size(), value);

            execute_operation(op, points0, points1, result);

        }
    }

    return std::make_shared<Point2DList>(result);
}

REGISTER_OPERATION_FUNCTION("points_arithmetic", PointsArithmeticOperation, ArithmeticOperation);


}