
#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>

namespace pixelpipes {

SharedVariable PointsCenter(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], Point2DType), "Not a point list");

    SharedList list = List::cast(inputs[0]);
    cv::Point2f accumulator;
    for (size_t i = 0; i < list->size(); i++) {
        accumulator += Point2D::get_value(list->get(i));
    }
    accumulator /= (float) list->size();

    return std::make_shared<Point2D>(accumulator);

}

REGISTER_OPERATION_FUNCTION("points_center", PointsCenter);

SharedVariable PointsFromRectangle(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], FloatType) && List::length(inputs[0]) == 4, "Not a float list of four elements");

    SharedList list = List::cast(inputs[0]);
    float left = Float::get_value(list->get(0));
    float top = Float::get_value(list->get(1));
    float right = Float::get_value(list->get(2));
    float bottom = Float::get_value(list->get(3));

    return std::make_shared<Point2DList>(std::vector<cv::Point2f>({cv::Point2f(left, top), cv::Point2f(right, top), cv::Point2f(right, bottom), cv::Point2f(left, bottom)}));

}

REGISTER_OPERATION_FUNCTION("points_from_rectangle", PointsFromRectangle);

SharedVariable PointFromInputs(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters, only two required");

    return MAKE_POINT(Float::get_value(inputs[0]), Float::get_value(inputs[1]));
}

REGISTER_OPERATION_FUNCTION("make_point", PointFromInputs);

SharedVariable PointsFromInputs(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() % 2 == 0, "Incorrect number of parameters, number should be even");

    std::vector<cv::Point2f> result;

    for (size_t i = 0; i < inputs.size(); i+=2) {
        result.push_back(cv::Point2f(Float::get_value(inputs[i]), Float::get_value(inputs[i+1])));
    }

    return std::make_shared<Point2DList>(result);
}

REGISTER_OPERATION_FUNCTION("make_points", PointsFromInputs);

SharedVariable PointsFromList(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], FloatType) && List::length(inputs[0]) % 2 == 0, "Not a float list with even element count");

    std::vector<cv::Point2f> result;

    SharedList list = List::cast(inputs[0]);
    for (size_t i = 0; i < list->size(); i+=2) {
        result.push_back(cv::Point2f(Float::get_value(list->get(i)), Float::get_value(list->get(i+1))));
    }

    return std::make_shared<Point2DList>(result);

}

REGISTER_OPERATION_FUNCTION("list_to_points", PointsFromList);


SharedVariable RandomPoints(std::vector<SharedVariable> inputs, ContextHandle context, int count) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    std::vector<cv::Point2f> data(count);

    RandomGenerator generator = StohasticOperation::create_generator(context);

    std::uniform_real_distribution<float> distribution(-1, 1);

    for (int i = 0; i < count; i++) {
        data.push_back(cv::Point2f(distribution(generator), distribution(generator)));
    }

    return std::make_shared<Point2DList>(data);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("random", RandomPoints, StohasticOperation, int);

inline void execute_operation(ArithmeticOperation op, std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& result) {

        result.resize(points0.size());

        switch (op) {
            case ArithmeticOperation::ADD: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i] = points0[i] + points1[i];
                }
                break;
            }
            case ArithmeticOperation::SUBTRACT: {
                for (size_t i = 0; i < points0.size(); i++) {
                    result[i] = points0[i] - points1[i];
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
                throw VariableException("Unsupported operation");
            }
        }

}

SharedVariable PointArithmeticOperation(std::vector<SharedVariable> inputs, ContextHandle context, ArithmeticOperation op) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(Point2D::is(inputs[0]) || Point2D::is(inputs[1]), "At least one input should be a point");

    cv::Point2f result;

    cv::Point2f point0 = Point2D::get_value(inputs[0]);
    cv::Point2f point1 = Point2D::get_value(inputs[1]);

    switch (op) {
        case ArithmeticOperation::ADD: {
            result = point0 + point1;
            break;
        }
        case ArithmeticOperation::SUBTRACT: {
            result = point0 - point1;
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
            throw VariableException("Unsupported operation");
        }
    }

    return std::make_shared<Point2D>(result);
}

REGISTER_OPERATION_FUNCTION("point_arithmetic", PointArithmeticOperation, ArithmeticOperation);

SharedVariable PointsArithmeticOperation(std::vector<SharedVariable> inputs, ContextHandle context, ArithmeticOperation op) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(Point2DList::is(inputs[0]) || Point2DList::is(inputs[1]), "At least one input should be list of points");

    std::vector<cv::Point2f> result;

    // Both inputs are lists
    if (List::is(inputs[0]) && Point2DList::is(inputs[1])) {

        SharedList list0 = List::cast(inputs[0]);
        SharedList list1 = List::cast(inputs[1]);

        VERIFY(list0->size() == list1->size(), "List sizes do not match");

        auto points0 = list0->elements<cv::Point2f>();
        auto points1 = list1->elements<cv::Point2f>();

        execute_operation(op, points0, points1, result);

    } else {

        if (Point2DList::is(inputs[0])) {

            SharedList list0 = List::cast(inputs[0]);
            result.resize(list0->size());
            std::vector<cv::Point2f> points0 = list0->elements<cv::Point2f>();

            cv::Point2f value = Point2D::is(inputs[1]) ? Point2D::get_value(inputs[1]) : cv::Point2f(Float::get_value(inputs[1]), Float::get_value(inputs[1]));
            std::vector<cv::Point2f> points1(points0.size(), value);

            execute_operation(op, points0, points1, result);

        } else {

            SharedList list1 = List::cast(inputs[1]);
            result.resize(list1->size());
            std::vector<cv::Point2f> points1 = list1->elements<cv::Point2f>();

            cv::Point2f value = Point2D::is(inputs[0]) ? Point2D::get_value(inputs[0]) : cv::Point2f(Float::get_value(inputs[0]), Float::get_value(inputs[0]));
            std::vector<cv::Point2f> points0(points1.size(), value);

            execute_operation(op, points0, points1, result);

        }
    }

    return std::make_shared<Point2DList>(result);
}

REGISTER_OPERATION_FUNCTION("points_arithmetic", PointsArithmeticOperation, ArithmeticOperation);


}