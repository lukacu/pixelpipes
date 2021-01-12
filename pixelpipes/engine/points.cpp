

#include "types.hpp"
#include "python.hpp"

using namespace pixelpipes;

SharedVariable PointsFromBoundingBox(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], VariableType::Float) && List::length(inputs[0]) == 4, "Not a float list of four elements");

    SharedList list = List::cast(inputs[0]);
    float left = Float::get_value(list->get(0));
    float top = Float::get_value(list->get(1));
    float right = Float::get_value(list->get(2));
    float bottom = Float::get_value(list->get(3));

    return std::make_shared<PointList>(std::vector<cv::Point2f>({cv::Point2f(left, top), cv::Point2f(right, top), cv::Point2f(right, bottom), cv::Point2f(left, bottom)}));

}

REGISTER_OPERATION_FUNCTION(PointsFromBoundingBox);

SharedVariable PointsFromList(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], VariableType::Float) && List::length(inputs[0]) % 2 == 0, "Not a float list with even element count");

    std::vector<cv::Point2f> result;

    SharedList list = List::cast(inputs[0]);
    for (size_t i = 0; i < list->size(); i+=2) {
        result.push_back(cv::Point2f(Float::get_value(list->get(i)), Float::get_value(list->get(i+1))));
    }

    return std::make_shared<PointList>(result);

}

REGISTER_OPERATION_FUNCTION(PointsFromList);



SharedVariable RandomPoints(std::vector<SharedVariable> inputs, ContextHandle context) {

    return std::make_shared<Integer>(0);

}

REGISTER_OPERATION_FUNCTION(RandomPoints);
