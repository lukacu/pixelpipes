
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "numbers.hpp"
#include "list.hpp"
#include "image.hpp"
#include "geometry.hpp"

namespace pixelpipes {

class ConstantList: public List {
public:
    ConstantList(int length, SharedVariable value) : value(value), length(length) {}

    ~ConstantList() = default;

    virtual size_t size() { return length; }

    virtual VariableType element_type() { return value->type(); }

    virtual SharedVariable get(int index) {
        return value;
    }

private:

    SharedVariable value;

    int length;

};

Sublist::Sublist(SharedList list, int from, int to): list(list), from(from), to(to) {

    if (!list) {
        throw VariableException("Empty parent list");
    }

    if (to < from || to >= (int) list->size() || from < 0) {
        throw VariableException("Illegal sublist range");
    }

}

size_t Sublist::size() {

    return to - from + 1;

}

VariableType Sublist::element_type() {

    return list->element_type();

}

SharedVariable Sublist::get(int index) {

    index += from;

    if (index < 0 || index > to) {
        throw VariableException("Index out of range");
    }

    return list->get(index);

}


MappedList::MappedList(SharedList list, std::vector<int> map): list(list), map(map) {

    if (!list) {
        throw VariableException("Empty parent list");
    }

    for (auto index : map) {
        if (index < 0 || index >= list->size())
            throw VariableException("Illegal list index");
    }

}

size_t MappedList::size() {

    return map.size();

}

VariableType MappedList::element_type() {

    return list->element_type();

}

SharedVariable MappedList::get(int index) {

    if (index < 0 || index >= map.size()) {
        throw VariableException("Index out of range");
    }

    return list->get(map[index]);

}


ImageFileList::ImageFileList(std::vector<std::string> list, std::string prefix, bool grayscale) : prefix(prefix), list(list), grayscale(grayscale) {

    if (list.empty())
        throw VariableException("File list is empty");

}

size_t ImageFileList::ImageFileList::size() {

    return list.size();

}

ImageList::ImageList(std::vector<cv::Mat> list) : list(list) {

}

size_t ImageList::size() {
    return list.size();
}

VariableType ImageList::element_type() {
    return VariableType::Image;
}

SharedVariable ImageList::get(int index) {
    return std::make_shared<Image>(list[index]);
}

PointsList::PointsList(std::vector<std::vector<cv::Point2f> > list) : list(list) {

}

PointsList::PointsList(std::vector<cv::Mat> input) {

    list.reserve(input.size());

    for (auto sample : input) {
        if (sample.cols != 2)
            throw VariableException("Illegal matrix shape, unable to convert to points"); 

        std::vector<cv::Point2f> points;
        for (int i = 0; i < sample.rows; i++) {
            points.push_back(cv::Point2f(sample.at<float>(i, 0), sample.at<float>(i, 1)));
        }

        list.push_back(points);

    }

}


size_t PointsList::size() {
    return list.size();
}

VariableType PointsList::element_type() {
    return VariableType::Points;
}

SharedVariable PointsList::get(int index) {
    return std::make_shared<Points>(list[index]);
}

VariableType ImageFileList::element_type() {
    return VariableType::Image;
}
 
SharedVariable ImageFileList::get(int index) {

    if (index < 0 || index >= (int)list.size()) {
        throw VariableException("Index out of range");
    }

    cv::Mat image = cv::imread(prefix + list[index], grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    
    if (image.empty()) {
        throw VariableException("Image not found: " + prefix + list[index]);
    }

    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }

    return std::make_shared<Image>(image);
 
}

IntegerList::IntegerList(std::vector<int> values) : values(values) {}

size_t IntegerList::size() {
    return values.size();
}

VariableType IntegerList::element_type() {
    return VariableType::Integer;
}

SharedVariable IntegerList::get(int index) {
    return std::make_shared<Integer>(values[index]);
}


FloatList::FloatList(std::vector<float> values) : values(values) {}

size_t FloatList::size() {
    return values.size();
}

VariableType FloatList::element_type() {
    return VariableType::Float;
}

SharedVariable FloatList::get(int index) {
    return std::make_shared<Float>(values[index]);
}


ListSource::ListSource(std::shared_ptr<List> bundle) : bundle(bundle)  {

}


SharedVariable ListSource::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    return bundle;

}

SharedVariable SublistSelect::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 3) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    int begin = Integer::get_value(inputs[1]);
    int end = Integer::get_value(inputs[2]);

    return std::make_shared<Sublist>(list, begin, end);

}

SharedVariable FilterSelect::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    if (List::is_list(inputs[0]) && List::is_list(inputs[1], VariableType::Integer))
        throw OperationException("Not a list", shared_from_this());

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    SharedList filter = std::static_pointer_cast<List>(inputs[1]);

    if (list->size() != filter->size())
        throw OperationException("Filter length mismatch", shared_from_this());


    std::vector<int> map;

    for (int i = 0; i < filter->size(); i++) {
        if (Integer::get_value(filter->get(i)) != 0)
            map.push_back(i);
    } 

    return std::make_shared<MappedList>(list, map);

}

SharedVariable ListElement::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 2) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    int index = Integer::get_value(inputs[1]);

    return list->get(index);

}

SharedVariable ListLength::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw OperationException("Incorrect number of parameters", shared_from_this());
    }

    SharedList list = std::static_pointer_cast<List>(inputs[0]);

    return std::make_shared<Integer>((int) list->size());

}

ListCompare::ListCompare(ComparisonOperation operation) : operation(operation) { }

SharedVariable ListCompare::run(std::vector<SharedVariable> inputs, ContextHandle context) {

        if (inputs.size() != 2) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        if (List::is_list(inputs[0], VariableType::Integer) || List::is_list(inputs[0], VariableType::Float))
            throw OperationException("Not an numeric list", shared_from_this());

        SharedList a = std::static_pointer_cast<List>(inputs[0]);
        SharedList b;

        if (inputs[1]->type() == VariableType::Integer || inputs[1]->type() == VariableType::Float) {
            b = std::make_shared<ConstantList>(a->size(), inputs[1]);
        } else if (List::is_list(inputs[1], VariableType::Integer) || List::is_list(inputs[1], VariableType::Float)) {
            throw OperationException("Not an numeric list", shared_from_this());
        } else {
            b = std::static_pointer_cast<List>(inputs[1]);
        }

        if (a->size() != b->size())
            throw OperationException("Filter length mismatch", shared_from_this());

    std::vector<int> result;
    result.reserve(a->size());

    switch (operation) {
    case ComparisonOperation::EQUAL: {
        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) == Float::get_value(b->get(i)));
        } 
        break;
    }
    case ComparisonOperation::LOWER: {
        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) < Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::LOWER_EQUAL: {
        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) <= Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::GREATER: {
        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) > Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::GREATER_EQUAL: {
        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) >= Float::get_value(b->get(i)));
        } 
    }
    }

    return std::make_shared<IntegerList>(result);

}

ListLogical::ListLogical(LogicalOperation operation) : operation(operation) { }

SharedVariable ListLogical::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    switch (operation) {
    case LogicalOperation::AND: {
        if (inputs.size() != 2) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        if (List::is_list(inputs[0], VariableType::Integer) && List::is_list(inputs[1], VariableType::Integer))
            throw OperationException("Not an integer list", shared_from_this());

        SharedList a = std::static_pointer_cast<List>(inputs[0]);
        SharedList b = std::static_pointer_cast<List>(inputs[1]);

        if (a->size() != b->size())
            throw OperationException("Filter length mismatch", shared_from_this());

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) && (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);
    }
    case LogicalOperation::OR: {
        if (inputs.size() != 2) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        if (List::is_list(inputs[0], VariableType::Integer) && List::is_list(inputs[1], VariableType::Integer))
            throw OperationException("Not an integer list", shared_from_this());

        SharedList a = std::static_pointer_cast<List>(inputs[0]);
        SharedList b = std::static_pointer_cast<List>(inputs[1]);

        if (a->size() != b->size())
            throw OperationException("Filter length mismatch", shared_from_this());

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) || (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    case LogicalOperation::NOT: {
        if (inputs.size() != 1) {
            throw OperationException("Incorrect number of parameters", shared_from_this());
        }

        if (List::is_list(inputs[0], VariableType::Integer))
            throw OperationException("Not an integer list", shared_from_this());

        SharedList a = std::static_pointer_cast<List>(inputs[0]);

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(!(Integer::get_value(a->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    }


}


}