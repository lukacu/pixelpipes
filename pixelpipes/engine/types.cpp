
#include "types.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


namespace pixelpipes {

BaseException::BaseException(std::string reason): reason(reason) {}

const char * BaseException::what() const throw () {
   	return reason.c_str();
}

VariableException::VariableException(std::string reason): BaseException(reason) {}

std::ostream& operator<<(std::ostream& os, const Variable& v) {
    v.print(os);
    return os;
}

int Integer::get_value(SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() != VariableType::Integer)
        throw VariableException("Not an integer value");

    return std::static_pointer_cast<Integer>(v)->get();
}

void Integer::print(std::ostream& os) const {
    os << value;
}

float Float::get_value(SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() == VariableType::Integer) {
        return std::static_pointer_cast<Integer>(v)->get();
    }

    if (v->type() != VariableType::Float)
        throw VariableException("Not a float value");

    return std::static_pointer_cast<Float>(v)->get();
}

void Float::print(std::ostream& os) const {
    os << value;
}

Point::Point(cv::Point2f value): value(value) {};

Point::Point(float x, float y): value(cv::Point2f(x, y)) {};

cv::Point2f Point::get() const { return value; };

void Point::print(std::ostream& os) const {
    os << value;
}

cv::Point2f Point::get_value(SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() != VariableType::Point)
        throw VariableException("Not a point value");

    return std::static_pointer_cast<Point>(v)->get();
}

View::View(const cv::Matx33f value): value(value) {};

cv::Matx33f View::get() { return value; };

void View::print(std::ostream& os) const {
    os << value;
}

cv::Matx33f View::get_value(SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() != VariableType::View)
        throw VariableException("Not a view value");

    return std::static_pointer_cast<View>(v)->get();
}

Image::Image(const cv::Mat value): value(value) {};

cv::Mat Image::get() { return value; };

cv::Mat Image::get_value(SharedVariable v) {
    VERIFY(Image::is(v), "Not an image");

    return std::static_pointer_cast<Image>(v)->get();
}

bool Image::is(SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    return (v->type() == VariableType::Image);
}

void Image::print(std::ostream& os) const {
    os << "[Image: width " << value.cols << " height " << value.rows << " channels" << value.channels() << "]";
}

void List::print(std::ostream& os) const {
    os << "[List: length " << size() << "]";
}

Sublist::Sublist(SharedList list, int from, int to): list(list), from(from), to(to) {

    if (!list) {
        throw VariableException("Empty parent list");
    }

    if (to < from || to >= (int) list->size() || from < 0) {
        throw VariableException("Illegal sublist range");
    }

}

size_t Sublist::size() const {

    return to - from + 1;

}

VariableType Sublist::element_type() const {

    return list->element_type();

}

SharedVariable Sublist::get(int index) const {

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
        if (index < 0 || index >= (int) list->size())
            throw VariableException("Illegal list index");
    }

}

size_t MappedList::size() const {

    return map.size();

}

VariableType MappedList::element_type() const {

    return list->element_type();

}

SharedVariable MappedList::get(int index) const {

    if (index < 0 || index >= (int) map.size()) {
        throw VariableException("Index out of range");
    }

    return list->get(map[index]);

}


ImageFileList::ImageFileList(std::vector<std::string> list, std::string prefix, bool grayscale) : prefix(prefix), list(list), grayscale(grayscale) {

    if (list.empty())
        throw VariableException("File list is empty");

}

size_t ImageFileList::ImageFileList::size() const {

    return list.size();

}


VariableType ImageFileList::element_type() const {
    return VariableType::Image;
}
 
SharedVariable ImageFileList::get(int index) const {

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

TableList::TableList(const cv::Mat source) : source(source) {

    VERIFY(source.channels() == 1, "Only 2D source supported");
    VERIFY(source.cols > 0 && source.rows > 0, "Table must not be empty");

}

size_t TableList::size() const {
    return source.rows;
}

VariableType TableList::element_type() const {

    if (source.depth() < 3) {
        return VariableType::Integer;
    } else {
        return VariableType::Float;
    }

}

SharedVariable TableList::get(int index) const {

    if (source.depth() < 3) {
        std::vector<int> row;
        source.row(index).copyTo(row);
        return std::make_shared<IntegerList>(row);
    } else {
        std::vector<float> row;
        source.row(index).copyTo(row);
        return std::make_shared<FloatList>(row);
    }

}


}