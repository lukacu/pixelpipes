
#ifndef __PP_LIST_H__
#define __PP_LIST_H__

#include "engine.hpp"

namespace pixelpipes {

class List: public Variable {
public:

    ~List() = default;

    virtual size_t size() = 0;

    virtual VariableType element_type() = 0;

    virtual SharedVariable get(int index) = 0; 

    virtual VariableType type() { return VariableType::List; };

    static bool is_list(SharedVariable v) {
        return (v->type() == VariableType::List);
    }

    static bool is_list(SharedVariable v, VariableType type) {
        return (v->type() == VariableType::List) && std::static_pointer_cast<List>(v)->element_type() == type;
    }

};

typedef std::shared_ptr<List> SharedList;

class Sublist: public List {
public:
    Sublist(SharedList list, int from, int to);

    ~Sublist() = default;

    virtual size_t size();

    virtual VariableType element_type();

    virtual SharedVariable get(int index); 

private:

    SharedList list;

    int from, to;

};

class MappedList: public List {
public:
    MappedList(SharedList list, std::vector<int> map);

    ~MappedList() = default;

    virtual size_t size();

    virtual VariableType element_type();

    virtual SharedVariable get(int index); 

private:

    SharedList list;

    std::vector<int> map;

};

class ImageFileList: public List {
public:

    ImageFileList(std::vector<std::string> list, std::string prefix = std::string(), bool grayscale = false);

    ~ImageFileList() = default;

    virtual size_t size();

    virtual VariableType element_type() ;

    virtual SharedVariable get(int index); 

private:

    std::string prefix;

    std::vector<std::string> list;

    bool grayscale;

};

class ImageList: public List {
public:

    ImageList(std::vector<cv::Mat> list);

    ~ImageList() = default;

    virtual size_t size();

    virtual VariableType element_type() ;

    virtual SharedVariable get(int index); 

private:

    std::vector<cv::Mat> list;

};

class PointsList: public List {
public:
 
    PointsList(std::vector<std::vector<cv::Point2f> > list);

    PointsList(std::vector<cv::Mat> list);

    ~PointsList() = default;

    virtual size_t size();

    virtual VariableType element_type();

    virtual SharedVariable get(int index); 

private:

    std::vector<std::vector<cv::Point2f> > list;

};

class IntegerList: public List {
public:

    IntegerList(std::vector<int> values);

    ~IntegerList() = default;

    virtual size_t size();

    virtual VariableType element_type();

    virtual SharedVariable get(int index); 

private:

    std::vector<int> values;

};

class FloatList: public List {
public:

    FloatList(std::vector<float> values);

    ~FloatList() = default;

    virtual size_t size();

    virtual VariableType element_type();

    virtual SharedVariable get(int index); 

private:

    std::vector<float> values;

};


class ListSource: public Operation {
public:

    ListSource(std::shared_ptr<List> bundle);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

private:

    std::shared_ptr<List> bundle;

};

/**
 * @brief Returns a sublist of a given list for a specified first and last element.
 * 
 */
class SublistSelect: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Filters a list with another list used as a mask.
 * 
 */
class FilterSelect: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Returns an element from a given list.
 * 
 */
class ListElement: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

/**
 * @brief Compares a list to a list or scalar. Returns a list of integer 0 or 1.
 * 
 */
class ListCompare: public Operation {
public:

    ListCompare(ComparisonOperation operation);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

private:

    ComparisonOperation operation;

};

/**
 * @brief Compares a list to a list or scalar. Returns a list of integer 0 or 1.
 * 
 */
class ListLogical: public Operation {
public:

    ListLogical(LogicalOperation operation);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

private:

    LogicalOperation operation;

};

/**
 * @brief Returns a scalar length of an input list.
 * 
 */
class ListLength: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false);

};

}

#endif