
#include <iostream>

#include "types.hpp"
#include "python.hpp"

namespace pixelpipes {

class ConstantList: public List {
public:
    ConstantList(SharedVariable value, int length) : value(value), length(length) {}

    ~ConstantList() = default;

    virtual size_t size() const { return length; }

    virtual VariableType element_type() const { return value->type(); }

    virtual SharedVariable get(int index) const {
        return value;
    }

private:

    SharedVariable value;

    int length;

};

class ListSource: public Operation {
public:

    ListSource(std::shared_ptr<List> bundle) : bundle(bundle)  {}

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) { return bundle; }

private:

    std::shared_ptr<List> bundle;

};

operation_initializer _list_source_initialization([](py::module &module) { 
    
    py::class_<ListSource, Operation, std::shared_ptr<ListSource> >(module, "ListSource")
        .def(py::init<std::shared_ptr<List> >());
 });

/**
 * @brief Returns a sublist of a given list for a specified first and last element.
 * 
 */
SharedVariable SublistSelect(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    int begin = Integer::get_value(inputs[1]);
    int end = Integer::get_value(inputs[2]);

    return std::make_shared<Sublist>(list, begin, end);

}

REGISTER_OPERATION_FUNCTION(SublistSelect);

/**
 * @brief Filters a list with another list used as a mask.
 * 
 */
SharedVariable FilterSelect(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    if (List::is_list(inputs[0]) && List::is_list(inputs[1], VariableType::Integer))
        throw VariableException("Not a list");

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    SharedList filter = std::static_pointer_cast<List>(inputs[1]);

    if (list->size() != filter->size())
        throw VariableException("Filter length mismatch");


    std::vector<int> map;

    for (int i = 0; i < filter->size(); i++) {
        if (Integer::get_value(filter->get(i)) != 0)
            map.push_back(i);
    } 

    return std::make_shared<MappedList>(list, map);

}

REGISTER_OPERATION_FUNCTION(FilterSelect);

/**
 * @brief Returns an element from a given list.
 * 
 */
SharedVariable ListElement(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    SharedList list = std::static_pointer_cast<List>(inputs[0]);
    int index = Integer::get_value(inputs[1]);

    return list->get(index);

}

REGISTER_OPERATION_FUNCTION(ListElement);


/**
 * @brief Returns a virtual list with the given variable replicated a given number of times.
 * 
 */
SharedVariable RepeatElement(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    int length = Integer::get_value(inputs[1]);

    VERIFY(length >= 1, "List length should be 1 or more");

    return std::make_shared<ConstantList>(inputs[0], length);

}

REGISTER_OPERATION_FUNCTION(RepeatElement);


/**
 * @brief Returns a scalar length of an input list.
 * 
 */
SharedVariable ListLength(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    SharedList list = std::static_pointer_cast<List>(inputs[0]);

    return std::make_shared<Integer>((int) list->size());

}

REGISTER_OPERATION_FUNCTION(ListLength);


/**
 * @brief Compares a list to a list or scalar. Returns a list of integer 0 or 1.
 * 
 */
SharedVariable ListCompare(std::vector<SharedVariable> inputs, ContextHandle context, ComparisonOperation operation) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    VERIFY(List::is_list(inputs[0], VariableType::Integer) || List::is_list(inputs[0], VariableType::Float), "Not an numeric list");

    SharedList a = std::static_pointer_cast<List>(inputs[0]);
    SharedList b;

    if (inputs[1]->type() == VariableType::Integer || inputs[1]->type() == VariableType::Float) {
        b = std::make_shared<ConstantList>(inputs[1], a->size());
    } else if (List::is_list(inputs[1], VariableType::Integer) || List::is_list(inputs[1], VariableType::Float)) {
        throw VariableException("Not an numeric list");
    } else {
        b = std::static_pointer_cast<List>(inputs[1]);
    }

    if (a->size() != b->size())
        throw VariableException("Filter length mismatch");

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

REGISTER_OPERATION_FUNCTION(ListCompare, ComparisonOperation);

SharedVariable ListLogical(std::vector<SharedVariable> inputs, ContextHandle context, LogicalOperation operation) {

    switch (operation) {
    case LogicalOperation::AND: {
        if (inputs.size() != 2) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!(List::is_list(inputs[0], VariableType::Integer) && List::is_list(inputs[1], VariableType::Integer)))
            throw VariableException("Not an integer list");

        SharedList a = std::static_pointer_cast<List>(inputs[0]);
        SharedList b = std::static_pointer_cast<List>(inputs[1]);

        if (a->size() != b->size())
            throw VariableException("Filter length mismatch");

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) && (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);
    }
    case LogicalOperation::OR: {
        if (inputs.size() != 2) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!(List::is_list(inputs[0], VariableType::Integer) && List::is_list(inputs[1], VariableType::Integer)))
            throw VariableException("Not an integer list");

        SharedList a = std::static_pointer_cast<List>(inputs[0]);
        SharedList b = std::static_pointer_cast<List>(inputs[1]);

        if (a->size() != b->size())
            throw VariableException("Filter length mismatch");

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) || (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    case LogicalOperation::NOT: {
        if (inputs.size() != 1) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!List::is_list(inputs[0], VariableType::Integer))
            throw VariableException("Not an integer list");

        SharedList a = std::static_pointer_cast<List>(inputs[0]);

        std::vector<int> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(!(Integer::get_value(a->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    }

}

REGISTER_OPERATION_FUNCTION(ListLogical, LogicalOperation);

// TODO: support for integer operations
SharedVariable ListArithmetic(std::vector<SharedVariable> inputs, ContextHandle context, ArithmeticOperation operation) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    VERIFY(List::is_list(inputs[0], VariableType::Float) && List::is_list(inputs[1], VariableType::Float), "Not a float list");

    SharedList a = std::static_pointer_cast<List>(inputs[0]);
    SharedList b = std::static_pointer_cast<List>(inputs[1]);

    if (a->size() != b->size())
        throw VariableException("Filter length mismatch");

    switch (operation) {
    case ArithmeticOperation::ADD: {

        std::vector<float> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) + Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);
    }
    case ArithmeticOperation::SUBTRACT: {
        std::vector<float> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) - Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::MULTIPY: {
        std::vector<float> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) * Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::DIVIDE: {
        std::vector<float> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) / Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::POWER: {
        std::vector<float> result;

        for (int i = 0; i < a->size(); i++) {
            result.push_back(pow(Float::get_value(a->get(i)), Float::get_value(b->get(i))));
        } 

        return std::make_shared<FloatList>(result);

    }
    }

}

REGISTER_OPERATION_FUNCTION(ListArithmetic, ArithmeticOperation);

// TODO: better detecton of integer lists vs float
SharedVariable ListBuild(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 0, "No inputs");

    if (inputs[0]->type() == VariableType::Integer) {

        std::vector<int> result;

        for (size_t i = 0; i < inputs.size(); i++) {
            result.push_back(Integer::get_value(inputs[i]));
        }

        return std::make_shared<IntegerList>(result);

    } else {

        std::vector<float> result;

        for (size_t i = 0; i < inputs.size(); i++) {
            result.push_back(Float::get_value(inputs[i]));
        }

        return std::make_shared<FloatList>(result);

    }

}

REGISTER_OPERATION_FUNCTION(ListBuild);

SharedVariable RandomNumbers(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 3, "No inputs");

    int n = Integer::get_value(inputs[0]);



}

REGISTER_OPERATION_FUNCTION(RandomNumbers);

}