
#include <iostream>
#include <algorithm>

#include <pixelpipes/operation.hpp>

namespace pixelpipes {


class Sublist: public List {
public:
    Sublist(SharedList list, int from, int to) : list(list), from(from), to(to) {

        if (!list) {
            throw VariableException("Empty parent list");
        }

        if (to < from || to >= (int) list->size() || from < 0) {
            throw VariableException("Illegal sublist range");
        }

    }

    ~Sublist() = default;

    virtual size_t size() const {

        return to - from + 1;

    }

    virtual TypeIdentifier element_type() const {

        return list->element_type();

    }

    virtual SharedVariable get(int index) const {

        index += from;

        if (index < 0 || index > to) {
            throw VariableException("Index out of range");
        }

        return list->get(index);

    }

private:

    SharedList list;

    int from, to;

};

class CompositeList: public List {
public:
    CompositeList(std::initializer_list<SharedList> ilists) : lists{ilists} {

        VERIFY(lists.size() > 0, "At least one list required");

        TypeIdentifier etype = lists[0]->element_type();

        for (SharedList l : lists) {
            VERIFY(l->element_type() == etype, "Inconsitent list types");
        }

    }

    CompositeList(std::vector<SharedList> lists) : lists{lists} {

        VERIFY(lists.size() > 0, "At least one list required");

        TypeIdentifier etype = lists[0]->element_type();

        for (SharedList l : lists) {
            VERIFY(l->element_type() == etype, "Inconsitent list types");
        }

    }

    ~CompositeList() = default;

    virtual size_t size() const {
        size_t total = 0;

        for (SharedList l : lists) {
            total = l->size();
        }

        return total;
    }

    virtual TypeIdentifier element_type() const {
        return lists[0]->element_type();
    }

    virtual SharedVariable get(int index) const {

        if (index < 0)
            throw VariableException("Index out of range");

        for (SharedList l : lists) {
            if (index < l->size())
                return l->get(index);
            index -= l->size() - 1;
        }

        throw VariableException("Index out of range");

    }

private:

    std::vector<SharedList> lists;

};

class MappedList: public List {
public:
    MappedList(SharedList list, std::vector<int> map) : list(list), map(map) {

        if (!list) {
            throw VariableException("Empty parent list");
        }

        for (auto index : map) {
            if (index < 0 || index >= (int) list->size())
                throw VariableException("Illegal list index");
        }

    }

    ~MappedList() = default;

    virtual size_t size() const {

        return map.size();

    }

    virtual TypeIdentifier element_type() const  {

        return list->element_type();

    }


    virtual SharedVariable get(int index) const {

        if (index < 0 || index >= (int) map.size()) {
            throw VariableException("Index out of range");
        }

        return list->get(map[index]);

    }

private:

    SharedList list;

    std::vector<int> map;

};

template<typename T>
class ConstantList: public List {
public:
    ConstantList(SharedVariable value, int length) : value(value), length(length) {

        if (!value || value->type() != Type<T>::identifier)
            throw VariableException("Wrong variable type");

    }

    ~ConstantList() = default;

    virtual size_t size() const { return length; }

    virtual TypeIdentifier element_type() const { return value->type(); }

    virtual SharedVariable get(int index) const {
        return value;
    }

private:

    SharedVariable value;

    int length;

};

/**
 * @brief Returns a sublist of a given list for a specified first and last element.
 * 
 */
SharedVariable SublistSelect(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    SharedList list = List::get_list(inputs[0]);
    int begin = Integer::get_value(inputs[1]);
    int end = Integer::get_value(inputs[2]);

    return std::make_shared<Sublist>(list, begin, end);

}

REGISTER_OPERATION_FUNCTION("list_sublist", SublistSelect);

/**
 * @brief Returns a concatenation of given input lists.
 * 
 */
SharedVariable ListConcatenate(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() > 1, "Incorrect number of parameters");

    std::vector<SharedList> lists;

    for (size_t i = 0; i < inputs.size(); i++) {
        lists.push_back(List::get_list(inputs[i]));
    }

    return std::make_shared<CompositeList>(lists);

}

REGISTER_OPERATION_FUNCTION("list_concatenate", ListConcatenate);


/**
 * @brief Filters a list with another list used as a mask.
 * 
 */
SharedVariable FilterSelect(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    if (List::is_list(inputs[0]) && List::is_list(inputs[1], IntegerType))
        throw VariableException("Not a list");

    SharedList list = List::get_list(inputs[0]);
    SharedList filter = List::get_list(inputs[1]);

    if (list->size() != filter->size())
        throw VariableException("Filter length mismatch");


    std::vector<int> map;

    for (size_t i = 0; i < filter->size(); i++) {
        if (Integer::get_value(filter->get(i)) != 0)
            map.push_back(i);
    } 

    return std::make_shared<MappedList>(list, map);

}

REGISTER_OPERATION_FUNCTION("list_filter", FilterSelect);

/**
 * @brief Maps elements from a list to another list using a list of indices.
 * 
 */
SharedVariable ListRemap(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    if (List::is_list(inputs[0]) && List::is_list(inputs[1], IntegerType))
        throw VariableException("Not a list");

    SharedList list = List::get_list(inputs[0]);
    SharedList map = List::get_list(inputs[1]);

    int length = list->size();

    std::vector<int> remap;

    for (size_t i = 0; i < map->size(); i++) {
        int k = Integer::get_value(map->get(i));
        if (k < 0 || k >= length)
            throw VariableException("Index out of bounds");
        remap.push_back(k);
    } 

    return std::make_shared<MappedList>(list, remap);

}

REGISTER_OPERATION_FUNCTION("list_remap", ListRemap);

/**
 * @brief Returns an element from a given list.
 * 
 */
SharedVariable ListElement(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    SharedList list = List::get_list(inputs[0]);
    int index = Integer::get_value(inputs[1]);

    return list->get(index);

}

REGISTER_OPERATION_FUNCTION("list_element", ListElement);


/**
 * @brief Returns a virtual list with the given variable replicated a given number of times.
 * 
 */
SharedVariable RepeatElement(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    int length = Integer::get_value(inputs[1]);

    VERIFY(length >= 1, "List length should be 1 or more");

    SharedVariable value = inputs[0];

    if (value->type() == IntegerType) {
        return std::make_shared<ConstantList<int>>(inputs[0], length);
    }

    if (value->type() == FloatType) {
        return std::make_shared<ConstantList<float>>(inputs[0], length);
    }

    if (value->type() == BooleanType) {
        return std::make_shared<ConstantList<bool>>(inputs[0], length);
    }

    if (value->type() == StringType) {
        return std::make_shared<ConstantList<std::string>>(inputs[0], length);
    }

    throw VariableException("Not a primitive type");

}

REGISTER_OPERATION_FUNCTION("list_repeat", RepeatElement);

/**
 * @brief Returns a list from start to end with linear progression over length elements.
 * 
 */
SharedVariable RangeList(std::vector<SharedVariable> inputs, bool round) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    float start = Float::get_value(inputs[0]);
    float end = Float::get_value(inputs[1]);
    size_t length = (size_t) Integer::get_value(inputs[2]);

    VERIFY(length >= 1, "List length should be 1 or more");

    if (round) {
        std::vector<int> data;
        for (size_t i = 0; i < length; i++) {
            data.push_back((int) ((i / (float) length) * (end - start) + start));
        }

        return std::make_shared<IntegerList>(data);

    } else {

        std::vector<float> data;
        for (size_t i = 0; i < length; i++) {
            data.push_back( (i / (float) length) * (end - start) + start );
        }

        return std::make_shared<FloatList>(data);
    }

}

REGISTER_OPERATION_FUNCTION("list_range", RangeList, bool);

class PermutationGenerator {
public:

    PermutationGenerator(SharedVariable seed) : generator(StohasticOperation::create_generator(seed)) { }

    int operator()(int lim) {

        std::uniform_int_distribution<int> distribution(0, lim);

        return distribution(generator);
    }

private:

    RandomGenerator generator;

};


/**
 * @brief Creates a permutation mapping.
 * 
 */
SharedVariable ListPermute(std::vector<SharedVariable> inputs) {
    
    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    SharedList list = List::get_list(inputs[0]);

    size_t length = list->size();

    std::vector<int> indices;
    for (size_t i = 0; i < length; i++) {
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end(), PermutationGenerator(inputs[1]));

    return std::make_shared<MappedList>(list, indices);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("list_permute", ListPermute, StohasticOperation);

/**
 * @brief Creates a random permutation of indices from 0 to length.
 * 
 */
SharedVariable MakePermutation(std::vector<SharedVariable> inputs) {
    
    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    size_t length = (size_t) Integer::get_value(inputs[0]);

    std::vector<int> indices;
    for (size_t i = 0; i < length; i++) {
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end(), PermutationGenerator(inputs[1]));

    return std::make_shared<IntegerList>(indices);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("list_permutation", MakePermutation, StohasticOperation);

/**
 * @brief Returns a scalar length of an input list.
 * 
 */
SharedVariable ListLength(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    SharedList list = List::get_list(inputs[0]);

    return std::make_shared<Integer>((int) list->size());

}

REGISTER_OPERATION_FUNCTION("list_length", ListLength);

/**
 * @brief Compares a list to a list or scalar. Returns a list of integer 0 or 1.
 * 
 */
SharedVariable ListCompare(std::vector<SharedVariable> inputs, ComparisonOperation operation) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    VERIFY(List::is_list(inputs[0], IntegerType) || List::is_list(inputs[0], FloatType), "Not an numeric list");

    SharedList a = List::get_list(inputs[0]);
    SharedList b;

    if (inputs[1]->type() == IntegerType || inputs[1]->type() == FloatType) {
        b = std::make_shared<ConstantList<int>>(inputs[1], a->size());
    } else if (List::is_list(inputs[1], IntegerType) || List::is_list(inputs[1], FloatType)) {
        throw VariableException("Not an numeric list");
    } else {
        b = List::get_list(inputs[1]);
    }

    if (a->size() != b->size())
        throw VariableException("Filter length mismatch");

    std::vector<int> result;
    result.reserve(a->size());

    switch (operation) {
    case ComparisonOperation::EQUAL: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) == Float::get_value(b->get(i)));
        } 
        break;
    }
    case ComparisonOperation::NOT_EQUAL: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) != Float::get_value(b->get(i)));
        } 
        break;
    }
    case ComparisonOperation::LOWER: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) < Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::LOWER_EQUAL: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) <= Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::GREATER: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) > Float::get_value(b->get(i)));
        } 
    }
    case ComparisonOperation::GREATER_EQUAL: {
        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) >= Float::get_value(b->get(i)));
        } 
    }
    }

    return std::make_shared<IntegerList>(result);

}

REGISTER_OPERATION_FUNCTION("list_compare", ListCompare, ComparisonOperation);

SharedVariable ListLogical(std::vector<SharedVariable> inputs, LogicalOperation operation) {

    switch (operation) {
    case LogicalOperation::AND: {
        if (inputs.size() != 2) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!(List::is_list(inputs[0], IntegerType) && List::is_list(inputs[1], IntegerType)))
            throw VariableException("Not an integer list");

        SharedList a = List::get_list(inputs[0]);
        SharedList b = List::get_list(inputs[1]);

        if (a->size() != b->size())
            throw VariableException("Filter length mismatch");

        std::vector<int> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) && (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);
    }
    case LogicalOperation::OR: {
        if (inputs.size() != 2) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!(List::is_list(inputs[0], IntegerType) && List::is_list(inputs[1], IntegerType)))
            throw VariableException("Not an integer list");

        SharedList a = List::get_list(inputs[0]);
        SharedList b = List::get_list(inputs[1]);

        if (a->size() != b->size())
            throw VariableException("Filter length mismatch");

        std::vector<int> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back((Integer::get_value(a->get(i)) != 0) || (Integer::get_value(b->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    case LogicalOperation::NOT: {
        if (inputs.size() != 1) {
            throw VariableException("Incorrect number of parameters");
        }

        if (!List::is_list(inputs[0], IntegerType))
            throw VariableException("Not an integer list");

        SharedList a = List::get_list(inputs[0]);

        std::vector<int> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(!(Integer::get_value(a->get(i)) != 0));
        } 

        return std::make_shared<IntegerList>(result);

    }
    }

    return std::make_shared<IntegerList>();

}

REGISTER_OPERATION_FUNCTION("logical", ListLogical, LogicalOperation);

// TODO: support for integer operations
SharedVariable ListArithmetic(std::vector<SharedVariable> inputs, ArithmeticOperation operation) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    VERIFY(List::is_list(inputs[0], FloatType) && List::is_list(inputs[1], FloatType), "Not a float list");

    SharedList a = List::get_list(inputs[0]);
    SharedList b = List::get_list(inputs[1]);

    if (a->size() != b->size())
        throw VariableException("Filter length mismatch");

    switch (operation) {
    case ArithmeticOperation::ADD: {

        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) + Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);
    }
    case ArithmeticOperation::SUBTRACT: {
        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) - Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::MULTIPLY: {
        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) * Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::DIVIDE: {
        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(Float::get_value(a->get(i)) / Float::get_value(b->get(i)));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::POWER: {
        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(pow(Float::get_value(a->get(i)), Float::get_value(b->get(i))));
        } 

        return std::make_shared<FloatList>(result);

    }
    case ArithmeticOperation::MODULO: {
        std::vector<float> result;

        for (size_t i = 0; i < a->size(); i++) {
            result.push_back(fmod(Float::get_value(a->get(i)), Float::get_value(b->get(i))));
        } 

        return std::make_shared<FloatList>(result);

    }
    default:
        throw VariableException("Unsupported operation");
    }

}

REGISTER_OPERATION_FUNCTION("list_arithmetic", ListArithmetic, ArithmeticOperation);

// TODO: better detecton of integer lists vs float
SharedVariable ListBuild(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() > 0, "No inputs");

    if (inputs[0]->type() == IntegerType) {

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

REGISTER_OPERATION_FUNCTION("list_build", ListBuild);
/*
SharedVariable RandomNumbers(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 3, "No inputs");

    int n = Integer::get_value(inputs[0]);

}

REGISTER_OPERATION_FUNCTION("random_numbers", RandomNumbers);
*/
}