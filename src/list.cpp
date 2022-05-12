
#include <iostream>
#include <algorithm>

#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    class Proxy : public List
    {
    public:
        Proxy(SharedList list) : list(list)
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }
        }

        virtual ~Proxy() = default;

        virtual size_t size() const
        {

            return list->size();
        }

        virtual TypeIdentifier type_id() const
        {

            return list->type_id();
        }

        virtual TypeIdentifier element_type_id() const
        {

            return list->element_type_id();
        }

        virtual SharedToken get(size_t index) const
        {

            return list->get(index);
        }

    protected:
        SharedList list;
    };

    class Sublist : public Proxy
    {
    public:
        Sublist(SharedList list, size_t from, size_t to) : Proxy(list), from(from), to(to)
        {

            if (to < from || to >= list->size())
            {
                throw TypeException("Illegal sublist range");
            }
        }

        ~Sublist() = default;

        virtual size_t size() const
        {

            return to - from + 1;
        }

        virtual SharedToken get(size_t index) const
        {

            index += from;

            if (index > to)
            {
                throw TypeException("Index out of range");
            }

            return list->get(index);
        }

    private:
        size_t from, to;
    };

    class Table : public Proxy
    {
    public:
        Table(SharedList list, size_t row) : Proxy(list), row(row)
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }

            VERIFY(list->size() % row == 0, "");
        }

        ~Table() = default;

        virtual size_t size() const
        {

            return list->size() / row;
        }

        virtual TypeIdentifier element_type_id() const
        {

            return list->element_type_id() & ListType;
        }

        virtual SharedToken get(size_t index) const
        {

            if (index >= size())
            {
                throw TypeException("Index out of range");
            }

            return std::make_shared<Sublist>(list, index * row, (index + 1) * row - 1);
        }

    private:
        size_t row;
    };

    class CompositeList : public List
    {
    public:
        CompositeList(std::initializer_list<SharedList> ilists) : lists{ilists}
        {

            VERIFY(lists.size() > 0, "At least one list required");

            TypeIdentifier etype = lists[0]->element_type_id();

            for (SharedList l : lists)
            {
                VERIFY(l->element_type_id() == etype, "Inconsitent list types");
            }
        }

        CompositeList(std::vector<SharedList> lists) : lists{lists}
        {

            VERIFY(lists.size() > 0, "At least one list required");

            TypeIdentifier etype = lists[0]->element_type_id();

            for (SharedList l : lists)
            {
                VERIFY(l->element_type_id() == etype, "Inconsitent list types");
            }
        }

        ~CompositeList() = default;

        virtual size_t size() const
        {
            size_t total = 0;

            for (SharedList l : lists)
            {
                total = l->size();
            }

            return total;
        }

        virtual TypeIdentifier element_type_id() const
        {
            return lists[0]->element_type_id();
        }

        virtual SharedToken get(size_t index) const
        {

            for (SharedList l : lists)
            {
                if (index < l->size())
                    return l->get(index);
                index -= l->size() - 1;
            }

            throw TypeException("Index out of range");
        }

    private:
        std::vector<SharedList> lists;
    };

    class MappedList : public List
    {
    public:
        MappedList(SharedList list, std::vector<int> map) : list(list), map(map)
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }

            for (auto index : map)
            {
                if (index < 0 || ((size_t)index >= list->size()))
                    throw TypeException("Illegal list index");
            }
        }

        ~MappedList() = default;

        virtual size_t size() const
        {

            return map.size();
        }

        virtual TypeIdentifier element_type_id() const
        {

            return list->element_type_id();
        }

        virtual SharedToken get(size_t index) const
        {

            if (index >= map.size())
            {
                throw TypeException("Index out of range");
            }

            return list->get(map[index]);
        }

    private:
        SharedList list;

        std::vector<int> map;
    };

    class ConstantList : public List
    {
    public:
        ConstantList(SharedToken value, size_t length) : value(value), length(length)
        {

            if (!value || List::is(value))
                throw TypeException("Wrong token type");
        }

        ~ConstantList() = default;

        virtual size_t size() const { return length; }

        virtual TypeIdentifier element_type_id() const { return value->type_id(); }

        virtual SharedToken get(size_t index) const
        {
            if (index >= length)
                throw TypeException("Index out of range");
            return value;
        }

    private:
        SharedToken value;

        size_t length;
    };

    /**
     * @brief Returns a sublist of a given list for a specified first and last element.
     *
     */
    SharedToken SublistSelect(TokenList inputs)
    {

        VERIFY(inputs.size() == 3, "Incorrect number of parameters");

        SharedList list = List::get_list(inputs[0]);
        int begin = Integer::get_value(inputs[1]);
        int end = Integer::get_value(inputs[2]);

        return std::make_shared<Sublist>(list, begin, end);
    }

    REGISTER_OPERATION_FUNCTION("list_sublist", SublistSelect);

    /**
     * @brief Returns a view of the list where every element is a row.
     *
     */
    SharedToken ListAsTable(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        SharedList list = List::get_list(inputs[0]);
        int row = Integer::get_value(inputs[1]);

        return std::make_shared<Table>(list, row);
    }

    REGISTER_OPERATION_FUNCTION("list_table", ListAsTable);

    /**
     * @brief Returns a concatenation of given input lists.
     *
     */
    SharedToken ListConcatenate(TokenList inputs)
    {

        VERIFY(inputs.size() > 1, "Incorrect number of parameters");

        std::vector<SharedList> lists;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            lists.push_back(List::get_list(inputs[i]));
        }

        return std::make_shared<CompositeList>(lists);
    }

    REGISTER_OPERATION_FUNCTION("list_concatenate", ListConcatenate);

    /**
     * @brief Filters a list with another list used as a mask.
     *
     */
    SharedToken FilterSelect(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        if (List::is_list(inputs[0]) && List::is_list(inputs[1], IntegerType))
            throw TypeException("Not a list");

        SharedList list = List::get_list(inputs[0]);
        SharedList filter = List::get_list(inputs[1]);

        if (list->size() != filter->size())
            throw TypeException("Filter length mismatch");

        std::vector<int> map;

        for (size_t i = 0; i < filter->size(); i++)
        {
            if (Integer::get_value(filter->get(i)) != 0)
                map.push_back((int)i);
        }

        return std::make_shared<MappedList>(list, map);
    }

    REGISTER_OPERATION_FUNCTION("list_filter", FilterSelect);

    /**
     * @brief Maps elements from a list to another list using a list of indices.
     *
     */
    SharedToken ListRemap(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        if (List::is_list(inputs[0]) && List::is_list(inputs[1], IntegerType))
            throw TypeException("Not a list");

        SharedList list = List::get_list(inputs[0]);
        SharedList map = List::get_list(inputs[1]);

        int length = (int)list->size();

        std::vector<int> remap;

        for (size_t i = 0; i < map->size(); i++)
        {
            int k = Integer::get_value(map->get(i));
            if (k < 0 || k >= length)
                throw TypeException("Index out of bounds");
            remap.push_back(k);
        }

        return std::make_shared<MappedList>(list, remap);
    }

    REGISTER_OPERATION_FUNCTION("list_remap", ListRemap);

    /**
     * @brief Returns an element from a given list.
     *
     */
    SharedToken ListElement(TokenList inputs)
    {

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
    SharedToken RepeatElement(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        int length = Integer::get_value(inputs[1]);

        VERIFY(length >= 1, "List length should be 1 or more");

        SharedToken value = inputs[0];

        return std::make_shared<ConstantList>(inputs[0], length);
    }

    REGISTER_OPERATION_FUNCTION("list_repeat", RepeatElement);

    /**
     * @brief Returns a list from start to end with linear progression over length elements.
     *
     */
    SharedToken RangeList(TokenList inputs, bool round)
    {

        VERIFY(inputs.size() == 3, "Incorrect number of parameters");

        float start = Float::get_value(inputs[0]);
        float end = Float::get_value(inputs[1]);
        size_t length = (size_t)Integer::get_value(inputs[2]);

        VERIFY(length >= 1, "List length should be 1 or more");

        if (round)
        {
            std::vector<int> data;
            for (size_t i = 0; i < length; i++)
            {
                data.push_back((int)((i / (float)length) * (end - start) + start));
            }

            return std::make_shared<IntegerList>(make_span(data));
        }
        else
        {

            std::vector<float> data;
            for (size_t i = 0; i < length; i++)
            {
                data.push_back((i / (float)length) * (end - start) + start);
            }

            return std::make_shared<FloatList>(make_span(data));
        }
    }

    REGISTER_OPERATION_FUNCTION("list_range", RangeList, bool);

    /**
     * @brief Creates a permutation mapping.
     *
     */
    SharedToken ListPermute(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        SharedList list = List::get_list(inputs[0]);

        size_t length = list->size();

        std::vector<int> indices;
        for (size_t i = 0; i < length; i++)
        {
            indices.push_back((int)i);
        }
        std::shuffle(indices.begin(), indices.end(), StohasticOperation::create_generator(inputs[1]));

        return std::make_shared<MappedList>(list, indices);
    }

    REGISTER_OPERATION_FUNCTION_WITH_BASE("list_permute", ListPermute, StohasticOperation);

    /**
     * @brief Creates a random permutation of indices from 0 to length.
     *
     */
    SharedToken MakePermutation(TokenList inputs)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        size_t length = (size_t)Integer::get_value(inputs[0]);

        std::vector<int> indices;
        for (size_t i = 0; i < length; i++)
        {
            indices.push_back((int)i);
        }
        std::shuffle(indices.begin(), indices.end(), StohasticOperation::create_generator(inputs[1]));

        return std::make_shared<IntegerList>(make_span(indices));
    }

    REGISTER_OPERATION_FUNCTION_WITH_BASE("list_permutation", MakePermutation, StohasticOperation);

    /**
     * @brief Returns a scalar length of an input list.
     *
     */
    SharedToken ListLength(TokenList inputs)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        SharedList list = List::get_list(inputs[0]);

        return std::make_shared<Integer>((int)list->size());
    }

    REGISTER_OPERATION_FUNCTION("list_length", ListLength);

    /**
     * @brief Compares a list to a list or scalar. Returns a list of integer 0 or 1.
     *
     */
    SharedToken ListCompare(TokenList inputs, ComparisonOperation operation)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        VERIFY(List::is_list(inputs[0], IntegerType) || List::is_list(inputs[0], FloatType), "Not an numeric list");

        SharedList a = List::get_list(inputs[0]);
        SharedList b;

        if (inputs[1]->type_id() == IntegerType || inputs[1]->type_id() == FloatType)
        {
            b = std::make_shared<ConstantList>(inputs[1], a->size());
        }
        else if (List::is_list(inputs[1], IntegerType) || List::is_list(inputs[1], FloatType))
        {
            throw TypeException("Not an numeric list");
        }
        else
        {
            b = List::get_list(inputs[1]);
        }

        if (a->size() != b->size())
            throw TypeException("Filter length mismatch");

        std::vector<int> result;
        result.reserve(a->size());

        switch (operation)
        {
        case ComparisonOperation::EQUAL:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) == Float::get_value(b->get(i)));
            }
            break;
        }
        case ComparisonOperation::NOT_EQUAL:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) != Float::get_value(b->get(i)));
            }
            break;
        }
        case ComparisonOperation::LOWER:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) < Float::get_value(b->get(i)));
            }
            break;
        }
        case ComparisonOperation::LOWER_EQUAL:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) <= Float::get_value(b->get(i)));
            }
            break;
        }
        case ComparisonOperation::GREATER:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) > Float::get_value(b->get(i)));
            }
            break;
        }
        case ComparisonOperation::GREATER_EQUAL:
        {
            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) >= Float::get_value(b->get(i)));
            }
            break;
        }
        }

        return std::make_shared<IntegerList>(make_span(result));
    }

    REGISTER_OPERATION_FUNCTION("list_compare", ListCompare, ComparisonOperation);

    SharedToken ListLogical(TokenList inputs, LogicalOperation operation)
    {

        switch (operation)
        {
        case LogicalOperation::AND:
        {
            if (inputs.size() != 2)
            {
                throw TypeException("Incorrect number of parameters");
            }

            if (!(List::is_list(inputs[0], IntegerType) && List::is_list(inputs[1], IntegerType)))
                throw TypeException("Not an integer list");

            SharedList a = List::get_list(inputs[0]);
            SharedList b = List::get_list(inputs[1]);

            if (a->size() != b->size())
                throw TypeException("Filter length mismatch");

            std::vector<int> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back((Integer::get_value(a->get(i)) != 0) && (Integer::get_value(b->get(i)) != 0));
            }

            return std::make_shared<IntegerList>(make_span(result));
        }
        case LogicalOperation::OR:
        {
            if (inputs.size() != 2)
            {
                throw TypeException("Incorrect number of parameters");
            }

            if (!(List::is_list(inputs[0], IntegerType) && List::is_list(inputs[1], IntegerType)))
                throw TypeException("Not an integer list");

            SharedList a = List::get_list(inputs[0]);
            SharedList b = List::get_list(inputs[1]);

            if (a->size() != b->size())
                throw TypeException("Filter length mismatch");

            std::vector<int> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back((Integer::get_value(a->get(i)) != 0) || (Integer::get_value(b->get(i)) != 0));
            }

            return std::make_shared<IntegerList>(make_span(result));
        }
        case LogicalOperation::NOT:
        {
            if (inputs.size() != 1)
            {
                throw TypeException("Incorrect number of parameters");
            }

            if (!List::is_list(inputs[0], IntegerType))
                throw TypeException("Not an integer list");

            SharedList a = List::get_list(inputs[0]);

            std::vector<int> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(!(Integer::get_value(a->get(i)) != 0));
            }

            return std::make_shared<IntegerList>(make_span(result));
        }
        }

        return std::make_shared<IntegerList>();
    }

    REGISTER_OPERATION_FUNCTION("logical", ListLogical, LogicalOperation);

    // TODO: support for integer operations
    SharedToken ListArithmetic(TokenList inputs, ArithmeticOperation operation)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        VERIFY(List::is_list(inputs[0], FloatType) && List::is_list(inputs[1], FloatType), "Not a float list");

        SharedList a = List::get_list(inputs[0]);
        SharedList b = List::get_list(inputs[1]);

        if (a->size() != b->size())
            throw TypeException("Filter length mismatch");

        switch (operation)
        {
        case ArithmeticOperation::ADD:
        {

            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) + Float::get_value(b->get(i)));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        case ArithmeticOperation::SUBTRACT:
        {
            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) - Float::get_value(b->get(i)));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        case ArithmeticOperation::MULTIPLY:
        {
            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) * Float::get_value(b->get(i)));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        case ArithmeticOperation::DIVIDE:
        {
            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(Float::get_value(a->get(i)) / Float::get_value(b->get(i)));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        case ArithmeticOperation::POWER:
        {
            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(pow(Float::get_value(a->get(i)), Float::get_value(b->get(i))));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        case ArithmeticOperation::MODULO:
        {
            std::vector<float> result;

            for (size_t i = 0; i < a->size(); i++)
            {
                result.push_back(fmod(Float::get_value(a->get(i)), Float::get_value(b->get(i))));
            }

            return std::make_shared<FloatList>(make_span(result));
        }
        default:
            throw TypeException("Unsupported operation");
        }
    }

    REGISTER_OPERATION_FUNCTION("list_arithmetic", ListArithmetic, ArithmeticOperation);

    // TODO: better detecton of integer lists vs float
    SharedToken ListBuild(TokenList inputs)
    {

        VERIFY(inputs.size() > 0, "No inputs");

        if (inputs[0]->type_id() == IntegerType)
        {

            std::vector<int> result;

            for (size_t i = 0; i < inputs.size(); i++)
            {
                result.push_back(Integer::get_value(inputs[i]));
            }

            return std::make_shared<IntegerList>(make_span(result));
        }
        else
        {

            std::vector<float> result;

            for (size_t i = 0; i < inputs.size(); i++)
            {
                result.push_back(Float::get_value(inputs[i]));
            }

            return std::make_shared<FloatList>(make_span(result));
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