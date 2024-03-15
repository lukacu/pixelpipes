
#include <iostream>
#include <algorithm>

#include <pixelpipes/tensor.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    inline Shape resize(Shape shape, Size size)
    {
        std::vector<Size> _s(shape.begin(), shape.end());
        _s[0] = size;

        return Shape(shape.element(), make_span(_s));
    }

    class Proxy : public List
    {
        PIXELPIPES_RTTI(Proxy, List)
    public:
        Proxy(const ListReference &list) : list(list.reborrow())
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }
        }

        virtual ~Proxy() = default;

        virtual Shape shape() const override
        {
            return list->shape();
        }

        virtual size_t length() const override
        {
            return list->length();
        }

        virtual TokenReference get(size_t index) const override
        {
            return list->get(index);
        }

    protected:
        ListReference list;
    };

    class Sublist : public Proxy
    {
        PIXELPIPES_RTTI(Sublist, Proxy)
    public:
        Sublist(const ListReference &list, size_t from, size_t to) : Proxy(list), from(from), to(to)
        {
            if (to < from || to >= list->length())
            {
                throw TypeException("Illegal sublist range");
            }
        }

        ~Sublist() = default;

        virtual Shape shape() const override
        {
            return resize(list->shape(), length());
        }

        virtual size_t length() const override
        {
            return to - from + 1;
        }

        virtual TokenReference get(size_t index) const override
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
        PIXELPIPES_RTTI(Table, Proxy)
    public:
        Table(const ListReference &list, size_t row) : Proxy(list), _row(row)
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }

            VERIFY(list->length() % row == 0, "");

            _shape = list->shape().pop().push(row).push(length());
        }

        ~Table() = default;

        virtual size_t length() const override
        {
            return list->length() / _row;
        }

        virtual TokenReference get(size_t index) const override
        {
            if (index >= length())
            {
                throw TypeException("Index out of range");
            }
            return create<Sublist>(list, index * _row, (index + 1) * _row - 1);
        }

    private:
        size_t _row;
        Shape _shape;
    };

    class CompositeList : public List
    {
        PIXELPIPES_RTTI(CompositeList, List)
    public:
        CompositeList(const std::initializer_list<ListReference> &ilists) : lists{ilists}
        {

            VERIFY(lists.size() > 0, "At least one list required");

            Shape eshape = lists[0]->shape().pop();

            for (auto l = lists.begin(); l != lists.end(); l++)
            {
                auto s = (*l)->shape().pop();
                VERIFY(s == eshape, "Inconsitent list types");
            }
        }

        CompositeList(const Span<ListReference> &lists) : lists{lists}
        {

            VERIFY(lists.size() > 0, "At least one list required");

            Shape eshape = lists[0]->shape().pop();

            for (auto l = lists.begin(); l != lists.end(); l++)
            {
                auto s = (*l)->shape().pop();
                VERIFY(s == eshape, "Inconsitent list types");
            }
        }

        ~CompositeList() = default;

        virtual Shape shape() const override
        {
            return resize(lists[0]->shape(), length());
        }

        virtual size_t length() const override
        {
            size_t total = 0;

            for (auto l = lists.begin(); l != lists.end(); l++)
            {
                total += (*l)->length();
            }

            return total;
        }

        virtual TokenReference get(size_t index) const override
        {

            for (auto l = lists.begin(); l != lists.end(); l++)
            {
                if (index < (*l)->length())
                    return (*l)->get(index);
                index -= (*l)->length();
            }

            throw TypeException("Index out of range");
        }

    private:
        Sequence<ListReference> lists;
    };

    class MappedList : public Proxy
    {
        PIXELPIPES_RTTI(MappedList, Proxy)
    public:
        MappedList(const ListReference &list, std::vector<int> map) : Proxy(list), map(map)
        {

            if (!list)
            {
                throw TypeException("Empty parent list");
            }

            for (auto index : map)
            {
                if (index < 0 || ((size_t)index >= list->length()))
                    throw TypeException("Illegal list index");
            }
        }

        ~MappedList() = default;

        virtual Shape shape() const override
        {
            return resize(list->shape(), length());
        }

        virtual size_t length() const override
        {

            return map.size();
        }

        virtual TokenReference get(size_t index) const override
        {

            if (index >= map.size())
            {
                throw TypeException("Index out of range");
            }

            return list->get(map[index]);
        }

    private:
        std::vector<int> map;
    };

    class ConstantList : public List
    {
        PIXELPIPES_RTTI(ConstantList, List)
    public:
        ConstantList(const TokenReference &value, size_t length) : _value(value.reborrow()), _length(length)
        {
            VERIFY((bool)_value, "Undefined value");

            _shape = _value->shape().push(length);
        }

        ~ConstantList() = default;

        virtual Shape shape() const override
        {
            return _shape;
        }

        virtual size_t length() const override { return _length; }

        virtual TokenReference get(size_t index) const override
        {
            if (index >= _length)
                throw TypeException("Index out of range");
            return _value.reborrow();
        }

    private:
        TokenReference _value;
        size_t _length;
        Shape _shape;
    };

    /**
     * @brief Returns a sublist of a given list for a specified first and last element.
     *
     */
    TokenReference sublist_select(const ListReference &list, int begin, int end)
    {
        return create<Sublist>(list, begin, end);
    }

    TokenReference _sublist_select_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 3, "Incorrect number of parameters");

        auto list = inputs[0]->shape();
        auto begin = get_size(inputs[1]);
        auto end = get_size(inputs[2]);

        if (list[0] == unknown || begin == unknown || end == unknown)
            return create<Placeholder>(list.pop().push(unknown));

        return create<Placeholder>(list.pop().push(end - begin + 1));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("list_sublist", sublist_select, _sublist_select_eval);

    /**
     * @brief Returns a view of the list where every element is a row.
     *
     */
    TokenReference list_table(const ListReference &list, int row)
    {
        return create<Table>(list, row);
    }

    TokenReference _list_table_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        auto list = inputs[0]->shape();
        auto row = get_size(inputs[1]);

        if (list[0] == unknown || row == unknown)
            return create<Placeholder>(list.push(unknown));

        VERIFY(list[0] % row == 0, "List length should be a multiple of row size");

        return create<Placeholder>(list.pop().push(row).push(list[0] / row));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("list_table", list_table, _list_table_eval);

    /**
     * @brief Returns a concatenation of given input lists.
     *
     */
    TokenReference list_concatenate(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 1, "Incorrect number of parameters");

        std::vector<ListReference> lists;

        for (size_t i = 0; i < inputs.size(); i++)
        {
            lists.push_back(extract<ListReference>(inputs[i]));
        }

        return create<CompositeList>(make_span(lists));
    }

    TokenReference _list_concatenate_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() > 1, "Incorrect number of parameters");

        Size length = 0;
        Shape inner_shape = inputs[0]->shape();

        for (size_t i = 1; i < inputs.size(); i++)
        {
            inner_shape = inner_shape & inputs[i]->shape().pop();
            length = length + inputs[i]->shape()[0];
        }

        return create<Placeholder>(inner_shape.push(length));
    }

    PIXELPIPES_COMPUTE_OPERATION("list_concatenate", list_concatenate, _list_concatenate_eval);

    /**
     * @brief Filters a list with another list used as a mask.
     *
     */
    TokenReference list_filter(const ListReference &list, Sequence<bool> map)
    {
        if (list->length() != map.size())
            throw TypeException("Filter length mismatch");

        std::vector<int> remap;

        for (size_t i = 0; i < map.size(); i++)
        {
            if (map[i])
                remap.push_back((int)i);
        }

        return create<MappedList>(list, remap);
    }

    TokenReference _list_filter_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        auto list = inputs[0]->shape();
        auto mask = inputs[1]->shape();

        if (list[0] == unknown || mask[0] == unknown)
            return create<Placeholder>(list.push(unknown));

        VERIFY(list[0] == mask[0], "List length should be equal to mask length");

        return create<Placeholder>(list);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("list_filter", list_filter, _list_filter_eval);

    /**
     * @brief Maps elements from a list to another list using a list of indices.
     *
     */
    TokenReference list_remap(const ListReference &list, Sequence<int> map)
    {
        int length = (int)list->length();

        std::vector<int> remap;

        for (size_t i = 0; i < map.size(); i++)
        {
            if (map[i] < 0 || map[i] >= length)
                throw TypeException("Index out of bounds");
            remap.push_back(map[i]);
        }

        return create<MappedList>(list, remap);
    }

    TokenReference _list_remap_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        auto list = inputs[0]->shape();
        auto map = inputs[1]->shape();

        if (list[0] == unknown || map[0] == unknown)
            return create<Placeholder>(list.pop().push(unknown));

        return create<Placeholder>(list.pop().push(map[0]));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("list_remap", list_remap, _list_remap_eval);

    /**
     * @brief Returns an element from a given list.
     *
     */
    TokenReference list_element(const ListReference &list, int index)
    {
        return list->get(index);
    }

    TokenReference _list_element_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");
        auto list = inputs[0]->shape();
        return create<Placeholder>(list.pop());
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("list_element", list_element, _list_element_eval);

    /**
     * @brief Returns a virtual list with the given variable replicated a given number of times.
     *
     */
    TokenReference repeat_element(const TokenReference &value, int length)
    {
        VERIFY(length >= 1, "List length should be 1 or more");
        return create<ConstantList>(value, length);
    }

    TokenReference _repeat_element_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");
        auto value = inputs[0]->shape();
        return create<Placeholder>(value.push(get_size(inputs[1])));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("list_repeat", repeat_element, _repeat_element_eval);

    /**
     * @brief Returns a list from start to end with linear progression over length elements.
     *
     */
    TokenReference list_range(float start, float end, int length, bool round)
    {
        VERIFY(length >= 1, "List length should be 1 or more");


        if (round)
        {
            std::vector<int> data;
            for (size_t i = 0; i < (size_t)length; i++)
            {
                data.push_back((int)((i / (float)length) * (end - start) + start));
            }

            return create<IntegerVector>(make_span(data));
        }
        else
        {

            std::vector<float> data;
            for (size_t i = 0; i < (size_t)length; i++)
            {
                data.push_back((i / (float)length) * (end - start) + start);
            }

            return create<FloatVector>(make_span(data));
        }
    }

    TokenReference _list_range_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 4, "Incorrect number of parameters");
        auto length = get_size(inputs[2]);
        auto round = get_size(inputs[3]);

        if (round)
            return create<Placeholder>(Shape(IntegerType, {length}));
        else
            return create<Placeholder>(Shape(FloatType, {length}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("list_range", list_range, _list_range_eval);

    /**
     * @brief Creates a permutation mapping.
     *
     */
    TokenReference list_permute(const ListReference &list, int seed)
    {
        size_t length = list->length();

        std::vector<int> indices;
        for (size_t i = 0; i < length; i++)
        {
            indices.push_back((int)i);
        }
        std::shuffle(indices.begin(), indices.end(), create_generator(seed));

        return create<MappedList>(list, indices);
    }

    TokenReference _list_permute_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");
        auto list = inputs[0]->shape();
        return create<Placeholder>(list);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("list_permute", list_permute, _list_permute_eval);

    /**
     * @brief Creates a random permutation of indices from 0 to length.
     *
     */
    TokenReference make_permutation(int length, int seed)
    {

        std::vector<int> indices;
        for (size_t i = 0; i < (size_t)length; i++)
        {
            indices.push_back((int)i);
        }
        std::shuffle(indices.begin(), indices.end(), create_generator(seed));

        return create<IntegerVector>(make_span(indices));
    }

    TokenReference _make_permutation_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Incorrect number of parameters");
        auto length = get_size(inputs[0]);
        return create<Placeholder>(Shape(IntegerType, {length}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("list_permutation", make_permutation, _make_permutation_eval);

    /**
     * @brief Returns a scalar length of an input list.
     *
     */
    int list_length(const ListReference &list)
    {
        return (int)list->length();
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("list_length", list_length, constant_shape<int>);

    /**
     * @brief Returns a list with the given elements. This is a generic list that can contain any type of elements.
     *
     */
    TokenReference make_list_run(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 0, "No inputs");

        std::vector<TokenReference> list;

        for (size_t i = 0; i < inputs.size(); i++)
            list.push_back(inputs[i].reborrow());

        return create<GenericList>(make_span(list));
    }

    TokenReference make_list_eval(const TokenList &inputs) 
    {
        // Determine list type from elements, the size is the number of elements

        VERIFY(inputs.size() > 0, "No inputs");

        Shape shape = inputs[0]->shape();

        for (size_t i = 1; i < inputs.size(); i++)
        {
            shape = shape & inputs[i]->shape();
        }

        return create<Placeholder>(shape.push(inputs.size()));

    }

    PIXELPIPES_COMPUTE_OPERATION("make_list", make_list_run, make_list_eval);

}
