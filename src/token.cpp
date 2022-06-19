
#include <pixelpipes/token.hpp>

namespace pixelpipes
{

    Shape Token::shape() const
    {
        return Shape(AnyType, SizeSpan{});
    }

    std::string Token::describe() const
    {
        std::stringstream ss;
        describe(ss);
        return ss.str();
    }

    void List::describe(std::ostream &os) const
    {
        os << "[List: length " << length() << "]";
    }

    GenericList::GenericList(const std::initializer_list<TokenReference> &elements) : _elements{elements}
    {

        if (elements.size() == 0)
        {
            _shape = Shape();
        }
        else
        {

            Shape eshape = _elements[0]->shape();

            for (auto l = _elements.begin(); l != _elements.end(); l++)
            {
                _shape = _shape & (*l)->shape();
            }

            _shape = _shape.push(_elements.size());
        }
    }

    GenericList::GenericList(const View<TokenReference> &elements) : _elements{elements}
    {

        if (elements.size() == 0)
        {
            _shape = Shape();
        }
        else
        {
            _shape = _elements[0]->shape();

            for (auto l = _elements.begin(); l != _elements.end(); l++)
            {
                _shape = _shape & (*l)->shape();
            }

            _shape = _shape.push(_elements.size());
        }
    }

    GenericList::~GenericList() = default;

    Shape GenericList::shape() const
    {
        return _shape;
    }

    size_t GenericList::length() const
    {
        return _elements.size();
    }

    TokenReference GenericList::get(size_t index) const
    {
        if (index >= _elements.size())
        {
            throw TypeException("Index out of range");
        }

        return _elements[index].reborrow();
    }

}
