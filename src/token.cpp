
#include <pixelpipes/token.hpp>
#include <pixelpipes/tensor.hpp>

namespace pixelpipes
{

    Token::~Token() {

    }

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

    Placeholder::Placeholder(const Shape &shape) : _shape{shape}
    {
    }

    Placeholder::Placeholder(const Placeholder &that) : _shape{that._shape} 
    {
    }

    Placeholder::Placeholder(const Type &type, const Sizes &shape) : _shape{type, shape}
    {
    }

    Placeholder::Placeholder(const Type &type) : _shape{type}
    {
    }

    Placeholder::~Placeholder() = default;

    void Placeholder::describe(std::ostream &os) const
    {
        os << "[Placeholder: " << _shape << "]";
    }

    Shape Placeholder::shape() const
    {
        return _shape;
    }

    TokenReference Placeholder::dummy() const
    {
        if (_shape.is_scalar()) {
            if (_shape.element() == IntegerType) {
                return create<IntegerScalar>(0);
            } else if (_shape.element() == FloatType) {
                return create<FloatScalar>(0.0f);
            } else {
                return create<CharScalar>(0);
            }
        } else {
            return create_tensor(_shape);
        }
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
