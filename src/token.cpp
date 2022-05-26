
#include <pixelpipes/token.hpp>

namespace pixelpipes
{
    
    PIXELPIPES_REGISTER_TYPE(TokenIdentifier, "token");

    PIXELPIPES_REGISTER_TYPE(IntegerIdentifier, "integer");
    PIXELPIPES_REGISTER_TYPE(FloatIdentifier, "float");
    PIXELPIPES_REGISTER_TYPE(BooleanIdentifier, "boolean");
    PIXELPIPES_REGISTER_TYPE(StringIdentifier, "string");

    PIXELPIPES_REGISTER_TYPE(IntegerListIdentifier, "integer_list");
    PIXELPIPES_REGISTER_TYPE(FloatListIdentifier, "float_list");
    PIXELPIPES_REGISTER_TYPE(BooleanListIdentifier, "boolean_list");
    PIXELPIPES_REGISTER_TYPE(StringListIdentifier, "string_list");

    Type Token::type() const {
        return Type(type_id(), {});
    }

    std::string Token::describe() const
    {
        std::stringstream ss;
        describe(ss);
        return ss.str();
    }

    std::ostream &operator<<(std::ostream &os, const Token &v)
    {
        v.describe(os);
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const SharedToken &v)
    {
        v->describe(os);
        return os;
    }

    void List::describe(std::ostream &os) const
    {
        os << "[List: length " << size() << ", elements: " << type_name(element_type_id()) << "]";
    }

    TypeIdentifier List::type_id() const
    {
        return element_type_id() | TensorIdentifierMask;
    }

    Type List::type() const {
        return ListType(type_id(), size());
    }


}