
#include <pixelpipes/token.hpp>

namespace pixelpipes
{
    
    PIXELPIPES_REGISTER_TYPE_DEFAULT(IntegerType, "integer");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(FloatType, "float");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(BooleanType, "boolean");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(StringType, "string");

    PIXELPIPES_REGISTER_TYPE_DEFAULT(IntegerListType, "integer_list");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(FloatListType, "float_list");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(BooleanListType, "boolean_list");
    PIXELPIPES_REGISTER_TYPE_DEFAULT(StringListType, "string_list");

    Type Token::type() const {
        return type_make(type_id(), {});
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
        return element_type_id() | ListType;
    }

    Type List::type() const {
        return type_make(type_id(), {});
    }


}