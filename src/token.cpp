
#include <pixelpipes/token.hpp>

namespace pixelpipes
{
    PIXELPIPES_REGISTER_TYPE(IntegerType, "integer", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(FloatType, "float", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(BooleanType, "boolean", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(StringType, "string", do_not_create, do_not_resolve);

    PIXELPIPES_REGISTER_TYPE(IntegerListType, "integer_list", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(FloatListType, "float_list", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(BooleanListType, "boolean_list", do_not_create, do_not_resolve);
    PIXELPIPES_REGISTER_TYPE(StringListType, "string_list", do_not_create, do_not_resolve);


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