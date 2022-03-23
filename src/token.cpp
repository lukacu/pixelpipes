
#include <pixelpipes/token.hpp>

namespace pixelpipes
{
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
        os << "[List: length " << size() << "]";
    }

    TypeIdentifier List::type() const
    {
        return GetTypeIdentifier<List>();
    }

}