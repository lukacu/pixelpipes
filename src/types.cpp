#include <sstream>

#include <pixelpipes/types.hpp>

namespace pixelpipes {

BaseException::BaseException(std::string reason): reason(reason) {}

const char * BaseException::what() const throw () {
   	return reason.c_str();
}

VariableException::VariableException(std::string reason): BaseException(reason) {}

std::string Variable::describe() const {
    std::stringstream ss;
    describe(ss);
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Variable& v) {
    v.describe(os);
    return os;
} 

std::ostream& operator<<(std::ostream& os, const SharedVariable& v) {
    v->describe(os);
    return os;
} 

void List::describe(std::ostream& os) const {
    os << "[List: length " << size() << "]";
}

TypeIdentifier List::type() const {
    return GetTypeIdentifier<List>();
}

}