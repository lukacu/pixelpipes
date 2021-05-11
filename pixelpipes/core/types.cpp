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

Sublist::Sublist(SharedList list, int from, int to): list(list), from(from), to(to) {

    if (!list) {
        throw VariableException("Empty parent list");
    }

    if (to < from || to >= (int) list->size() || from < 0) {
        throw VariableException("Illegal sublist range");
    }

}
 
size_t Sublist::size() const {

    return to - from + 1;

}

TypeIdentifier Sublist::element_type() const {

    return list->element_type();

}

SharedVariable Sublist::get(int index) const {

    index += from;

    if (index < 0 || index > to) {
        throw VariableException("Index out of range");
    }

    return list->get(index);

}


MappedList::MappedList(SharedList list, std::vector<int> map): list(list), map(map) {

    if (!list) {
        throw VariableException("Empty parent list");
    }

    for (auto index : map) {
        if (index < 0 || index >= (int) list->size())
            throw VariableException("Illegal list index");
    }

}

size_t MappedList::size() const {

    return map.size();

}

TypeIdentifier MappedList::element_type() const {

    return list->element_type();

}

SharedVariable MappedList::get(int index) const {

    if (index < 0 || index >= (int) map.size()) {
        throw VariableException("Index out of range");
    }

    return list->get(map[index]);

}


}