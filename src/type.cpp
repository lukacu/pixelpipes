#include <sstream>

#include <pixelpipes/type.hpp>

namespace pixelpipes
{

	// TODO: base exception impl probably does not belong here

	BaseException::BaseException(std::string reason) : reason(reason) {}

	const char *BaseException::what() const throw()
	{
		return reason.c_str();
	}

	TypeException::TypeException(std::string reason) : BaseException(reason) {}

}