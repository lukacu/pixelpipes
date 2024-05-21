#include <cstring>
#include <sstream>
#include <algorithm>

#include <pixelpipes/type.hpp>

#include "debug.h"

namespace pixelpipes
{

	static std::atomic<uint32_t> _ref_count(0);

    uint32_t debug_ref_count() {
		return _ref_count.load();
	}

    uint32_t debug_ref_inc() {
		return _ref_count.fetch_add(1);
	}

    uint32_t debug_ref_dec() {
		return _ref_count.fetch_sub(1);
	}

	// TODO: base exception impl probably does not belong here

    BaseException::BaseException() : BaseException("")
	{

	}

	BaseException::BaseException(std::string r) : reason(nullptr)
	{

		reason = new char[r.size()+1];
		std::strcpy(reason, r.c_str());
	}

	BaseException::BaseException(const BaseException& e) : reason(nullptr)
	{
		reason = new char[strlen(e.what())+1];
		std::strcpy(reason, e.what());
	}


	BaseException::~BaseException()
	{
		if (reason)
			delete[] reason;
	}

    BaseException& BaseException::operator=( const BaseException& e ) noexcept {
		if (reason)
			delete[] reason;

		reason = new char[strlen(e.what())];
		std::strcpy(reason, e.what());
		return *this;
	}


	const char *BaseException::what() const throw()
	{
		return reason;
	}

	IllegalStateException::IllegalStateException(std::string reason) : BaseException(reason) {}

	TypeException::TypeException(std::string reason) : BaseException(reason) {}
	
	#define IS_UNKNOWN(x) (x == unknown)

	Size Size::operator+(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return MAX(data, other.data);
		return data + other.data;
	}

	Size Size::operator-(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return MAX(data, other.data);
		return data - other.data;
	}

	Size Size::operator*(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return MAX(data, other.data);
		return data * other.data;
	}

	Size Size::operator/(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return MAX(data, other.data);
		return data / other.data;
	}

	Size Size::operator%(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return MAX(data, other.data);
		return data % other.data;
	}

	bool Size::operator==(const Size &other) const
	{
		return data == other.data;
	}


	Size::operator bool() const
	{
		return !IS_UNKNOWN(data);
	}

	Size Size::operator&(const Size &other) const
	{
		if (IS_UNKNOWN(data) || IS_UNKNOWN(other.data))
			return unknown;

		return Size(data == other.data ? data : unknown);
	}

	Shape::Shape() : Shape(AnyType)
	{
	}

	Shape::Shape(Type element) : Shape(element, SizeSpan{})
	{
	}

	Shape::Shape(Type element, const std::initializer_list<Size>& shape) : _element(element), _shape(shape)
	{
	}

	Shape::Shape(Type element, const View<Size>& shape) : _element(element), _shape(shape)
	{
	}

	Shape::Shape(Type element, const Sizes& shape) : _element(element), _shape(SizeSequence(shape))
	{
	}

	Type Shape::element() const
	{
		return (Type) (_element);
	}

	Size Shape::operator[](size_t index) const
	{
		if (is_anything())
			return unknown;

		if (index >= _shape.size())
			return 1;
		return _shape.at(index);
	}

	size_t Shape::rank() const
	{
		if (is_anything())
			return 0;

		return _shape.size();
	}

	size_t Shape::size() const
	{
		if (is_anything())
			return 0;

		Size s = 1;
      
		for (size_t i = 0; i < _shape.size(); i++) {
			s = s * _shape[i];
		}

		return (size_t)s;
	}

	bool Shape::is_fixed() const
	{	
		if (is_anything())
			return false;

		bool fixed = true;
		for (const auto& d : _shape)
		{
			fixed &= !IS_UNKNOWN(d);
		}
		return fixed;
	}

	bool Shape::is_scalar() const
	{
		if (is_anything())
			return false;

		if (element() != IntegerType && element() != FloatType && element() != BooleanType && element() != CharType && element() != ShortType && element() != UnsignedShortType)
			return false;

		bool scalar = true;
		for (auto d : std::as_const(_shape))
		{
			scalar &= d.data == 1;
		}
		return scalar;
	}

	bool Shape::is_anything() const
	{
		return _shape.size() == 1 && _shape[0] == 0;
	}

	Shape Shape::cast(Type t) const
	{
		return Shape(t, _shape);
	}

	Shape Shape::push(Size s) const
	{
		if (is_anything())
			return Shape(element(), {0});

		std::vector<Size> _s;
		_s.push_back(s);
		_s.insert(_s.end(), _shape.begin(), _shape.end());

		return Shape(element(), make_view(_s));
	}

	Shape Shape::pop() const
	{
		if (is_anything())
			return Shape(element(), {0});

		return Shape(element(), make_view(_shape, 1));
	}

	/** Calculates common denominator of two shapes
	 * 
	 * @param a First shape
	 * @param b Second shape
	 * @return Common denominator, undefined where shapes do not match
	*/
	Shape Shape::operator&(const Shape &other) const
	{
		Type e = (other.element() == element()) ? element() : AnyType;

		if (is_anything() || other.is_anything())
			return Shape(e, {0});

		std::vector<Size> _s;

		size_t _d = MAX(rank(), other.rank());

		_s.reserve(_d);

		for (size_t i = 0; i < _d; i++)
		{
			_s.push_back(this->operator[](i) & other[i]);
		}

		return Shape(e, make_span(_s));
	}

	bool Shape::operator==(const Shape &other) const
	{
		// TODO: what to do with Anything?
		if (is_anything() || other.is_anything())
			return true;
		
		if (element() != other.element())
			return false;

		if (rank() != other.rank())
			return false;

		for (size_t i = 0; i < rank(); i++)
		{
			if ((size_t) this->operator[](i) != (size_t) other[i])
				return false;
		}

		return true;
	}


    Shape AnythingShape()
    {
        return Shape(AnyType, {0});
    }


	typedef std::map<std::string, EnumerationMap> EnumerationRegistry;

	EnumerationRegistry &_enum_registry()
	{
		static EnumerationRegistry map;
		return map;
	}

	EnumerationMap describe_enumeration(std::string &name)
	{

		auto item = _enum_registry().find(name);

		if (item == _enum_registry().end())
		{

			throw std::invalid_argument("Unknown enumeration");
		}

		return item->second;
	}

	void register_enumeration(const std::string &name, EnumerationMap mapping)
	{

		auto item = _enum_registry().find(name);

		if (item == _enum_registry().end())
		{
			DEBUGMSG("Adding enumeration %s\n", name.c_str());
			_enum_registry().insert(std::pair<std::string, EnumerationMap>(name, mapping));
		}
	}

	const char* type_name(const Type t)
	{
		
		switch (t)
		{
		case BooleanType:
			return "boolean";
		case IntegerType:
			return "integer";
		case FloatType:
			return "float";
		case CharType:
			return "char";
		case ShortType:
			return "short";
		case UnsignedShortType:
			return "ushort";
		case AnyType:
			return "any";
		default:
			return "unknown";
		}
				
	}

}