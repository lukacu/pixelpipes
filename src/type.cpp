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
	/*
		struct String::Data
		{
		public:
			~Data()
			{
				delete text;
			}

			Data(char const *const text, size_t len) : len(len) 	//: text(std::strdup(text))
			{
				this->text = new char[len];
				std::copy_n(text, len, this->text);
			}

			Data *clone() const
			{
				return new Data(text, len);
			}

			char const * text = nullptr;
			const size_t len = 0;
		};

		String::String(char const *const text)
			: data(text)
		{
		}

		String::String(String const &other)
			: data(data->text)
		{
		}

		String &String::operator=(char const *const text)
		{
			return *this = String(text);
		}

		const char *String::get() const
		{
			return data->text;
		}
	*/

#define IS_UNKNOWN(S) (S == (unknown))

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

	Shape::Shape(TypeIdentifier element) : Shape(element, SizeSpan{})
	{
	}

	Shape::Shape(TypeIdentifier element, const View<Size>& shape) : _element(element), _shape(shape)
	{
	}

	Shape::Shape(TypeIdentifier element, const Sizes& shape) : _element(element), _shape(SizeSequence(shape))
	{
	}

	TypeIdentifier Shape::element() const
	{
		return _element;
	}

	Size Shape::operator[](size_t index) const
	{
		if (index >= _shape.size())
			return 1;
		return _shape.at(index);
	}

	size_t Shape::rank() const
	{
		return _shape.size();
	}

	size_t Shape::size() const
	{
		size_t s = 1;

		for (size_t i = 0; i < _shape.size(); i++) {
			s *= (size_t)_shape[i];
		}

		return s;
	}

	bool Shape::is_fixed() const
	{
		bool fixed = true;
		for (const auto& d : _shape)
		{
			fixed &= d.data > 0;
		}
		return fixed;
	}

	bool Shape::is_scalar() const
	{
		bool scalar = true;
		for (auto d : std::as_const(_shape))
		{
			scalar &= d.data == 1;
		}
		return scalar;
	}

	Shape Shape::cast(TypeIdentifier t) const
	{
		return Shape(t, _shape);
	}

	Shape Shape::push(Size s) const
	{
		std::vector<Size> _s;
		_s.push_back(s);
		_s.insert(_s.end(), _shape.begin(), _shape.end());

		return Shape(element(), make_view(_s));
	}

	Shape Shape::pop() const
	{
		return Shape(element(), make_view(_shape, 1));
	}

	Shape Shape::operator&(const Shape &other) const
	{

		std::vector<Size> _s;

		TypeIdentifier e = (other.element() == element()) ? element() : AnyType;

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
	/*
		void type_register(TypeIdentifier i, std::string_view name)
		{

			DEBUGMSG("Registering type %s (%ld) \n", std::string(name).data(), i);

			if (types().find(i) != types().end())
			{
				throw TypeException("Type already registered");
			}

			for (auto d = types().begin(); d != types().end(); d++)
			{

				if (std::get<0>(d->second) == name)
				{
					throw TypeException(Formatter() << "Type already registered with name " << name);
				}
			}

			ModuleReference source = Module::context();

			types().insert(TypeMap::value_type{i, TypeData{name, source.reborrow()}});
		}

		TypeIdentifier type_find(TypeName name)
		{
			for (auto d = types().begin(); d != types().end(); d++)
			{
				if (std::get<0>(d->second) == name)
				{
					return d->first;
				}
			}
			throw TypeException(Formatter() << "Type for name " << name << " not found");
		}

		ModuleReference type_source(TypeIdentifier i)
		{

			auto item = types().find(i);

			if (item == types().end())
			{
				throw TypeException((Formatter() << "Unknown type identifier: " << i << "").str());
			}

			return std::get<1>(item->second).reborrow();
		}

		TypeName type_name(TypeIdentifier i)
		{

			auto item = types().find(i);

			if (item == types().end())
			{
				return (Formatter() << "??? (" << i << ")").str();
			}

			return std::string(std::get<0>(item->second));
		}
	*/

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

}