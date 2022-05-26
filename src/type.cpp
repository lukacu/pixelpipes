#include <cstring>
#include <sstream>
#include <algorithm>

#include <pixelpipes/type.hpp>

#include "debug.h"

namespace pixelpipes
{

	// TODO: base exception impl probably does not belong here

	BaseException::BaseException(std::string r) : reason(nullptr)
	{

		reason = new char[r.size()];
		std::strcpy(reason, r.c_str());
	}

	BaseException::~BaseException()
	{
		if (reason)
			delete[] reason;
	}

	const char *BaseException::what() const throw()
	{
		return reason;
	}

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
	typedef std::tuple<TypeName, SharedModule> TypeData;
	typedef std::map<TypeIdentifier, TypeData> TypeMap;

	TypeSize::TypeSize() : data(-1) {}

	TypeSize::TypeSize(int x) : data(x) {}

	TypeSize::TypeSize(size_t x) : data((int)x) {}

	TypeSize::TypeSize(const TypeSize &x) : data(x.data) {}

	TypeSize TypeSize::operator+(const TypeSize &other) const
	{
		if (data < 0 || other.data < 0)
			return MIN(data, other.data);
		return data + other.data;
	}

	TypeSize TypeSize::operator-(const TypeSize &other) const
	{
		if (data < 0 || other.data < 0)
			return MIN(data, other.data);
		return data - other.data;
	}

	TypeSize TypeSize::operator*(const TypeSize &other) const
	{
		if (data < 0 || other.data < 0)
			return MIN(data, other.data);
		return data * other.data;
	}

	TypeSize TypeSize::operator/(const TypeSize &other) const
	{
		if (data < 0 || other.data < 0)
			return MIN(data, other.data);
		return data / other.data;
	}

	bool TypeSize::operator==(const TypeSize &other) const
	{
		// if (data == -1 || other.data < 0)

		return data == other.data;
	}

	TypeMap &types()
	{
		static TypeMap map;
		return map;
	}

	Type::Type(const Type &t) : Type(t._element, t._shape)
	{
	}

	Type::Type(TypeIdentifier element) : Type(element, {})
	{
	}

	Type::Type(TypeIdentifier element, const Span<TypeSize> shape) : _element(element), _shape(shape)
	{
		VERIFY((element & TensorIdentifierMask) == 0, "Only scalar element type identifiers allowed");
	}

	TypeIdentifier Type::element() const
	{
		return _element;
	}

	TypeName Type::name() const
	{
		auto data = types().find(_element);
		return std::get<0>(data->second);
	}

	TypeSize Type::shape(size_t i) const
	{
		return _shape[i];
	}

	size_t Type::dimensions() const
	{
		return _shape.size();
	}

	bool Type::is_fixed() const
	{
		bool fixed = true;
		for (auto d : _shape) {
			fixed &= d.data > 0;
		}
		return fixed;
	}

	bool Type::is_scalar() const
	{
		bool scalar = true;
		for (auto d : _shape) {
			scalar &= d.data == 1;
		}
		return scalar;
	}

	void type_register(TypeIdentifier i, std::string_view name)
	{

		DEBUGMSG("Registering type %s (%ld) \n", std::string(name).data(), i);

		if (types().find(i) != types().end())
		{
			throw TypeException("Type already registered");
		}

		for (auto d : types())
		{

			if (std::get<0>(d.second) == name)
			{
				throw TypeException(Formatter() << "Type already registered with name " << name);
			}
		}

		SharedModule source = Module::context();

		types().insert(TypeMap::value_type{i, TypeData{name, source}});
	}

	TypeIdentifier type_find(TypeName name)
	{
		for (auto d : types())
		{
			if (std::get<0>(d.second) == name)
			{
				return d.first;
			}
		}
		throw TypeException(Formatter() << "Type for name " << name << " not found");
	}

	SharedModule type_source(TypeIdentifier i)
	{

		auto item = types().find(i);

		if (item == types().end())
		{
			throw TypeException((Formatter() << "Unknown type identifier: " << i << "").str());
		}

		return std::get<1>(item->second);
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

}