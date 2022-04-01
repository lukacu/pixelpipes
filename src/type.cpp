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

	typedef std::tuple<TypeName, TypeValidator, TypeResolver, SharedModule> TypeData;
	typedef std::map<TypeIdentifier, TypeData> TypeMap;

	TypeMap &types()
	{
		static TypeMap map;
		return map;
	}

    Type::Type(const Type& t): _id(t._id), _parameters(t._parameters) {

	}

	Type::Type(TypeIdentifier id) : Type(id, {})
	{
	}

	Type::Type(TypeIdentifier id, const TypeParameters parameters) : _id(id), _parameters(parameters)
	{
		if (id == AnyType) {
			return;
		}

/*
		if (types().find(id) == types().end())
		{
			throw TypeException("Unknown type");
		}
*/
	}

	TypeIdentifier Type::identifier() const
	{
		return _id;
	}

	TypeName Type::name() const
	{

		auto data = types().find(_id);

		return std::get<0>(data->second);
	}

	bool Type::has(const std::string key) const
	{

		auto val = _parameters.find(key);

		return !(val == _parameters.end());
	}

	void type_register(TypeIdentifier i, std::string_view name, TypeValidator validator, TypeResolver resolver)
	{

		DEBUGMSG("Registering reader for type %s (%p) \n", std::string(name).data(), i);

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

		types().insert(TypeMap::value_type{i, TypeData{name, validator, resolver, source}});
	}

	Type type_make(TypeIdentifier i, std::map<std::string, std::any> parameters)
	{
		auto item = types().find(i);

		if (item == types().end())
		{
			return Type(AnyType);
		}

		return std::get<1>(item->second)(parameters);
	}

    TypeIdentifier type_find(TypeName name) {
		for (auto d : types())
		{
			if (std::get<0>(d.second) == name)
			{
				return d.first;
			}
		}
		throw TypeException(Formatter() << "Type for name " << name << " not found");
	}


	Type type_common(const Type &me, const Type &other)
	{
		auto item = types().find(me.identifier());

		if (item == types().end())
		{
			return Type(AnyType);
		}

		return std::get<2>(item->second)(me, other);


	}

	SharedModule type_source(TypeIdentifier i)
	{

		auto item = types().find(i);

		if (item == types().end())
		{
			throw TypeException("Unknown type");
		}

		return std::get<3>(item->second);
	}

	std::string_view type_name(TypeIdentifier i)
	{

		auto item = types().find(i);

		if (item == types().end())
		{
			return "???";
		}

		return std::get<0>(item->second);
	}

	Type default_type_resolve(const Type & a, const Type & b)
	{
		if (a.identifier() == b.identifier())
			return Type(a.identifier());

		return Type(AnyType);
	}


}