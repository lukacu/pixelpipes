#pragma once
#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <string_view>
#include <map>
#include <any>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <optional>

#include <pixelpipes/base.hpp>
#include <pixelpipes/module.hpp>

namespace pixelpipes
{

    namespace detail
    {

#if _BUILD_SUPPORT_MAGIC_NAMES
        template <typename T>
        constexpr const auto &_type_name_raw()
        {
#ifdef _MSC_VER
            return __FUNCSIG__;
#elif defined(__clang__) || defined(__GNUC__)
            return __PRETTY_FUNCTION__;
#endif
        }

        struct _raw_name_format
        {
            std::size_t leading_junk = 0, trailing_junk = 0;
        };

        // Returns `false` on failure.
        inline constexpr bool _get_raw_name_format(_raw_name_format *format)
        {
            const auto &str = _type_name_raw<int>();
            for (std::size_t i = 0;; i++)
            {
                if (str[i] == 'i' && str[i + 1] == 'n' && str[i + 2] == 't')
                {
                    if (format)
                    {
                        format->leading_junk = i;
                        format->trailing_junk = sizeof(str) - i - 3 - 1; // `3` is the length of "int", `1` is the space for the null terminator.
                    }
                    return true;
                }
            }
            return false;
        }

        inline static constexpr _raw_name_format format =
            []
        {
            static_assert(_get_raw_name_format(nullptr), "Unable to figure out how to generate type names on this compiler.");
            _raw_name_format fmt;
            _get_raw_name_format(&fmt);
            return fmt;
        }();

        template <typename T>
        [[nodiscard]] constexpr auto type_name()
        {
            return std::string_view{_type_name_raw<T>() + format.leading_junk, sizeof(_type_name_raw<T>()) - format.trailing_junk - format.leading_junk - 1};
        }

#else

        constexpr std::string_view _undefined = "unsupported";

        template <typename T>
        [[nodiscard]] constexpr auto type_name()
        {

            return _undefined; //{_undefined, sizeof(undefined)-1};
        }
#endif

        template <typename T>
        struct inner_type
        {
            using type = T;
        };

        template <class T, class Alloc>
        struct inner_type<std::vector<T, Alloc>>
        {
            using type = typename inner_type<T>::type;
        };

        template <class T>
        using inner_type_t = typename inner_type<T>::type;

    }

    /**
     * The type of a type id.
     */
    typedef uintptr_t TypeIdentifier;

    constexpr static TypeIdentifier AnyType = 0;

    constexpr static TypeIdentifier ListType = 1;

    /**
     * The function that returns the type id.
     *
     * It uses the pointer to the static data member of a class template to achieve this.
     * Altough the value is not predictible, it's stable (I hope).
     */
    template <typename T>
    auto PIXELPIPES_TYPE_API GetTypeIdentifier() noexcept -> TypeIdentifier
    {
        if constexpr (detail::is_container<T>::value) {
            return GetTypeIdentifier<typename T::value_type >() + ListType;
        } else if constexpr (detail::is_reference<T>::value) {
            return  GetTypeIdentifier<typename std::pointer_traits<T>::element_type >();
        } else  {
            return reinterpret_cast<uintptr_t>(&detail::TypeIdentifierToken<T>::id);
        }
    }

    typedef std::string TypeName;

#define VIEWCHARS(S) std::string(S).c_str()

    class PIXELPIPES_API TypeException : public BaseException
    {
    public:
        TypeException(std::string reason);
    };

    inline void verify(bool condition, std::string reason = std::string("Assertion failed"))
    {

        if (!condition)
            throw TypeException(reason);
    }

    template <typename T>
    auto PIXELPIPES_TYPE_API GetListIdentifier() noexcept -> TypeIdentifier
    {
        return ((uintptr_t)&detail::TypeIdentifierToken<T>::id) + ListType;
    }

    typedef std::map<std::string, std::any> TypeParameters;
/*
    class PIXELPIPES_API String
    {
    public:
        ~String();

        explicit String(char const *const text);

        String(String const &other);

        String &operator=(String rhs);

        String &operator=(char const *const text);

        String &swap(String &other);

        operator char const *() const
        {
            return c_str();
        }

        char const *c_str() const
        {
            return get();
        }

        friend String as_string(std::string const &text)
        {
            return String(text.c_str());
        }

        friend std::string to_string(String const &text)
        {
            return text.c_str();
        }

        friend char const *to_cstr(String const &text)
        {
            return text.c_str();
        }

    private:
        const char *get() const;

    private:
        struct Data;
        Implementation<Data> data;
    };
*/
    class Type
    {
    public:
        PIXELPIPES_API Type(const Type &);

        PIXELPIPES_API Type(TypeIdentifier id, const TypeParameters parameters);

        PIXELPIPES_API Type(TypeIdentifier id);

        virtual PIXELPIPES_API ~Type() = default;

        TypeIdentifier PIXELPIPES_API identifier() const;

        TypeName PIXELPIPES_API name() const;

        template <typename T>
        T parameter(const std::string key) const
        {

            auto val = _parameters.find(key);

            if (val == _parameters.end())
                throw TypeException("Parameter not found");

            try
            {
                return std::any_cast<T>(val->second);
            }
            catch (const std::bad_any_cast &e)
            {
                throw TypeException("Parameter not convertable");
            }
        }

        bool has(const std::string key) const;

    private:
        TypeIdentifier _id;

        std::map<std::string, std::any> _parameters;
    };


	typedef Type(*TypeResolver) (const Type &, const Type &);

	typedef Type(*TypeValidator) (const TypeParameters &);

    //typedef Function<Type(const Type &, const Type &)> TypeResolver;

    //typedef Function<Type(const TypeParameters &)> TypeValidator;

    void PIXELPIPES_API type_register(TypeIdentifier i, std::string_view name, TypeValidator, TypeResolver);

    Type PIXELPIPES_API type_make(TypeIdentifier i, std::map<std::string, std::any> parameters);

    Type PIXELPIPES_API type_common(const Type &me, const Type &other);

    SharedModule PIXELPIPES_API type_source(TypeIdentifier i);

    TypeIdentifier PIXELPIPES_API type_find(TypeName name);

    TypeName PIXELPIPES_API type_name(TypeIdentifier i);

    Type default_type_resolve(const Type &, const Type &);
	/*
	template<TypeIdentifier T, auto NAME, auto V, auto R>
	void PIXELPIPES_API type_register() {
		type_register(T, NAME, V, RESOLVER);
	}*/

	template<TypeIdentifier T>
	Type default_type_constructor(const TypeParameters&) {
		return Type(T);
	}

#define DEFAULT_TYPE_CONSTRUCTOR(T) [](const TypeParameters&) { return Type(T); }

#define PIXELPIPES_REGISTER_TYPE(T, NAME, VALIDATOR, RESOLVER) static AddModuleInitializer CONCAT(__type_init_, __COUNTER__)([]() { type_register(T, NAME, VALIDATOR, RESOLVER); })

#define PIXELPIPES_REGISTER_TYPE_DEFAULT(T, NAME) static AddModuleInitializer CONCAT(__type_init_, __COUNTER__)([]() { type_register(T, NAME, DEFAULT_TYPE_CONSTRUCTOR(T), default_type_resolve); })

}