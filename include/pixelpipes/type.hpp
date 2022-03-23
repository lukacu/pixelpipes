#pragma once
#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <string_view>
#include <map>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>
#include <initializer_list>

#include <pixelpipes/base.hpp>

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
            _raw_name_format format;
            _get_raw_name_format(&format);
            return format;
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

    template <typename T>
    struct is_vector
    {
        static constexpr bool value = false;
    };

    template <template <typename...> typename C, typename U>
    struct is_vector<C<U>>
    {
        static constexpr bool value =
            std::is_same<C<U>, std::vector<U>>::value;
    };

    template <typename C>
    struct is_string
    {
        static constexpr bool value =
            std::is_same<C, std::string>::value;
    };

    /**
     * The type of a type id.
     */
    typedef const void *TypeIdentifier;

    /**
     * The function that returns the type id.
     *
     * It uses the pointer to the static data member of a class template to achieve this.
     * Altough the value is not predictible, it's stable (I hope).
     */
    template <typename T>
    constexpr auto GetTypeIdentifier() noexcept -> TypeIdentifier
    {
        return &detail::TypeIdentifierToken<T>::id;
    }

    typedef std::string_view TypeName;

    template <typename T>
    constexpr TypeName GetTypeName();

#define VIEWCHARS(S) std::string(S).c_str()

    template <typename T, typename dummy = T>
    struct Type
    {

        constexpr static TypeIdentifier identifier = GetTypeIdentifier<T>();
        constexpr static TypeName name = GetTypeName<T>();
        constexpr static bool scalar = true;
    };

    template <typename T>
    struct Type<T, typename std::enable_if<is_vector<T>::value, T>::type>
    {

        constexpr static TypeIdentifier identifier = GetTypeIdentifier<T>();
        constexpr static std::string_view name = GetTypeName<T>();
        constexpr static bool scalar = false;
    };

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

}