
#pragma once

#include <cstdio>
#include <sstream>
#include <type_traits>

#ifdef _WIN32
#ifdef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_API __declspec(dllexport)
#else
#define PIXELPIPES_API __declspec(dllimport)
#endif
#define PIXELPIPES_INTERNAL
#elif __GNUC__ >= 4
/* allow use of -fvisibility=hidden -fvisibility-inlines-hidden */
#define PIXELPIPES_API __attribute__((visibility("default")))
#define PIXELPIPES_INTERNAL __attribute__((visibility("hidden")))
#else
#define PIXELPIPES_API
#define PIXELPIPES_INTERNAL
#endif

#ifdef _WIN32
const std::string os_pathsep(";");
#else
const std::string os_pathsep(":");
#endif

#ifdef PIXELPIPES_DEBUG
#define DEBUGMSG(...)        \
    {                        \
        std::printf(__VA_ARGS__); \
    }
#else
#define DEBUGMSG(...) \
    {                 \
    }
#endif

#define PRINTMSG(...)        \
    {                        \
        std::printf(__VA_ARGS__); \
    }

#define _STRINGIFY_IMPL(X) #X
#define STRINGIFY(X) _STRINGIFY_IMPL(X)

#define _CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) _CONCAT_IMPL(x, y)

#ifdef _MSC_VER // TODO: add version
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#elif defined(__clang__) // TODO: add version
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#elif defined(__GNUC__) && (__GNUC__ >= 8)
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#else
#define _BUILD_SUPPORT_MAGIC_NAMES 0
#endif

namespace pixelpipes
{

    namespace detail
    {

        template <class... T>
        constexpr bool always_false = false;

        inline void current_function_helper()
        {
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600))
#define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#define CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
#define CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#define CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#define CURRENT_FUNCTION __func__
#else
#define CURRENT_FUNCTION "(unknown)"
#endif
        }

        template <bool B, typename T = void>
        using enable_if_t = typename std::enable_if<B, T>::type;

        template <typename T, typename = void>
        struct is_input_iterator : std::false_type
        {
        };
        template <typename T>
        struct is_input_iterator<T, std::void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>
            : std::true_type
        {
        };

        template <typename T>
        class any_container
        {
            std::vector<T> v;

        public:
            any_container() = default;

            // Can construct from a pair of iterators
            template <typename It, typename = enable_if_t<is_input_iterator<It>::value>>
            any_container(It first, It last) : v(first, last) {}

            // Implicit conversion constructor from any arbitrary container type with values convertible to T
            template <typename Container, typename = enable_if_t<std::is_convertible<decltype(*std::begin(std::declval<const Container &>())), T>::value>>
            any_container(const Container &c) : any_container(std::begin(c), std::end(c)) {}

            // initializer_list's aren't deducible, so don't get matched by the above template; we need this
            // to explicitly allow implicit conversion from one:
            template <typename TIn, typename = enable_if_t<std::is_convertible<TIn, T>::value>>
            any_container(const std::initializer_list<TIn> &c) : any_container(c.begin(), c.end()) {}

            // Avoid copying if given an rvalue vector of the correct type.
            any_container(std::vector<T> &&v) : v(std::move(v)) {}

            // Moves the vector out of an rvalue any_container
            operator std::vector<T> &&() && { return std::move(v); }

            // Dereferencing obtains a reference to the underlying vector
            std::vector<T> &operator*() { return v; }
            const std::vector<T> &operator*() const { return v; }

            // -> lets you call methods on the underlying vector
            std::vector<T> *operator->() { return &v; }
            const std::vector<T> *operator->() const { return &v; }
        };

        template <std::size_t N>
        class static_string
        {
        public:
            constexpr explicit static_string(std::string_view str) noexcept : static_string{str, std::make_index_sequence<N>{}}
            {
                // static_assert(str.size() == N, "Illegal size");
            }

            constexpr const char *data() const noexcept { return chars_; }

            constexpr std::size_t size() const noexcept { return N; }

            constexpr operator std::string_view() const noexcept { return {data(), size()}; }

        private:
            template <std::size_t... I>
            constexpr static_string(std::string_view str, std::index_sequence<I...>) noexcept : chars_{str[I]..., '\0'} {}

            char chars_[N + 1];
        };

        template <>
        class static_string<0>
        {
        public:
            constexpr explicit static_string(std::string_view) noexcept {}

            constexpr const char *data() const noexcept { return nullptr; }

            constexpr std::size_t size() const noexcept { return 0; }

            constexpr operator std::string_view() const noexcept { return {}; }
        };

        template <typename T>
        struct TypeIdentifierToken
        {
            // Having a static data member will ensure that it has only one address for the whole program.
            // Satic data member having different types will ensure it won't get optimized.
            static const T *const id;
        };

        template <typename T>
        const T *const TypeIdentifierToken<T>::id = nullptr;

    }

    class Formatter
    {
    public:
        Formatter() {}
        ~Formatter() {}

        template <typename Type>
        Formatter & operator << (const Type & value)
        {
            stream << value;
            return *this;
        }

        std::string str() const         { return stream.str(); }
        operator std::string () const   { return stream.str(); }

        enum ConvertToString 
        {
            to_str
        };
        std::string operator >> (ConvertToString) { return stream.str(); }

    private:
        std::stringstream stream;

        Formatter(const Formatter &);
        Formatter & operator = (Formatter &);
    };

    class PIXELPIPES_API BaseException : public std::exception
    {
    public:
        BaseException(std::string reason);
        ~BaseException() = default;

        const char *what() const throw();

    private:
        std::string reason;
    };

}