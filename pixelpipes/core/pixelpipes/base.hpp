
#pragma once

#include <cstdio>

#ifdef _WIN32
# ifdef PIXELPIPES_BUILD_CORE
#  define PIXELPIPES_API __declspec(dllexport)
# else
#  define PIXELPIPES_API __declspec(dllimport)
# endif
# define PIXELPIPES_INTERNAL
#elif __GNUC__ >= 4
/* allow use of -fvisibility=hidden -fvisibility-inlines-hidden */
# define PIXELPIPES_API __attribute__ ((visibility("default")))
# define PIXELPIPES_INTERNAL __attribute__ ((visibility("hidden")))
#else
# define PIXELPIPES_API
# define PIXELPIPES_INTERNAL
#endif

#ifdef PIXELPIPES_DEBUG
#define DEBUGMSG(...) { printf(__VA_ARGS__); }
#else
#define DEBUGMSG(...) {}
#endif

#define PRINTMSG(...) { printf(__VA_ARGS__); }

#define _STRINGIFY_IMPL(X) #X 
#define STRINGIFY(X) _STRINGIFY_IMPL(X) 

#ifdef _MSC_VER // TODO: add version
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#elif defined(__clang__) // TODO: add version
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#elif  defined(__GNUC__) && (__GNUC__ >= 8)
#define _BUILD_SUPPORT_MAGIC_NAMES 1
#else
#define _BUILD_SUPPORT_MAGIC_NAMES 0
#endif

namespace pixelpipes {

namespace detail {

inline void current_function_helper()
{
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600))
# define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
# define CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
# define CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
# define CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
# define CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
# define CURRENT_FUNCTION __func__
#else
# define CURRENT_FUNCTION "(unknown)"
#endif

}

template <std::size_t N>
    class static_string {
    public:
    constexpr explicit static_string(std::string_view str) noexcept : static_string{str, std::make_index_sequence<N>{}} {
        //static_assert(str.size() == N, "Illegal size");
    }

    constexpr const char* data() const noexcept { return chars_; }

    constexpr std::size_t size() const noexcept { return N; }

    constexpr operator std::string_view() const noexcept { return {data(), size()}; }

    private:
    template <std::size_t... I>
    constexpr static_string(std::string_view str, std::index_sequence<I...>) noexcept : chars_{str[I]..., '\0'} {}

    char chars_[N + 1];
    };

    template <>
    class static_string<0> {
    public:
    constexpr explicit static_string(std::string_view) noexcept {}

    constexpr const char* data() const noexcept { return nullptr; }

    constexpr std::size_t size() const noexcept { return 0; }

    constexpr operator std::string_view() const noexcept { return {}; }
    };

}

class PIXELPIPES_API BaseException : public std::exception {
public:
    BaseException(std::string reason);
    ~BaseException() = default;

	const char * what () const throw ();

private:
    std::string reason;

};

}