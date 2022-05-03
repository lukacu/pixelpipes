
#pragma once

#include <cstring>
#include <sstream>
#include <type_traits>
#include <vector>
#include <utility>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#ifdef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_API __declspec(dllexport)
#else
#define PIXELPIPES_API __declspec(dllimport)
#endif
#define PIXELPIPES_INTERNAL
#else
/* allow use of -fvisibility=hidden -fvisibility-inlines-hidden */
#define PIXELPIPES_API __attribute__((visibility("default")))
#define PIXELPIPES_INTERNAL __attribute__((visibility("hidden")))
#endif

#ifdef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_API_TEMPLATE
#else
#define PIXELPIPES_API_TEMPLATE extern
#endif

#ifdef _WIN32
const std::string os_pathsep(";");
#else
const std::string os_pathsep(":");
#endif

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

#define UNUSED(expr)  \
    do                \
    {                 \
        (void)(expr); \
    } while (0)

namespace pixelpipes
{

    template <typename T>
    class Span;
    template <typename T>
    class Sequence;

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
        struct is_container
        {
            static constexpr bool value = false;
        };

        template <template <typename...> typename C, typename U>
        struct is_container<C<U>>
        {
            static constexpr bool value =
                std::is_same<C<U>, std::vector<U>>::value || std::is_same<C<U>, Span<U>>::value  || std::is_same<C<U>, Sequence<U>>::value;
        };

        template <typename T>
        struct is_reference
        {
            static constexpr bool value = std::is_pointer<T>::value;
        };

        template <template <typename...> typename C, typename U>
        struct is_reference<C<U>>
        {
            static constexpr bool value =
                std::is_same<C<U>, std::shared_ptr<U>>::value || std::is_same<C<U>, std::weak_ptr<U>>::value;
        };

        template <typename C>
        struct is_string
        {
            static constexpr bool value =
                std::is_same<C, std::string>::value;
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
        struct has_contiguous_memory : std::false_type
        {
        };

        template <typename T, typename U>
        struct has_contiguous_memory<std::vector<T, U>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<std::vector<bool, T>> : std::false_type
        {
        };

        template <typename T, typename U, typename V>
        struct has_contiguous_memory<std::basic_string<T, U, V>> : std::true_type
        {
        };

        template <typename T, std::size_t N>
        struct has_contiguous_memory<std::array<T, N>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<Span<T>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<Sequence<T>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<T[]> : std::true_type
        {
        };

        template <typename T, std::size_t N>
        struct has_contiguous_memory<T[N]> : std::true_type
        {
        };

        template <typename T>
        struct PIXELPIPES_API TypeIdentifierToken
        {
            // Having a static data member will ensure that it has only one address for the whole program.
            // Satic data member having different types will ensure it won't get optimized.
            static const T *const id;
        };

        template <typename T>
        const T *const TypeIdentifierToken<T>::id = nullptr;

    }

    template <typename T>
    class PIXELPIPES_API Implementation
    {
    protected:
        //[[no_unique_address]]
        std::default_delete<T> del;
        T *raw;

    public:
        template <typename... Args>
        Implementation(Args... args) : raw(new T(std::forward<Args>(args)...))
        {
        }

        Implementation(const Implementation<T> &other) : raw(new T(*other.raw))
        {
        }

        Implementation(Implementation<T> &&other) : raw(new T(std::move(*other.raw)))
        {
        }

        ~Implementation() noexcept(std::is_nothrow_destructible<T>::value)
        {
            del(raw);
        }

        Implementation &operator=(const Implementation<T> &other) noexcept(std::is_nothrow_copy_assignable<T>::value)
        {
            *raw = *other.raw;
            return *this;
        }

        Implementation &operator=(Implementation<T> &&other) noexcept(std::is_nothrow_move_assignable<T>::value)
        {
            *raw = std::move(*other.raw);
            return *this;
        }

        T *operator->() noexcept
        {
            return raw;
        }

        const T *operator->() const noexcept
        {
            return raw;
        }

        T &operator*() &noexcept
        {
            return *raw;
        }

        const T &operator*() const &noexcept
        {
            return *raw;
        }

        T &&operator*() &&noexcept
        {
            return std::move(*raw);
        }
    };

    template <typename T>
    class PIXELPIPES_API Span
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;
        using iterator = T *;
        using const_iterator = const T *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        Span() = default;

        Span(const T *ptr, size_t size)
            : m_ptr(ptr), m_size(size)
        {
        }

        Span(const Span &) = default;
        Span(Span &&) = default;

        Span &operator=(const Span &) = default;
        Span &operator=(Span &&) = default;

        const_pointer get() const noexcept { return m_ptr; }

        const_reference at(size_type i) const
        {
            return *(m_ptr + i);
        }

        const_reference operator[](size_type i) const
        {
            return at(i);
        }

        const_reference front() const
        {
            return at(0);
        }

        const_reference back() const
        {
            return at(m_size - 1);
        }

        const_pointer data() const noexcept
        {
            return m_ptr;
        }

        const_iterator begin() const noexcept
        {
            return data();
        }

        const_iterator cbegin() const noexcept
        {
            return data();
        }

        const_iterator end() const noexcept
        {
            return data() + m_size;
        }

        const_iterator cend() const noexcept
        {
            return data() + m_size;
        }

        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator crbegin() const noexcept
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator(begin());
        }

        const_reverse_iterator crend() const noexcept
        {
            return const_reverse_iterator(begin());
        }

        // capacity
        bool empty() const noexcept
        {
            return m_size == 0;
        }

        size_t size() const noexcept
        {
            return m_size;
        }

        // slicing
        Span slice(size_t off, size_t count = size_t(-1)) const
        {
            if (off > m_size)
                return Span(m_ptr + m_size, 0);
            auto newSize = m_size - off;
            if (count > newSize)
                count = newSize;
            return Span(m_ptr + off, count);
        }

    protected:
        const T *m_ptr = nullptr;
        size_t m_size = 0;
    };

    template <typename T>
    Span<T> make_span(const T *ptr, size_t size)
    {
        return Span<T>(ptr, size);
    }

    template <typename T, size_t N>
    Span<T> make_span(const T (&ar)[N])
    {
        return Span<T>(ar, N);
    }

    template <typename Container>
    auto make_span(const Container &c, size_t offset = 0) -> Span<typename Container::value_type>
    {
        static_assert(detail::has_contiguous_memory<Container>::value, "Only contiguous containers");

        return Span<typename Container::value_type>(std::data(c) + offset, c.size() - offset);
    }

    template <typename T>
    class PIXELPIPES_API Sequence : public Span<T>
    {

    public:
        Sequence() : Span<T>()
        {
        }

        Sequence(const T *ptr, size_t size) : Span<T>()
        {
            copy(ptr, size);
        }

        Sequence(std::initializer_list<T> c) : Span<T>()
        {
            auto data = new T[c.size()];
            this->m_size = c.size();
            size_t i = 0;
            for (auto x = c.begin(); x != c.end(); x++, i++)
            {
                data[i] = *x;
            }
            this->m_ptr = data;
        }

        template <typename Container>
        Sequence(const Container &c, size_t offset = 0) : Span<T>()
        {
            if constexpr (detail::has_contiguous_memory<Container>::value)
            {
                copy(std::data(c) + offset, c.size() - offset);
            }
            else
            {
                auto data = new T[c.size() - offset];
                this->m_size = c.size() - offset;
                size_t i = 0;
                for (auto x = c.begin() + offset; x != c.end(); x++, i++)
                {
                    data[i] = *x;
                }
                this->m_ptr = data;
            }
        }

        Sequence(const Span<T> &s) : Sequence(std::data(s), s.size())
        {
        }

        Sequence(const Sequence &s) : Sequence(std::data(s), s.size())
        {
        }

        Sequence(Span<T> &&s) : Sequence(std::data(s), s.size())
        {
        }

        Sequence(Sequence &&s) : Span<T>(s.m_ptr, s.m_size)
        {
            // Steal data
            s.m_ptr = nullptr;
        }

        Sequence &operator=(const Sequence &s)
        {
            copy(s.m_ptr, s.m_size);
        }

        Sequence &operator=(Sequence &&s)
        {

            cleanup();
            this->m_ptr = s.m_ptr;
            this->m_size = s.m_size;
            s.m_ptr = nullptr;
        }

        ~Sequence()
        {
            cleanup();
        }

    private:
        void cleanup()
        {
            if (this->m_ptr)
            {
                delete[] this->m_ptr;
            }
            this->m_ptr = nullptr;
        }

        void copy(const T *ptr, size_t size)
        {
            auto data = new T[size];

            if constexpr (std::is_trivially_copyable<T>::value)
            {
                std::memcpy((void *)data, (void *)ptr, sizeof(T) * size);
            }
            else
            {
                std::copy(ptr, ptr + size, data);
            }

            cleanup();

            this->m_ptr = data;
            this->m_size = size;
        }
    };

    class Formatter
    {
    public:
        PIXELPIPES_API Formatter() {}
        PIXELPIPES_API ~Formatter() {}

        template <typename Type>
        PIXELPIPES_API Formatter &operator<<(const std::vector<Type> &value)
        {

            stream << value.size() << " - ";
            for (Type t : value)
                stream << t << " ";
            return *this;
        }

        template <typename Type>
        Formatter &operator<<(const Type &value)
        {
            stream << value;
            return *this;
        }

        PIXELPIPES_API std::string str() const { return stream.str(); }
        PIXELPIPES_API operator std::string() const { return stream.str(); }

        PIXELPIPES_API const char *c_str() const { return stream.str().c_str(); }

        enum ConvertToString
        {
            to_str
        };
        PIXELPIPES_API std::string operator>>(ConvertToString) { return stream.str(); }

    private:
        std::stringstream stream;

        Formatter(const Formatter &);
        Formatter &operator=(Formatter &);
    };

    class PIXELPIPES_API BaseException
    {
    public:
        BaseException(std::string reason);
        ~BaseException();

        const char *what() const throw();

    private:
        char *reason;
    };

}