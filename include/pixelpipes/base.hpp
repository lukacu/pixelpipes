
#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>
#include <utility>
#include <memory>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

#include <pixelpipes/details/rtti.hpp>
#include <pixelpipes/details/enum.hpp>
#include <pixelpipes/details/utilities.hpp>
#include <pixelpipes/details/api.hpp>
#include <pixelpipes/details/pointer.hpp>

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

#ifndef MAX
#define MAX std::max
#endif

#ifndef MIN
#define MIN std::min
#endif

namespace pixelpipes
{

    template <typename T>
    class Span;
    template <typename T>
    class Sequence;

    namespace details
    {

        template <typename T>
        struct has_contiguous_memory<Span<T>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<Sequence<T>> : std::true_type
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
                std::is_same<C<U>, std::vector<U>>::value || std::is_same<C<U>, Span<U>>::value || std::is_same<C<U>, Sequence<U>>::value;
        };
    }

    class PIXELPIPES_API BaseException
    {
    public:
        BaseException();
        BaseException(std::string reason);
        BaseException(const BaseException& reason);
        ~BaseException();

        const char *what() const throw();

        BaseException& operator=( const BaseException& other ) noexcept;

    private:
        char *reason;
    };

    class PIXELPIPES_API IllegalStateException: public BaseException
    {
    public:
        IllegalStateException(std::string reason);
        IllegalStateException(const IllegalStateException& e) = default;

    };

inline void verify(bool condition, std::string reason = std::string("Assertion failed"))
    {

        if (!condition)
            throw IllegalStateException(reason);
    }

#define DEBUG_MODE

#ifdef DEBUG_MODE
#define DEBUG(X)                     \
    {                                \
        std::cout << X << std::endl; \
    }
#define VERIFY(C, M) verify((C), (M))
#else
#define DEBUG(X)
#define VERIFY(C, M)
#endif

    template <typename Result, typename... Args>
    struct PIXELPIPES_API abstract_function
    {
        virtual Result operator()(Args... args) = 0;
        virtual abstract_function *clone() const = 0;
        virtual ~abstract_function() = default;
    };

    template <typename Func, typename Result, typename... Args>
    class PIXELPIPES_API concrete_function : public abstract_function<Result, Args...>
    {
        Func f;

    public:
        concrete_function(const Func &x)
            : f(x)
        {
        }
        Result operator()(Args... args) override
        {
            return f(args...);
        }
        concrete_function *clone() const override
        {
            return new concrete_function{f};
        }
    };

    template <typename Func>
    struct PIXELPIPES_API func_filter
    {
        typedef Func type;
    };
    template <typename Result, typename... Args>
    struct func_filter<Result(Args...)>
    {
        typedef Result (*type)(Args...);
    };

    template <typename signature>
    class Function;

    template <typename Result, typename... Args>
    class PIXELPIPES_API Function<Result(Args...)>
    {
        abstract_function<Result, Args...> *f;

    public:
        Function()
            : f(nullptr)
        {
        }
        template <typename Func>
        Function(const Func &x)
            : f(new concrete_function<typename func_filter<Func>::type, Result, Args...>(x))
        {
        }
        Function(const Function &rhs)
            : f(rhs.f ? rhs.f->clone() : nullptr)
        {
        }
        Function &operator=(const Function &rhs)
        {
            if ((&rhs != this) && (rhs.f))
            {
                auto *temp = rhs.f->clone();
                delete f;
                f = temp;
            }
            return *this;
        }
        template <typename Func>
        Function &operator=(const Func &x)
        {
            auto *temp = new concrete_function<typename func_filter<Func>::type, Result, Args...>(x);
            delete f;
            f = temp;
            return *this;
        }
        Result operator()(Args... args)
        {
            if (f)
                return (*f)(args...);
            else
                throw BaseException("Illegal function pointer");
        }
        ~Function()
        {
            delete f;
        }
        operator bool() const
        {
            return (f) ? true : false;
        }
    };

    template <typename... Args>
    class PIXELPIPES_API Function<void(Args...)>
    {
        abstract_function<void, Args...> *f;

    public:
        Function()
            : f(nullptr)
        {
        }
        template <typename Func>
        Function(const Func &x)
            : f(new concrete_function<typename func_filter<Func>::type, void, Args...>(x))
        {
        }
        Function(const Function &rhs)
            : f(rhs.f ? rhs.f->clone() : nullptr)
        {
        }
        Function &operator=(const Function &rhs)
        {
            if ((&rhs != this) && (rhs.f))
            {
                auto *temp = rhs.f->clone();
                delete f;
                f = temp;
            }
            return *this;
        }
        template <typename Func>
        Function &operator=(const Func &x)
        {
            auto *temp = new concrete_function<typename func_filter<Func>::type, void, Args...>(x);
            delete f;
            f = temp;
            return *this;
        }
        void operator()(Args... args)
        {
            if (f)
                (*f)(args...);
            else
                throw BaseException("Illegal function pointer");
        }
        ~Function()
        {
            delete f;
        }
        operator bool() const
        {
            return (f) ? true : false;
        }
    };

    template <typename T, typename D = std::default_delete<T>>
    class PIXELPIPES_API Implementation
    {
    protected:
        //[[no_unique_address]]
        D del;
        T *raw;

        void reset()
        {
            if (raw != nullptr)
            {
                del(raw);
                raw = nullptr;
            }
        }

    public:
        using element_type = T;
        using deleter_type = D;

        Implementation(const Implementation<T> &other) : raw(new T(*other.raw))
        {
        }

        Implementation(Implementation<T> &&other)
        {
            raw = std::move(other.raw);
            other.raw = nullptr;
        }

        template <typename... Args>
        Implementation(Args... args) : raw(new T(std::forward<Args>(args)...))
        {
        }

        ~Implementation() noexcept //(std::is_nothrow_destructible<T>::value)
        {
            reset();
        }

        Implementation &operator=(const Implementation<T> &other) noexcept //(std::is_nothrow_copy_assignable<T>::value)
        {
            if (this != &other)
            {
                *raw = *(other.raw);
            }
            return *this;
        }

        Implementation &operator=(Implementation<T> &&other) noexcept //(std::is_nothrow_move_assignable<T>::value)
        {
            if (this != &other)
            {
                reset();
                raw = other.raw;
                other.raw = nullptr;
            }
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
    class PIXELPIPES_API View
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = ptrdiff_t;
        using const_reference = const T &;
        using const_pointer = const T *;
        using const_iterator = const T *;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        View() = default;

        View(const T *ptr, size_t size)
            : _ptr(ptr), _size(size)
        {
        }

        // Span(const Span &source, size_t offset)

        View(const View &) = default;
        View(View &&) = default;

        View &operator=(const View &) = default;
        View &operator=(View &&) = default;

        const_pointer get() const noexcept { return _ptr; }

        template <typename C>
        View<C> reinterpret() const
        {
            VERIFY((this->_size * sizeof(T)) % sizeof(C) == 0, "Unaligned size");
            return View<C>(reinterpret_cast<const C *>(_ptr), (_size * sizeof(T)) / sizeof(C));
        }

        template <typename C>
        const C &at(size_type i) const
        {
            return *((C *)(_ptr + i));
        }

        inline const_reference at(size_type i) const
        {
            return *(_ptr + i);
        }

        const_pointer data() const noexcept
        {
            return _ptr;
        }

        inline const_reference operator[](size_type i) const
        {
            return at(i);
        }

        inline operator bool() const
        {
            return data() != nullptr && size() != 0;
        }

        const_reference front() const
        {
            return at(0);
        }

        const_reference back() const
        {
            return at(_size - 1);
        }

        const_iterator begin() const noexcept
        {
            return data();
        }

        const_iterator end() const noexcept
        {
            return data() + _size;
        }

        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator(begin());
        }

        inline bool empty() const noexcept
        {
            return _size == 0;
        }

        inline size_t size() const noexcept
        {
            return _size;
        }

        // slicing
        View slice(size_t off, size_t count = size_t(-1)) const
        {
            if (off > _size)
                return View(_ptr + _size, 0);
            auto newSize = _size - off;
            if (count > newSize)
                count = newSize;
            return View(_ptr + off, count);
        }

    protected:
        const T *_ptr = nullptr;
        size_t _size = 0;
    };


    template <typename T>
    class PIXELPIPES_API Span : public View<T>
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

        Span(T *ptr, size_t size) : View<T>(ptr, size)
        {
        }

        // Span(const Span &source, size_t offset)

        Span(const Span &) = default;
        Span(Span &&) = default;

        Span &operator=(const Span &) = default;
        Span &operator=(Span &&) = default;

        const_pointer get() const noexcept { return this->_ptr; }

        template <typename C>
        Span<C> reinterpret()
        {
            VERIFY((this->_size * sizeof(T)) % sizeof(C) == 0, "Unaligned size");
            return Span<C>(reinterpret_cast<C *>( (pointer) this->_ptr), (this->_size * sizeof(T)) / sizeof(C));
        }

        template <typename C>
        View<C> reinterpret() const
        {
            VERIFY((this->_size * sizeof(T)) % sizeof(C) == 0, "Unaligned size");
            return View<C>(reinterpret_cast<const C *>(this->_ptr), (this->_size * sizeof(T)) / sizeof(C));
        }

        template <typename C>
        C &at(size_type i) 
        {
            return *((C *)(this->_ptr + i));
        }

        template <typename C>
        C at(size_type i) const
        {
            return *((C *)(this->_ptr + i));
        }

        inline const_reference at(size_type i) const
        {
            return  *((pointer) this->_ptr + i);
        }

        inline reference at(size_type i)
        {
            return  *((pointer) this->_ptr + i);
        }

        pointer data() noexcept
        {
            return (pointer) this->_ptr;
        }

        const_pointer data() const noexcept 
        {
            return this->_ptr;
        }

        inline reference operator[](size_type i)
        {
            return at(i);
        }

        inline const_reference operator[](size_type i) const
        {
            return at(i);
        }

        reference front()
        {
            return at(0);
        }

        reference back()
        {
            return at(this->size() - 1);
        }

        iterator begin() noexcept
        {
            return data();
        }

        iterator end() noexcept
        {
            return data() + this->size();
        }

        reverse_iterator rbegin() noexcept
        {
            return reverse_iterator(end());
        }

        reverse_iterator rend() noexcept
        {
            return reverse_iterator(begin());
        }

        const_iterator begin() const noexcept
        {
            return data();
        }

        const_iterator end() const noexcept
        {
            return data() + this->size();
        }

        const_reverse_iterator rbegin() const noexcept
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator rend() const noexcept
        {
            return const_reverse_iterator(begin());
        }

        // slicing
        Span slice(size_t off, size_t count = size_t(-1))
        {
            if (off > this->_size)
                return Span((pointer)this->_ptr + this->_size, 0);
            auto newSize = this->_size - off;
            if (count > newSize)
                count = newSize;
            return Span((pointer)this->_ptr + off, count);
        }

    };

    template <typename T>
    View<T> make_view(const T *ptr, size_t size)
    {
        return View<T>(ptr, size);
    }

    template <typename T, size_t N>
    View<T> make_view(const T (&ar)[N])
    {
        return View<T>(ar, N);
    }

    template <typename Container>
    auto make_view(const Container &c, size_t offset = 0) -> View<typename Container::value_type>
    {
        static_assert(details::has_contiguous_memory<Container>::value, "Only contiguous containers");

        return View<typename Container::value_type>(std::data(c) + offset, c.size() - offset);
    }

    template <typename Container>
    auto make_span(Container &c, size_t offset = 0) -> Span<typename Container::value_type>
    {
        static_assert(details::has_contiguous_memory<Container>::value, "Only contiguous containers");

        return Span<typename Container::value_type>(std::data(c) + offset, c.size() - offset);
    }

    template <typename T>
    class PIXELPIPES_API Sequence : public Span<T>
    {

    public:
        static Sequence claim(T *ptr, size_t size)
        {
            auto s = Sequence<T>();
            s._ptr = ptr;
            s._size = size;
            return s;
        }

        static Sequence<T> repeat(size_t size, const T& value)
        {
            auto ptr = new T[size];
            std::fill_n(ptr, size, value);
            return claim(ptr, size);
        }


        Sequence() : Span<T>()
        {
        }

        Sequence(size_t size) : Span<T>()
        {
            this->_ptr = new T[size];
            this->_size = size;
        }

        Sequence(const T *ptr, size_t size) : Span<T>()
        {
            copy(ptr, size);
        }

        Sequence(std::initializer_list<T> c) : Span<T>()
        {
            auto data = new T[c.size()];
            this->_size = c.size();
            size_t i = 0;
            for (auto x = c.begin(); x != c.end(); x++, i++)
            {
                if constexpr (is_pointer_v<T>)
                {
                    data[i] = x->reborrow();
                }
                else
                {
                    data[i] = *x;
                }
            }
            this->_ptr = data;
        }

        template <typename Container>
        Sequence(const Container &c, size_t offset = 0) : Span<T>()
        {
            using E = typename Container::value_type;
            if constexpr (details::has_contiguous_memory<Container>::value && sizeof(E) == sizeof(T))
            {
                copy((T *)std::data(c) + offset, c.size() - offset);
            }
            else
            {
                auto data = new T[c.size() - offset];
                this->_size = c.size() - offset;
                size_t i = 0;
                for (auto x = c.begin() + offset; x != c.end(); x++, i++)
                {
                    if constexpr (is_pointer_v<T>)
                    {
                        data[i] = x->reborrow();
                    }
                    else
                    {
                        data[i] = *x;
                    }
                }
                this->_ptr = data;
            }
        }

        Sequence(const Sequence &s) : Sequence(s, 0)
        {
        }

        Sequence(Sequence &&s) : Span<T>((T*)s._ptr, s._size)
        {
            // Steal data
            s._ptr = nullptr;
        }

        template<typename U>
        Sequence<U> convert()
        {

            Span<U> v = this->template reinterpret<U>();
            auto s = Sequence<U>::claim(v.data(), v.size());
            this->_ptr = nullptr;
            this->_size = 0;
            return s;
        }

        Sequence &operator=(const Sequence &s)
        {
            copy(s._ptr, s._size);
            return *this;
        }

        Sequence &operator=(Sequence &&s)
        {

            cleanup();
            this->_ptr = s._ptr;
            this->_size = s._size;
            s._ptr = nullptr;
            return *this;
        }

        ~Sequence()
        {
            cleanup();
        }

    private:
        void cleanup()
        {
            if (this->_ptr)
            {
                delete[] this->_ptr;
            }
            this->_ptr = nullptr;
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
                if constexpr (is_pointer_v<T>)
                {
                    for (size_t i = 0; i < size; i++)
                    {
                        data[i] = ptr[i].reborrow();
                    }
                }
                else
                {
                    std::copy(ptr, ptr + size, data);
                }
            }

            cleanup();

            this->_ptr = data;
            this->_size = size;
        }
    };

    template < class T >
    std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
    {
        os << "[";
        for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
        {
            os << " " << *ii;
        }
        os << "]";
        return os;
    }

    template < class T >
    std::ostream& operator<<(std::ostream& os, const Span<T>& s)
    {
        os << " [ ";
        for (auto t = s.begin(); t != s.end(); t++)
            os << *t << ", ";
        os << " ] ";
        return os;
    }

    class Formatter
    {
    public:
        PIXELPIPES_API Formatter() {}
        PIXELPIPES_API ~Formatter() {}

        template <typename Type>
        Formatter &operator<<(const Type &value)
        {
            stream << value;
            return *this;
        }

        PIXELPIPES_API std::string str() const { return stream.str(); }
        PIXELPIPES_API operator std::string() const { return stream.str(); }

        // PIXELPIPES_API const char *c_str() const { return stream.str().c_str(); }

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

    class PIXELPIPES_API RandomGenerator
    {
        typedef uint32_t (*Function)(uint32_t state);

        Function f_;

        uint32_t seed_;

    public:
        using result_type = uint32_t;

        RandomGenerator(Function _f, uint32_t _seed = 1) : f_(_f), seed_(_seed) {}

        uint32_t operator()(void)
        {

            seed_ = f_(seed_);
            return seed_;
        }

        static constexpr uint32_t min()
        {
            return std::numeric_limits<uint32_t>::min();
        }

        static constexpr uint32_t max()
        {
            return std::numeric_limits<uint32_t>::max();
        }
    };

}
