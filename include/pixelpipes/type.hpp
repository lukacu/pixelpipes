#pragma once
#include <memory>
#include <vector>
#include <array>
#include <type_traits>
#include <string_view>
#include <map>
#include <any>
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

    using uchar = unsigned char;
    using ushort = unsigned short;

    namespace details
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

    using Type = pixelpipes::details::Type;

    /**
     * The type identifier for any type.
    */
    constexpr static Type AnyType = 0;

    /**
     * The function that returns the type id.
     *
     * It uses the pointer to the static data member of a class template to achieve this.
     * Altough the value is not predictible, it's stable (I hope).
     */
    template <typename T>
    constexpr Type PIXELPIPES_TYPE_API GetType() noexcept
    {
        return details::TypeInfo<T>::Id();
    }

#define VIEWCHARS(S) std::string(S).c_str()

    class PIXELPIPES_API TypeException : public BaseException
    {
    public:
        TypeException(std::string reason);
        TypeException(const TypeException& e) = default;
    };

#define IntegerType GetType<int>()
#define ShortType GetType<short>()
#define FloatType GetType<float>()
#define BooleanType GetType<bool>()
#define CharType GetType<char>()
#define UnsignedShortType GetType<ushort>()

	PIXELPIPES_API const char* type_name(Type t);

    // Special valu - size cannot be determined
    constexpr size_t unknown = ~0L;

    struct PIXELPIPES_API Size
    {
        size_t data;

        constexpr Size() : data(unknown){};
        constexpr Size(int x) : data((size_t)x){};
        constexpr Size(size_t x) : data(x){};
        Size(const Size &) = default;
        Size(Size &&s) = default;

        Size operator+(const Size &other) const;
        Size operator-(const Size &other) const;
        Size operator*(const Size &other) const;
        Size operator/(const Size &other) const;
        Size operator%(const Size &other) const;
        bool operator==(const Size &other) const;

        template <typename T>
        Size operator+(const T &other) const
        {
            if (data == unknown)
                return unknown;
            return (size_t) ((size_t)data + other);
        }

        template <typename T>
        Size operator-(const T &other) const
        {
            if (data == unknown)
                return unknown;
            return (size_t) ((size_t)data - other);
        }

        template <typename T>
        Size operator*(const T &other) const
        {
            if (data == unknown)
                return unknown;
            return (size_t) ((size_t)data * other);
        }

        template <typename T>
        Size operator/(const T &other) const
        {
            if (data == unknown)
                return unknown;
            return (size_t) ((size_t)data / other);
        }

        template <typename T>
        Size operator%(const T &other) const
        {
            if (data == unknown)
                return unknown;
            return (size_t) ((size_t)data % other);
        }

        template <typename T>
        bool operator==(const T &other) const
        {
            return data == (size_t)other;
        }

        inline bool operator==(int v) const
        {
            return (data != unknown) && data == (size_t)v;
        }

        inline bool operator!=(const Size &other) const {
            return other.data != data;

        }

        inline bool operator==(size_t v) const
        {
            return (data != unknown) && data == v;
        }

        inline bool operator!=(int v) const
        {
            return data != (size_t)v;
        }

        inline bool operator!=(size_t v) const
        {
            return data != v;
        }

        operator bool() const;

        inline operator size_t() const 
        {
            return data;
        }

        inline operator int() const 
        {
            return (int)data;
        }


        Size operator&(const Size &other) const;

        Size &operator=(const Size &s) = default;
        Size &operator=(Size &&s) = default;
    };

    typedef Sequence<size_t> SizeSequence;
    typedef Span<size_t> SizeSpan;
    typedef View<size_t> Sizes;

    inline Size aggregate(const Sizes& sizes) 
    {
        size_t result = 1;
        for (auto s : sizes)
        {
            if (s == unknown)
                return unknown;
            result *= s;
        }
        return result;
    }

    // TODO: Add support for larger ranks, make it so that anything above MAX is a dynamicly allocated array
    #define SHAPE_RANK_MAX 6

    class PIXELPIPES_API Shape
    {
    public:
        using value_type = Size;
        using size_type = size_t;

        Shape();

        Shape(const Shape &) = default;
        Shape(Shape &&s) = default;

        Shape(Type element, const std::initializer_list<Size>& shape);
        Shape(Type element, const View<Size>& shape);
        Shape(Type element, const Sizes& shape);

        Shape(Type element);

        virtual ~Shape() = default;

        Type element() const;

        value_type operator[](size_t index) const;

        size_t rank() const;

        size_t size() const;

        bool is_fixed() const;

        bool is_scalar() const;

        bool is_anything() const;

        Shape cast(Type t) const;

        Shape push(Size s) const;

        Shape pop() const;

        inline Span<value_type>::const_iterator begin() const noexcept
        {
            return _shape.begin();
        }

        inline Span<value_type>::const_iterator end() const noexcept
        {
            return _shape.end();
        }

        Shape common(Shape &other) const;

        Shape operator&(const Shape &other) const;
        Shape operator|(const Shape &other) const;
        bool operator==(const Shape &other) const;
        Shape &operator=(const Shape &s) = default;
        Shape &operator=(Shape &&s) = default;

        inline Sizes sizes() const
        {
            SizeSequence _s(rank());
            for (size_t i = 0; i < rank(); i++)
            {
                _s[i] = _shape[i];
            }

            return _s;
        }

    private:
    
        Type _element;
        Sequence<Size> _shape;
    };

    // Convert a shape to a string
    template <typename T>
    inline std::string to_string(const T& s)
    {
        std::stringstream ss;
        ss << s;
        return ss.str();
    }

    inline std::string to_string(const Size& t)
    {
        if (!(t))
            return "unknown";
        else
            return std::to_string(static_cast<size_t>(t));
    }

    inline std::string to_string(const Type& t)
    {
        return type_name(t);
    }

    inline std::ostream& operator<<(std::ostream& os, const Sizes& s)
    {
        auto t = s.begin();
        if (t != s.end()) {
            os << to_string(*t);
            for (t++; t != s.end(); t++)
                os << " x " << to_string(*t);
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const Shape& s)
    {
        if (s.is_anything()) {
            os << "[ anything ]";
            return os;
        }

        os << " [ " << to_string(s.element());
        auto t = s.begin();
        if (t != s.end()) {
            os << " - " << to_string(*t);
            for (t++; t != s.end(); t++)
                os << " x " << to_string( *t);
        }
        os << " ] ";
        return os;
    }

    template <typename T, size_t... sizes>
    Shape make_shape()
    {

        std::vector<size_t> _t = {sizes...};
        std::vector<Size> _s;
        _s.reserve(_t.size());

        for (auto t : _t)
        {
            _s.push_back(Size(t));
        }

        return Shape(GetType<T>(), make_span(_s));
    }

    inline size_t type_size(Type t)
    {
        if (t == GetType<int>())
        {
            return sizeof(int);
        }
        else if (t == GetType<float>())
        {
            return sizeof(float);
        }
        else if (t == GetType<uchar>() || t == GetType<char>())
        {
            return sizeof(char);
        }
        else if (t == GetType<bool>())
        {
            return sizeof(bool);
        }
        else if (t == GetType<ushort>() || t == GetType<short>())
        {
            return sizeof(short);
        }

        return 0;
    }

    template <typename T>
    inline Shape ScalarType()
    {
        return Shape(GetType<T>());
    }

    template <typename T>
    inline Shape ListType(Size length)
    {
        return Shape(GetType<T>(), Sequence<Size>({length}));
    }

    inline Shape ListType(Type element, Size length)
    {
        return Shape(element, Sequence<Size>({length}));
    }

    inline Shape MatrixType(Type element, Size width, Size height)
    {
        return Shape(element, Sequence<Size>({width, height}));
    }

    inline Shape ImageType(Type element, Size width, Size height, Size channels)
    {
        return Shape(element, Sequence<Size>({width, height, channels}));
    }

    Shape PIXELPIPES_API AnythingShape();

    typedef std::map<std::string, int> EnumerationMap;

    EnumerationMap PIXELPIPES_API describe_enumeration(std::string &name);

    void PIXELPIPES_API register_enumeration(const std::string &name, EnumerationMap mapping);

    template <typename T>
    inline void register_enumeration(const std::string &name)
    {
        auto pairs = details::enum_entries<T>();
        EnumerationMap mapping;
        for (auto pair : pairs)
        {
            mapping.insert(EnumerationMap::value_type(pair.second, (int)pair.first));
        }
        register_enumeration(name, mapping);
    }

#define PIXELPIPES_REGISTER_ENUM(N, T) static AddModuleInitializer CONCAT(__enum_init_, __COUNTER__)([]() { register_enumeration<T>(N); })

}