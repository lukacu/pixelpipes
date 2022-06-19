#pragma once

#include <pixelpipes/type.hpp>

namespace pixelpipes
{
    class Token;
    typedef Pointer<Token> TokenReference;

    template <typename T = Token>
    Pointer<T> empty()
    {
        return Pointer<T>();
    }

    class PIXELPIPES_API Token : public virtual details::RTTI
    {
        PIXELPIPES_RTTI(Token)
    public:
        virtual Shape shape() const;

        virtual void describe(std::ostream &os) const = 0;

        std::string describe() const;
    };

    inline std::ostream& operator<<(std::ostream& os, const TokenReference& token)
    {
        if (!token) {
            os << "[Empty token]";
        } else {
            os << token->describe();
        }
        return os;
    }

    template <typename T>
    class PIXELPIPES_API ContainerToken : public Token
    {
        PIXELPIPES_RTTI(ContainerToken<T>, Token)
    public:
        ContainerToken(T value) : value(value)
        {
        };

        ~ContainerToken() = default;

        virtual Shape shape() const
        {
            return Shape(GetTypeIdentifier<T>());
        }

        virtual TypeIdentifier container_type() const
        {

            return GetTypeIdentifier<ContainerToken<T>>();
        }

        inline T get() const { return value; };

        virtual void describe(std::ostream &os) const
        {
            os << "[Container for " << details::TypeInfo<T>::Name() << "]";
        }

    protected:
        T value;
    };

    template <typename T>
    T extract(const TokenReference& v)
    {
        UNUSED(v);
        static_assert(details::always_false<T>, "Conversion not supported");
    }

    template <typename T>
    inline TokenReference wrap(T v)
    {
        UNUSED(v);
        static_assert(details::always_false<T>, "Conversion not supported");
        return empty();
    }

    template <>
    inline TokenReference extract(const TokenReference& v)
    {
        return v.reborrow();
    }

    template <>
    inline TokenReference wrap(const TokenReference v)
    {
        return v.reborrow();
    }

    template <typename T>
    class PIXELPIPES_API ScalarToken : public ContainerToken<T>
    {
        PIXELPIPES_RTTI(ScalarToken<T>, ContainerToken<T>)
    public:
        ScalarToken(T value) : ContainerToken<T>(value) { static_assert(std::is_fundamental_v<T>, "Only primitive types allowed"); };

        ~ScalarToken() = default;

        virtual void describe(std::ostream &os) const
        {
            os << "[Scalar " << details::TypeInfo<T>::Name() << ", value: " << this->get() << "]";
        }
    };

#define TokenIdentifier GetTypeIdentifier<TokenReference>()

#define IntegerIdentifier GetTypeIdentifier<int>()
#define ShortIdentifier GetTypeIdentifier<ushort>()
#define FloatIdentifier GetTypeIdentifier<float>()
#define BooleanIdentifier GetTypeIdentifier<bool>()
#define CharIdentifier GetTypeIdentifier<uchar>()

    typedef ScalarToken<int> IntegerScalar;
    typedef ScalarToken<float> FloatScalar;
    typedef ScalarToken<bool> BooleanScalar;
    typedef ScalarToken<uchar> CharScalar;
    typedef ScalarToken<ushort> ShortScalar;

#define _IS_SCALAR_TOKEN(TOKEN, INNER) (((TOKEN)->is<ContainerToken<INNER>>()))

// TODO: validate
#define PIXELPIPES_CONVERT_ENUM(E)            \
    template <>                               \
    inline E extract(const TokenReference &v)    \
    {                                         \
        int raw = extract<int>(v);            \
        return (E)raw;                        \
    }                                         \
    template <>                               \
    inline TokenReference wrap(E v)       \
    {                                         \
        return create<IntegerScalar>((int)v); \
    }

    template <>
    inline int extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (_IS_SCALAR_TOKEN(v, int))
            return v->cast<ContainerToken<int>>()->get();

        if (_IS_SCALAR_TOKEN(v, short))
            return v->cast<ContainerToken<short>>()->get();

        if (_IS_SCALAR_TOKEN(v, bool))
            return v->cast<ContainerToken<bool>>()->get() ? 1 : 0;

        throw TypeException("Unexpected token type: expected int, got " + v->describe());
    }

    template <>
    inline TokenReference wrap(const int v)
    {
        return create<IntegerScalar>(v);
    }


    template <>
    inline short extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (_IS_SCALAR_TOKEN(v, short))
            return v->cast<ContainerToken<short>>()->get();

        if (_IS_SCALAR_TOKEN(v, bool))
            return v->cast<ContainerToken<bool>>()->get() ? 1 : 0;

        throw TypeException("Unexpected token type: expected short, got " + v->describe());
    }

    template <>
    inline TokenReference wrap(const short v)
    {
        return create<ShortScalar>(v);
    }

    template <>
    inline bool extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (_IS_SCALAR_TOKEN(v, bool))
            return v->cast<ContainerToken<bool>>()->get();

        if (_IS_SCALAR_TOKEN(v, float))
            return v->cast<ScalarToken<float>>()->get() != 0;

        if (_IS_SCALAR_TOKEN(v, int))
            return v->cast<ContainerToken<int>>()->get() != 0;

        throw TypeException("Unexpected token type: expected bool, got " + v->describe());
    }

    template <>
    inline TokenReference wrap(const bool v)
    {
        return create<BooleanScalar>(v);
    }

    template <>
    inline char extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (_IS_SCALAR_TOKEN(v, char))
            return v->cast<ContainerToken<char>>()->get();

        throw TypeException("Unexpected token type: expected char, got " + v->describe());
    }

    template <>
    inline TokenReference wrap(const char v)
    {
        return create<ScalarToken<char>>(v);
    }

    template <>
    inline float extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (_IS_SCALAR_TOKEN(v, float))
            return (float)v->cast<ContainerToken<float>>()->get();

        if (_IS_SCALAR_TOKEN(v, int))
            return (float)v->cast<ContainerToken<int>>()->get();

        if (_IS_SCALAR_TOKEN(v, short))
            return (float)v->cast<ContainerToken<short>>()->get();

        if (_IS_SCALAR_TOKEN(v, bool))
            return (float)v->cast<ContainerToken<bool>>()->get();

        throw TypeException("Unexpected token type: expected float, got " + v->describe());
    }

    template <>
    inline TokenReference wrap(const float v)
    {
        return create<FloatScalar>(v);
    }

    class PIXELPIPES_API List : public virtual Token
    {
        PIXELPIPES_RTTI(List, Token)
    public:
        virtual ~List() = default;

        virtual size_t length() const = 0;

        virtual Shape shape() const = 0;

        virtual TokenReference get(size_t index) const = 0;

        // Slow default template version
        template <class C>
        const Sequence<C> elements() const
        {

            std::vector<C> result;
            result.reserve(length());

            for (size_t i = 0; i < length(); i++)
            {
                result.push_back(extract<C>(get(i)));
            }

            return Sequence<C>(result);
        }

        virtual void describe(std::ostream &os) const;
    };

    typedef Pointer<List> ListReference;

    template <>
    inline ListReference extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized token");
        VERIFY(v->is<List>(), "Not a list");
        return cast<List>(v);
    }

    class PIXELPIPES_API GenericList : public List
    {
        PIXELPIPES_RTTI(GenericList, List)
    public:
        GenericList(const std::initializer_list<TokenReference> &elements);
        GenericList(const View<TokenReference> &elements);

        virtual ~GenericList();

        virtual Shape shape() const;

        virtual size_t length() const;
        virtual TokenReference get(size_t index) const;

    private:
        Sequence<TokenReference> _elements;
        Shape _shape;
    };

}