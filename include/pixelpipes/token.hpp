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

    class PIXELPIPES_API Token : public virtual details::RTTI, public enable_pointer_from_this<Token>
    {
        PIXELPIPES_RTTI(Token)
    public:
        ~Token();

        virtual Shape shape() const;

        virtual void describe(std::ostream &os) const = 0;

        std::string describe() const;
    };

    class PIXELPIPES_API Placeholder : public Token
    /**
    Placeholder token is used to represent a token with unknown value, but known shape. It is used in inference phase.
    */
    {
        PIXELPIPES_RTTI(Placeholder, Token)
    public:
        /** Construct a placeholder token with given shape.
         */
        Placeholder(const Shape &shape = AnythingShape());

        Placeholder(const Placeholder &that);

        Placeholder(const Type &type, const Sizes &shape);

        Placeholder(const Type &type);

        /** Destruct the placeholder token.
         */
        virtual ~Placeholder();

        virtual void describe(std::ostream &os) const override;

        virtual Shape shape() const override;

        TokenReference dummy() const;

    private:
        Shape _shape;
    };

    inline std::ostream &operator<<(std::ostream &os, const TokenReference &token)
    {
        if (!token)
        {
            os << "[Empty token]";
        }
        else
        {
            os << token->describe();
        }
        return os;
    }

    template <typename T>
    class PIXELPIPES_API ContainerToken : public Token
    {
        PIXELPIPES_RTTI(ContainerToken<T>, Token)
    public:
        ContainerToken(T value) : value(value){};

        ~ContainerToken() = default;

        virtual Shape shape() const override
        {
            return Shape(GetType<T>());
        }

        virtual Type container_type() const
        {

            return GetType<ContainerToken<T>>();
        }

        inline T get() const { return value; };

        virtual void describe(std::ostream &os) const override
        {
            os << "[Container for " << details::TypeInfo<T>::Name() << "]";
        }

    protected:
        T value;
    };

    template <typename T>
    T extract(const TokenReference &v)
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
    inline TokenReference extract(const TokenReference &v)
    {
        return v.reborrow();
    }

    template <>
    inline TokenReference wrap(const TokenReference v)
    {
        return v.reborrow();
    }

#define TokenIdentifier GetType<TokenReference>()

typedef View<TokenReference> TokenList;

#define _IS_CONTAINER_TOKEN(TOKEN, INNER) (((TOKEN)->is<ContainerToken<INNER>>()))

#define _IS_PLACEHOLDER(TOKEN) (((TOKEN)->is<Placeholder>()))

// TODO: validate
#define PIXELPIPES_CONVERT_ENUM(E)            \
    template <>                               \
    inline E extract(const TokenReference &v) \
    {                                         \
        int raw = extract<int>(v);            \
        return (E)raw;                        \
    }                                         \
    template <>                               \
    inline TokenReference wrap(E v)           \
    {                                         \
        return create<IntegerScalar>((int)v); \
    }

    class PIXELPIPES_API List : public virtual Token
    {
        PIXELPIPES_RTTI(List, Token)
    public:
        virtual ~List() = default;

        virtual size_t length() const = 0;

        virtual Shape shape() const override = 0;

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

        virtual void describe(std::ostream &os) const override;
    };

    typedef Pointer<List> ListReference;

    template <>
    inline ListReference extract(const TokenReference &v)
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

        virtual Shape shape() const override;

        virtual size_t length() const override;
        virtual TokenReference get(size_t index) const override;

    private:
        Sequence<TokenReference> _elements;
        Shape _shape;
    };

}
