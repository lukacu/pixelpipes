#pragma once

#include <pixelpipes/type.hpp>

namespace pixelpipes
{

    class Token;
    typedef std::shared_ptr<Token> SharedToken;

    template <typename T = Token>
    std::shared_ptr<T> empty()
    {
        return std::shared_ptr<T>();
    }

    class PIXELPIPES_API Token
    {
    public:
        virtual TypeIdentifier type_id() const = 0;

        virtual Type type() const;

        virtual void describe(std::ostream &os) const = 0;

        std::string describe() const;

        friend std::ostream &operator<<(std::ostream &os, const Token &v);

        friend std::ostream &operator<<(std::ostream &os, const SharedToken &v);
    };

    template <typename T>
    T extract(SharedToken)
    {
        static_assert(detail::always_false<T>, "Conversion not supported");
    }

    template <typename T>
    SharedToken wrap(T value)
    {
        static_assert(detail::always_false<T>, "Conversion not supported");
        return SharedToken();
    }

    template <>
    inline SharedToken extract(const SharedToken v)
    {
        return v;
    }

    template <>
    inline SharedToken wrap(const SharedToken v)
    {
        return v;
    }

    template <typename T>
    class PIXELPIPES_API ContainerToken : public Token
    {
    public:
        ContainerToken(T value) : value(value)
        {
            get();
        };

        ~ContainerToken() = default;

        virtual TypeIdentifier type_id() const { return GetTypeIdentifier<T>(); };

        T get() const { return value; };

        inline static bool is(SharedToken v)
        {
            return (v->type_id() == GetTypeIdentifier<T>());
        }

        static T get_value(const SharedToken v)
        {

            return extract<T>(v);
        }

        virtual void describe(std::ostream &os) const
        {
            os << "[Container for " << type_name(type_id()) << "]";
        }

    protected:
        T value;
    };

    template <typename T>
    class PIXELPIPES_API ScalarToken : public ContainerToken<T>
    {
    public:
        ScalarToken(T value) : ContainerToken<T>(value){};

        ~ScalarToken() = default;

        virtual void describe(std::ostream &os) const
        {
            os << "[Scalar " << type_name(this->type_id()) << ", value: " << this->value << "]";
        }
    };

/*
    class PIXELPIPES_API StringToken : public Token
    {
    public:
        StringToken(T value) : value(value)
        {
            get();
        };

        ~StringToken() = default;

        virtual TypeIdentifier type_id() const { return GetTypeIdentifier<char *>(); };

        T get() const { return value; };

        inline static bool is(SharedToken v)
        {
            return (v->type_id() == GetTypeIdentifier<T>());
        }

        static T get_value(const SharedToken v)
        {

            return extract<T>(v);
        }

        virtual void describe(std::ostream &os) const
        {
            os << "[String token " << type_name(type_id()) << "]";
        }

    protected:
        T value;
    };
*/

    #define TokenIdentifier GetTypeIdentifier<SharedToken>()

    #define IntegerIdentifier GetTypeIdentifier<int>()
    #define FloatIdentifier GetTypeIdentifier<float>()
    #define BooleanIdentifier GetTypeIdentifier<bool>()
    #define StringIdentifier GetTypeIdentifier<std::string>()

    typedef ScalarToken<int> Integer;
    typedef ScalarToken<float> Float;
    typedef ScalarToken<bool> Boolean;
    typedef ScalarToken<std::string> String;

// TODO: validate
#define PIXELPIPES_CONVERT_ENUM(E)                                                                   \
    template <>                                                                                      \
    inline E extract(const SharedToken v)                                                         \
    {                                                                                                \
        VERIFY((bool)v, "Uninitialized token");                                                   \
        if (v->type_id() != IntegerIdentifier)                                                                \
            throw TypeException("Unexpected token type: expected int, got " + v->describe()); \
        return (E)std::static_pointer_cast<Integer>(v)->get();                                       \
    }                                                                                                \
    template <>                                                                                      \
    inline SharedToken wrap(const E v)                                                            \
    {                                                                                                \
        return std::make_shared<Integer>((int)v);                                                    \
    }

    template <>
    inline int extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type_id() != IntegerIdentifier)
            throw TypeException("Unexpected token type: expected int, got " + v->describe());

        return std::static_pointer_cast<Integer>(v)->get();
    }

    template <>
    inline SharedToken wrap(const int v)
    {
        return std::make_shared<Integer>(v);
    }

    template <>
    inline bool extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type_id() == FloatIdentifier)
            return std::static_pointer_cast<ContainerToken<float>>(v)->get() != 0;

        if (v->type_id() == IntegerIdentifier)
            return std::static_pointer_cast<ContainerToken<int>>(v)->get() != 0;

        if (v->type_id() == BooleanIdentifier)
            return std::static_pointer_cast<ContainerToken<bool>>(v)->get();

        throw TypeException("Unexpected token type: expected bool, got " + v->describe());
    }

    template <>
    inline SharedToken wrap(const bool v)
    {
        return std::make_shared<Boolean>(v);
    }

    template <>
    inline float extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type_id() == IntegerIdentifier)
            return (float)std::static_pointer_cast<ContainerToken<int>>(v)->get();

        if (v->type_id() == BooleanIdentifier)
            return (float)std::static_pointer_cast<ContainerToken<bool>>(v)->get();

        if (v->type_id() != FloatIdentifier)
            throw TypeException("Unexpected token type: expected float, got " + v->describe());

        return std::static_pointer_cast<Float>(v)->get();
    }

    template <>
    inline SharedToken wrap(const float v)
    {
        return std::make_shared<Float>(v);
    }

    template <>
    inline std::string extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type_id() != StringIdentifier)
            throw TypeException("Unexpected token type: expected string, got " + v->describe());

        return std::static_pointer_cast<String>(v)->get();
    }

    template <>
    inline SharedToken wrap(const std::string v)
    {
        return std::make_shared<String>(v);
    }


    class PIXELPIPES_API Container: public Token
    {
    public:
        ~Container() = default;

        virtual TypeIdentifier type_id() const = 0;

        virtual TypeIdentifier element_type_id() const = 0;

    };


    class PIXELPIPES_API List : public Container
    {
    public:
        ~List() = default;

        virtual size_t size() const = 0;

        virtual TypeIdentifier type_id() const;

        virtual Type type() const;

        virtual TypeIdentifier element_type_id() const = 0;

        virtual SharedToken get(size_t index) const = 0;

        inline static bool is(SharedToken v)
        {
            return ((bool)v && ((v->type_id() & TensorIdentifierMask) != 0));
        }

        inline static bool is_list(SharedToken v)
        {
            return ((bool)v && List::is(v));
        }

        inline static bool is_list(SharedToken v, TypeIdentifier type)
        {
            return ((bool)v && List::is(v)) && std::static_pointer_cast<List>(v)->element_type_id() == type;
        }

        inline static std::shared_ptr<List> get_list(SharedToken v)
        {
            VERIFY(is_list(v), "Not a list");
            return std::static_pointer_cast<List>(v);
        }

        inline static std::shared_ptr<List> get_list(SharedToken v, TypeIdentifier type)
        {
            VERIFY(is_list(v, type), "Not a list");
            return std::static_pointer_cast<List>(v);
        }

        inline static size_t length(SharedToken v)
        {
            VERIFY((bool)v, "Uninitialized token");

            if (!List::is_list(v))
                throw TypeException("Not a list");
            return std::static_pointer_cast<List>(v)->size();
        }

        static std::shared_ptr<List> cast(SharedToken v)
        {
            VERIFY((bool)v, "Uninitialized token");

            if (!List::is_list(v))
                throw TypeException("Not a list");
            return std::static_pointer_cast<List>(v);
        }

        // Slow default template version
        template <class C>
        const std::vector<C> elements() const
        {

            std::vector<C> result;

            for (size_t i = 0; i < size(); i++)
            {
                result.push_back(extract<C>(get(i)));
            }

            return result;
        }

        virtual void describe(std::ostream &os) const;
    };

    typedef std::shared_ptr<List> SharedList;

    template <typename C>
    class PIXELPIPES_API ContainerList : public List
    {
    public:
        ContainerList(){};
        ContainerList(Span<C> list) : list(list){};
        ~ContainerList() = default;

        inline static bool is_list(SharedToken v)
        {
            return List::is_list(v, GetTypeIdentifier<C>());
        }

        virtual size_t size() const { return list.size(); }

        virtual TypeIdentifier element_type_id() const { return GetTypeIdentifier<C>(); };

        virtual SharedToken get(size_t index) const { return std::make_shared<ScalarToken<C>>(list[index]); }

        template <class T>
        const std::vector<T> elements() const
        {
            if constexpr (std::is_same<T, C>::value) {
                return std::vector<T>(list);
            } else {
                return List::elements<T>();
            }
        }

    private:

        Sequence<C> list;

    };

    typedef ContainerList<float> FloatList;
    typedef ContainerList<int> IntegerList;
    typedef ContainerList<bool> BooleanList;
    typedef ContainerList<std::string> StringList;

    #define FloatListIdentifier GetTypeIdentifier<Span<float>>()
    #define IntegerListIdentifier GetTypeIdentifier<Span<int>>()
    #define BooleanListIdentifier GetTypeIdentifier<Span<bool>>()
    #define StringListIdentifier GetTypeIdentifier<Span<std::string>>()

#define _LIST_MAKE_EXTRACTORS(T)                      \
    template <>                                           \
    inline std::vector<T> extract(const SharedToken v) \
    {                                                     \
        VERIFY((bool)v, "Uninitialized token");        \
        VERIFY(List::is_list(v), "Not a list");           \
        auto list = std::static_pointer_cast<List>(v);    \
        return list->elements<T>();                       \
    }                                                     \
    template <>                                           \
    inline Sequence<T> extract(const SharedToken v) \
    {                                                     \
        VERIFY((bool)v, "Uninitialized token");        \
        VERIFY(List::is_list(v), "Not a list");           \
        auto list = std::static_pointer_cast<List>(v);    \
        return Sequence<T>(list->elements<T>());          \
    }   

    template <>                                           
    inline SharedToken wrap(const std::vector<int> v)    
    {                                                     
        return std::make_shared<ContainerList<int>>(make_span(v));     
    }

    template <>                                           
    inline SharedToken wrap(const std::vector<float> v)    
    {                                                     
        return std::make_shared<ContainerList<float>>(make_span(v));     
    }

    template <>                                           
    inline SharedToken wrap(const std::vector<std::string> v)    
    {                                                     
        return std::make_shared<ContainerList<std::string>>(make_span(v));     
    }

    template <>                                           
    inline SharedToken wrap(const std::vector<bool> v)    
    {                                                     
        Sequence<bool> s(v);
        return std::make_shared<ContainerList<bool>>(make_span(s));     
    }

    _LIST_MAKE_EXTRACTORS(int)
    _LIST_MAKE_EXTRACTORS(bool)
    _LIST_MAKE_EXTRACTORS(float)
    _LIST_MAKE_EXTRACTORS(std::string)

    /*
                auto table = std::static_pointer_cast<Table<detail::inner_type_t<T>>>(v);
                return table->elements();


    */

    /*
    template<typename T>
    SharedToken wrap(const T v) {
        if constexpr (is_vector<typename T::value_type>::value) {
            return std::make_shared<Table<detail::inner_type_t<T>>>(v);
        } else {
            return std::make_shared<ContainerList>(v);
        }
    }*/

}