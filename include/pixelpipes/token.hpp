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
    class ContainerToken : public Token
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
    class ScalarToken : public ContainerToken<T>
    {
    public:
        ScalarToken(T value) : ContainerToken<T>(value){};

        ~ScalarToken() = default;

        virtual void describe(std::ostream &os) const
        {
            os << "[Scalar " << type_name(this->type_id()) << ", value: " << this->value << "]";
        }
    };

    #define TokenType GetTypeIdentifier<SharedToken>()

    #define IntegerType GetTypeIdentifier<int>()
    #define FloatType GetTypeIdentifier<float>()
    #define BooleanType GetTypeIdentifier<bool>()
    #define StringType GetTypeIdentifier<std::string>()

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
        if (v->type_id() != IntegerType)                                                                \
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

        if (v->type_id() != IntegerType)
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

        if (v->type_id() == FloatType)
            return std::static_pointer_cast<ContainerToken<float>>(v)->get() != 0;

        if (v->type_id() == IntegerType)
            return std::static_pointer_cast<ContainerToken<int>>(v)->get() != 0;

        if (v->type_id() == BooleanType)
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

        if (v->type_id() == IntegerType)
            return (float)std::static_pointer_cast<ContainerToken<int>>(v)->get();

        if (v->type_id() == BooleanType)
            return (float)std::static_pointer_cast<ContainerToken<bool>>(v)->get();

        if (v->type_id() != FloatType)
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

        if (v->type_id() != StringType)
            throw TypeException("Unexpected token type: expected string, got " + v->describe());

        return std::static_pointer_cast<String>(v)->get();
    }

    template <>
    inline SharedToken wrap(const std::string v)
    {
        return std::make_shared<String>(v);
    }


    class Container: public Token
    {
    public:
        ~Container() = default;

        virtual TypeIdentifier type_id() const = 0;

        virtual TypeIdentifier element_type_id() const = 0;

    };


    class List : public Container
    {
    public:
        ~List() = default;

        virtual size_t size() const = 0;

        virtual TypeIdentifier type_id() const;

        virtual Type type() const;

        virtual TypeIdentifier element_type_id() const = 0;

        virtual SharedToken get(int index) const = 0;

        inline static bool is(SharedToken v)
        {
            return ((bool)v && ((v->type_id() & ListType) != 0));
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
    class ContainerList : public List
    {
    public:
        ContainerList(){};
        ContainerList(std::initializer_list<C> list) : list(list) {};
        ContainerList(const std::vector<C> list) : list(list){};
        ~ContainerList() = default;

        inline static bool is_list(SharedToken v)
        {
            return List::is_list(v, GetTypeIdentifier<C>());
        }

        virtual size_t size() const { return list.size(); }

        virtual TypeIdentifier element_type_id() const { return GetTypeIdentifier<C>(); };

        virtual SharedToken get(int index) const { return std::make_shared<ScalarToken<C>>(list[index]); }

        template <class T>
        const std::vector<T> elements() const
        {
            if (std::is_same<T, C>::value) {
                return std::vector<T>(list);
            } else {
                return List::elements<T>();
            }
        }

    private:
        std::vector<C> list;

    };

    typedef ContainerList<float> FloatList;
    typedef ContainerList<int> IntegerList;
    typedef ContainerList<bool> BooleanList;
    typedef ContainerList<std::string> StringList;

    #define FloatListType GetListIdentifier<float>()
    #define IntegerListType GetListIdentifier<int>()
    #define BooleanListType GetListIdentifier<bool>()
    #define StringListType GetListIdentifier<std::string>()

#define _LIST_GENERATE_CONVERTERS(T)                      \
    template <>                                           \
    inline std::vector<T> extract(const SharedToken v) \
    {                                                     \
        VERIFY((bool)v, "Uninitialized token");        \
        VERIFY(List::is_list(v), "Not a list");           \
        auto list = std::static_pointer_cast<List>(v);    \
        return list->elements<T>();                       \
    }                                                     \
    template <>                                           \
    inline SharedToken wrap(const std::vector<T> v)    \
    {                                                     \
        return std::make_shared<ContainerList<T>>(v);     \
    }

    _LIST_GENERATE_CONVERTERS(int);
    _LIST_GENERATE_CONVERTERS(bool);
    _LIST_GENERATE_CONVERTERS(float);
    _LIST_GENERATE_CONVERTERS(std::string);

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