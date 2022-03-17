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

namespace pixelpipes {

namespace detail {

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
            if (str[i] == 'i' && str[i+1] == 'n' && str[i+2] == 't')
            {
                if (format)
                {
                    format->leading_junk = i;
                    format->trailing_junk = sizeof(str)-i-3-1; // `3` is the length of "int", `1` is the space for the null terminator.
                }
                return true;
            }
        }
        return false;
    }

    inline static constexpr _raw_name_format format =
    []{
        static_assert(_get_raw_name_format(nullptr), "Unable to figure out how to generate type names on this compiler.");
        _raw_name_format format;
        _get_raw_name_format(&format);
        return format;
    }();

    template <typename T>
    [[nodiscard]] constexpr auto type_name() {
        return std::string_view{_type_name_raw<T>() + format.leading_junk, sizeof(_type_name_raw<T>()) - format.trailing_junk - format.leading_junk - 1};
    }

#else 

    constexpr std::string_view _undefined = "unsupported"; 

    template <typename T>
    [[nodiscard]] constexpr auto type_name() {
        
        return _undefined; //{_undefined, sizeof(undefined)-1};

    }
#endif

    template <typename T> struct inner_type { using type = T; };

    template<class T, class Alloc>
    struct inner_type<std::vector<T, Alloc>> { using type = typename inner_type<T>::type; };

    template<class T>
    using inner_type_t = typename inner_type<T>::type;

} 

template<typename T>
struct is_vector {
    static constexpr bool value = false;
};

template<template<typename...> typename C, typename U>
struct is_vector<C<U>> {
    static constexpr bool value = 
        std::is_same<C<U>, std::vector<U>>::value;
};

template<typename C>
struct is_string {
    static constexpr bool value = 
        std::is_same<C, std::string>::value;
};

/**
 * The type of a type id.
 */
typedef const void* TypeIdentifier;

/**
 * The function that returns the type id.
 * 
 * It uses the pointer to the static data member of a class template to achieve this.
 * Altough the value is not predictible, it's stable (I hope).
 */
template <typename T>
constexpr auto GetTypeIdentifier() noexcept -> TypeIdentifier {
    return &detail::TypeIdentifierToken<T>::id;
}

typedef std::string_view TypeName;

template<typename T> constexpr TypeName GetTypeName();

#define VIEWCHARS(S) std::string(S).c_str()

template <typename T, typename dummy = T>
struct Type {

    constexpr static TypeIdentifier identifier = GetTypeIdentifier<T>();
    constexpr static TypeName name = GetTypeName<T>();
    constexpr static bool scalar = true; 

};

template<typename T>
struct Type <T, typename std::enable_if<is_vector<T>::value, T >::type> {

    constexpr static TypeIdentifier identifier = GetTypeIdentifier<T>();
    constexpr static std::string_view name = GetTypeName<T>();
    constexpr static bool scalar = false; 

};

class Variable;
typedef std::shared_ptr<Variable> SharedVariable;

template<typename T = Variable>
std::shared_ptr<T> empty() {
    return std::shared_ptr<T>();
}

class PIXELPIPES_API VariableException : public BaseException {
public:
    VariableException(std::string reason);
};

inline void verify(bool condition, std::string reason = std::string("Assertion failed")) {

    if (!condition)
        throw VariableException(reason);

}

#define DEBUG_MODE 

#ifdef DEBUG_MODE
#define DEBUG(X) {std::cout << X << std::endl;}
#define VERIFY(C, M) verify((C), (M))
#else
#define DEBUG(X) 
#define VERIFY(C, M)
#endif

class PIXELPIPES_API Variable {
public:

    virtual TypeIdentifier type() const = 0;

    virtual void describe(std::ostream& os) const = 0;

    virtual bool is_scalar() const = 0;

    std::string describe() const;

    friend std::ostream& operator<<(std::ostream& os, const Variable& v);

    friend std::ostream& operator<<(std::ostream& os, const SharedVariable& v);

};

template<typename T>
T extract(SharedVariable) {
    static_assert(detail::always_false<T>, "Conversion not supported");
}

template<typename T>
SharedVariable wrap(T value) {
    static_assert(detail::always_false<T>, "Conversion not supported");
    return SharedVariable();
}

template<>
inline SharedVariable extract(const SharedVariable v) {
    return v;
}

template<>
inline SharedVariable wrap(const SharedVariable v) {
    return v;
}

template <typename T> 
class ContainerVariable : public Variable {
public:

    ContainerVariable(T value): value(value) {
        get();
    };

    ~ContainerVariable() = default;

    virtual TypeIdentifier type() const { return Type<T>::identifier; };

    T get() const { return value; };

    virtual bool is_scalar() const { return true; }

    inline static bool is(SharedVariable v) {
        return (v->type() == Type<T>::identifier);
    }

    static T get_value(const SharedVariable v) {

        return extract<T>(v);

    }

    virtual void describe(std::ostream& os) const {
        os << "[Container for " << Type<T>::name << "]";
    }



protected:

    T value;

};

template <typename T> 
class ScalarVariable : public ContainerVariable<T> {
public:

    ScalarVariable(T value): ContainerVariable<T>(value) {};

    ~ScalarVariable() = default;

    virtual void describe(std::ostream& os) const {
        os << "[Scalar " << Type<T>::name << ", value: " << this->value << "]";
    }

};

constexpr static TypeIdentifier VariableType = Type<SharedVariable>::identifier;

constexpr static TypeIdentifier IntegerType = Type<int>::identifier;
constexpr static TypeIdentifier FloatType = Type<float>::identifier;
constexpr static TypeIdentifier BooleanType = Type<bool>::identifier;
constexpr static TypeIdentifier StringType = Type<std::string>::identifier;

#define PIXELPIPES_TYPE_NAME(T, N) template <> constexpr std::string_view GetTypeName<T>() { return N; }

PIXELPIPES_TYPE_NAME(SharedVariable, "variable");
PIXELPIPES_TYPE_NAME(int, "integer");
PIXELPIPES_TYPE_NAME(float, "float");
PIXELPIPES_TYPE_NAME(bool, "boolean");
PIXELPIPES_TYPE_NAME(std::string, "string");

typedef ScalarVariable<int> Integer;
typedef ScalarVariable<float> Float;
typedef ScalarVariable<bool> Boolean;
typedef ScalarVariable<std::string> String;

// TODO: validate 
#define PIXELPIPES_CONVERT_ENUM(E) \
template<> inline E extract(const SharedVariable v) { \
    VERIFY((bool) v, "Uninitialized variable"); \
    if (v->type() != IntegerType) throw VariableException("Unexpected variable type: expected int, got " + v->describe()); \
    return (E) std::static_pointer_cast<Integer>(v)->get(); \
} \
template<> inline  SharedVariable wrap(const E v) { \
    return std::make_shared<Integer>((int) v); \
}

template<>
inline int extract(const SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() != IntegerType)
        throw VariableException("Unexpected variable type: expected int, got " + v->describe());

    return std::static_pointer_cast<Integer>(v)->get();
}

template<>
inline SharedVariable wrap(const int v) {
    return std::make_shared<Integer>(v);
}

template<>
inline bool extract(const SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() == FloatType)
        return std::static_pointer_cast<ContainerVariable<float>>(v)->get() != 0;

    if (v->type() == IntegerType)
        return std::static_pointer_cast<ContainerVariable<int>>(v)->get() != 0;

    if (v->type() == BooleanType)
        return std::static_pointer_cast<ContainerVariable<bool>>(v)->get();

    throw VariableException("Unexpected variable type: expected bool, got " + v->describe());

}

template<>
inline SharedVariable wrap(const bool v) {
    return std::make_shared<Boolean>(v);
}

template<>
inline float extract(const SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() == IntegerType)
        return (float) std::static_pointer_cast<ContainerVariable<int>>(v)->get();

    if (v->type() == BooleanType)
        return (float) std::static_pointer_cast<ContainerVariable<bool>>(v)->get();

    if (v->type() != FloatType)
        throw VariableException("Unexpected variable type: expected float, got " + v->describe());

    return std::static_pointer_cast<Float>(v)->get();
}

template<>
inline SharedVariable wrap(const float v) {
    return std::make_shared<Float>(v);
}

template<>
inline std::string extract(const SharedVariable v) {
    VERIFY((bool) v, "Uninitialized variable");

    if (v->type() != StringType)
        throw VariableException("Unexpected variable type: expected string, got " + v->describe());

    return std::static_pointer_cast<String>(v)->get();
}

template<>
inline SharedVariable wrap(const std::string v) {
    return std::make_shared<String>(v);
}

class List: public Variable {
public:

    ~List() = default;

    virtual size_t size() const = 0;

    virtual TypeIdentifier type() const;

    virtual TypeIdentifier element_type() const = 0;

    virtual SharedVariable get(int index) const = 0; 

    virtual bool is_scalar() const { return false; }

    inline static bool is(SharedVariable v) {
        return ((bool) v && v->type() == GetTypeIdentifier<List>());
    }

    inline static bool is_list(SharedVariable v) {
        return ((bool) v && List::is(v));
    }

    inline static bool is_list(SharedVariable v, TypeIdentifier type) {
        return ((bool) v && List::is(v)) && std::static_pointer_cast<List>(v)->element_type() == type;
    }

    inline static std::shared_ptr<List> get_list(SharedVariable v) {
        VERIFY(is_list(v), "Not a list");
        return std::static_pointer_cast<List>(v);
    }

    inline static std::shared_ptr<List> get_list(SharedVariable v, TypeIdentifier type) {
        VERIFY(is_list(v, type), "Not a list");
        return std::static_pointer_cast<List>(v);
    }

    inline static size_t length(SharedVariable v) {
        VERIFY((bool) v, "Uninitialized variable");

        if (!List::is_list(v)) throw VariableException("Not a list");
        return std::static_pointer_cast<List>(v)->size();
    }

    static std::shared_ptr<List> cast(SharedVariable v) {
        VERIFY((bool) v, "Uninitialized variable");

        if (!List::is_list(v)) throw VariableException("Not a list");
        return std::static_pointer_cast<List>(v);
    }

    // Slow default template version
    template<class C> const std::vector<C> elements() const {

        std::vector<C> result;

        for (size_t i = 0; i < size(); i++) {
            result.push_back(extract<C>(get(i)));
        }

        return result;

    }

    virtual void describe(std::ostream& os) const;


};

typedef std::shared_ptr<List> SharedList;

template<typename C>
class ContainerList: public List {
public:

    ContainerList() {};
    ContainerList(std::initializer_list<C> list) : list(list) {};
    ContainerList(const std::vector<C> list) : list(list) {};
    ~ContainerList() = default;

    inline static bool is_list(SharedVariable v) {
        return List::is_list(v, GetTypeIdentifier<C>());
    }

    virtual size_t size() const { return list.size(); }

    virtual TypeIdentifier element_type() const { return Type<C>::identifier; };

    virtual SharedVariable get(int index) const { return std::make_shared<ScalarVariable<C>>(list[index]); }

    const std::vector<C> elements() const {

        return list;

    }

private:

    std::vector<C> list;

};

typedef ContainerList<float> FloatList;
typedef ContainerList<int> IntegerList;
typedef ContainerList<bool> BooleanList;
typedef ContainerList<std::string> StringList;

PIXELPIPES_TYPE_NAME(std::vector<int>, "integer_list");
PIXELPIPES_TYPE_NAME(std::vector<float>, "float_list");
PIXELPIPES_TYPE_NAME(std::vector<bool>, "boolean_list");
PIXELPIPES_TYPE_NAME(std::vector<std::string>, "string_list");

constexpr static TypeIdentifier FloatListType = Type<std::vector<float>>::identifier;
constexpr static TypeIdentifier IntegerListType = Type<std::vector<int>>::identifier;
constexpr static TypeIdentifier BooleanListType = Type<std::vector<bool>>::identifier;
constexpr static TypeIdentifier StringListType = Type<std::vector<std::string>>::identifier;

template<typename T>
class Table: public List {
public:

    Table(const std::vector<std::vector<T>> source) : source(source) {

        if (source.size() > 0) {
            size_t rowsize = source[0].size();

            for (auto row : source) {
                VERIFY(rowsize == row.size(), "All rows must have equal length");
            }
            
        }

    }

    ~Table() = default;

    virtual size_t size() const {
        return source.size();
    }


    virtual TypeIdentifier element_type() const {
        return Type<std::vector<T>>::identifier;
    }

    virtual SharedVariable get(int index) const {

        return std::make_shared<ContainerList<T>>(source[index]);

    }

    virtual std::vector<std::vector<T>> elements() {

        return source;

    }

private:

    std::vector<std::vector<T>> source;

};

typedef Table<float> FloatTable;
typedef Table<int> IntegerTable;
typedef Table<bool> BooleanTable;
typedef Table<std::string> StringTable;

constexpr static TypeIdentifier FloatTableType = Type<std::vector<std::vector<float>>>::identifier;
constexpr static TypeIdentifier IntegerTableType = Type<std::vector<std::vector<int>>>::identifier;
constexpr static TypeIdentifier BooleanTableType = Type<std::vector<std::vector<bool>>>::identifier;
constexpr static TypeIdentifier StringTableType = Type<std::vector<std::vector<std::string>>>::identifier;

PIXELPIPES_TYPE_NAME(std::vector<std::vector<int>>, "integer_table");
PIXELPIPES_TYPE_NAME(std::vector<std::vector<float>>, "float_table");
PIXELPIPES_TYPE_NAME(std::vector<std::vector<bool>>, "boolean_table");
PIXELPIPES_TYPE_NAME(std::vector<std::vector<std::string>>, "string_table");

#define _LIST_GENERATE_CONVERTERS(T) template<> \
inline std::vector<T> extract(const SharedVariable v) { \
    VERIFY((bool) v, "Uninitialized variable"); \
    VERIFY(List::is_list(v), "Not a list"); \
    auto list = std::static_pointer_cast<List>(v); \
    return list->elements<T>(); \
} \
template<> \
inline SharedVariable wrap(const std::vector<T> v) { \
    return std::make_shared<ContainerList<T>>(v); \
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
SharedVariable wrap(const T v) {
    if constexpr (is_vector<typename T::value_type>::value) {
        return std::make_shared<Table<detail::inner_type_t<T>>>(v);
    } else {
        return std::make_shared<ContainerList>(v);
    }
}*/

}