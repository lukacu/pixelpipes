#pragma once
#include <memory>
#include <vector>
#include <type_traits>
#include <map>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>
#include <initializer_list>

#include <opencv2/core.hpp>

namespace pixelpipes {

enum class VariableType {Integer, Float, Image, Point, View, List};

enum class ComparisonOperation {EQUAL, LOWER, LOWER_EQUAL, GREATER, GREATER_EQUAL};
enum class LogicalOperation {AND, OR, NOT};
enum class ArithmeticOperation {ADD, SUBTRACT, MULTIPY, DIVIDE, POWER};

// TODO: move this to random
enum class Distribution {Normal, Uniform};

// TODO: move this to image processing module
enum class ImageDepth {Byte = 8, Short = 16, Float = 32, Double = 64};
enum class BorderStrategy {ConstantHigh, ConstantLow, Replicate, Reflect, Wrap};
enum class Interpolation {Nearest, Linear, Area, Cubic, Lanczos};

class Variable;
typedef std::shared_ptr<Variable> SharedVariable;

class Variable {
public:
    virtual VariableType type() const = 0;

    virtual void print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, const Variable& v);

};

template <VariableType T> 
class TypedVariable: public Variable {
public:

    static const VariableType dtype = T;

    virtual VariableType type() const { return T; };

};

class BaseException : public std::exception {
public:
    BaseException(std::string reason);

	const char * what () const throw ();

private:
    std::string reason;

};

class VariableException : public BaseException {
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

class Integer: public TypedVariable<VariableType::Integer> {
public:

    Integer(int value): value(value) {};

    ~Integer() = default;

    int get() const { return value; };

    static int get_value(SharedVariable v);

    virtual void print(std::ostream& os) const;

private:

    int value;

};

class Float: public TypedVariable<VariableType::Float> {
public:

    Float(float value): value(value) {};

    ~Float() = default;

    float get() const { return value; };

    static float get_value(SharedVariable v);

    virtual void print(std::ostream& os) const;

private:

    float value;

};

class Point: public TypedVariable<VariableType::Point> {
public:

    Point(cv::Point2f value): value(value) {};

    ~Point() = default;

    cv::Point2f get() const { return value; };

    //static cv::Point2f get_value(SharedVariable v) ;

    virtual void print(std::ostream& os) const;

private:

    cv::Point2f value;

};


class View: public TypedVariable<VariableType::View> {
public:
    View(const cv::Matx33f value);

    ~View() = default;

    cv::Matx33f get();

    static cv::Matx33f get_value(SharedVariable v);

    virtual void print(std::ostream& os) const;

private:

    cv::Matx33f value;

};

/**
 * @brief Variable container for an image. It contains an OpenCV Mat object that represents an image.
 * 
 */
class Image: public TypedVariable<VariableType::Image> {
public:
    Image(const cv::Mat value);

    ~Image() = default;

    cv::Mat get();

    /**
     * @brief Get the internal image representation for a variable or throw a VariableException.
     * 
     * @param v Variable container
     * @return cv::Mat Internal image representation
     */
    static cv::Mat get_value(SharedVariable v);

    virtual void print(std::ostream& os) const;

private:

    cv::Mat value;

};


class List: public TypedVariable<VariableType::List> {
public:

    ~List() = default;

    virtual size_t size() const = 0;

    virtual VariableType element_type() const = 0;

    virtual SharedVariable get(int index) const = 0; 

    inline static bool is_list(SharedVariable v) {
        return (v->type() == VariableType::List);
    }

    inline static bool is_list(SharedVariable v, VariableType type) {
        return (v->type() == VariableType::List) && std::static_pointer_cast<List>(v)->element_type() == type;
    }

    inline static size_t length(SharedVariable v) {
        if (!List::is_list(v)) throw VariableException("Not a list");
        return std::static_pointer_cast<List>(v)->size();
    }

    inline static std::shared_ptr<List> cast(SharedVariable v) {
        if (!List::is_list(v)) throw VariableException("Not a list");
        return std::static_pointer_cast<List>(v);
    }

    // Slow default template version
    template<class C, class V> std::vector<C> elements() {

        std::vector<C> result;

        for (size_t i = 0; i < size(); i++) {
            result.push_back(std::static_pointer_cast<V>(get(i))->get());
        }

        return result;

    }

    virtual void print(std::ostream& os) const;

};

typedef std::shared_ptr<List> SharedList;

class Sublist: public List {
public:
    Sublist(SharedList list, int from, int to);

    ~Sublist() = default;

    virtual size_t size() const;

    virtual VariableType element_type() const;

    virtual SharedVariable get(int index) const; 

private:

    SharedList list;

    int from, to;

};

class MappedList: public List {
public:
    MappedList(SharedList list, std::vector<int> map);

    ~MappedList() = default;

    virtual size_t size() const;

    virtual VariableType element_type() const;

    virtual SharedVariable get(int index) const; 

private:

    SharedList list;

    std::vector<int> map;

};

class ImageFileList: public List {
public:

    ImageFileList(std::vector<std::string> list, std::string prefix = std::string(), bool grayscale = false);

    ~ImageFileList() = default;

    virtual size_t size() const;

    virtual VariableType element_type() const;

    virtual SharedVariable get(int index) const; 

private:

    std::string prefix;

    std::vector<std::string> list;

    bool grayscale;

};

template<typename C, typename V>
class ContainerList: public List {
public:

    ContainerList(std::initializer_list<C> list) : list(list) {};
    ContainerList(const std::vector<C> list) : list(list) {};
    ~ContainerList() = default;

    virtual size_t size() const { return list.size(); }

    virtual VariableType element_type() const { return V::dtype; } ;

    virtual SharedVariable get(int index) const { return std::make_shared<V>(list[index]); }

    std::vector<C> elements() {

        return std::vector<C>(list);

    }

private:

    std::vector<C> list;

};

typedef ContainerList<cv::Mat, Image> ImageList;

typedef ContainerList<float, Float> FloatList;

typedef ContainerList<int, Integer> IntegerList;

typedef ContainerList<cv::Point2f, Point> PointList;

}