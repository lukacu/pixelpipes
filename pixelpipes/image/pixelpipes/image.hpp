#pragma once

#include <pixelpipes/types.hpp>
#include <pixelpipes/operation.hpp>

#include <opencv2/core.hpp>

namespace pixelpipes {

enum class Interpolation {Nearest, Linear, Area, Cubic, Lanczos};
enum class BorderStrategy {ConstantHigh, ConstantLow, Replicate, Reflect, Wrap};
enum class ImageDepth {Byte = 8, Short = 16, Float = 32, Double = 64};


constexpr static TypeIdentifier ImageType = GetTypeIdentifier<cv::Mat>();

class Image: public ScalarVariable<cv::Mat> {
public:

    Image(cv::Mat value);
    ~Image() = default;

    virtual void describe(std::ostream& os) const;

};

class ImageFileList: public List {
public:

    ImageFileList(std::vector<std::string> list, std::string prefix = std::string(), bool grayscale = false);

    ~ImageFileList() = default;

    virtual size_t size() const;

    virtual TypeIdentifier element_type() const;

    virtual SharedVariable get(int index) const; 

private:

    std::string prefix;

    std::vector<std::string> list;

    bool grayscale;

};

typedef ContainerList<cv::Mat> ImageList;

constexpr static TypeIdentifier ImageListType = Type<std::vector<cv::Mat>>::identifier;

template<typename T>
struct Conversion <T, typename std::enable_if<std::is_same<T, cv::Mat>::value, T >::type> {

    static T extract(const SharedVariable v) {
        if (!Image::is(v))
            throw VariableException("Not an image value");

        return std::static_pointer_cast<Image>(v)->get();
    }

    static SharedVariable wrap(const T v) {
        return std::make_shared<Image>(v);
    }

};


}