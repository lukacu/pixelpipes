#pragma once

#include <functional>
#include <memory>
#include <iterator>

#include <pixelpipes/types.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    enum class Interpolation
    {
        Nearest,
        Linear,
        Area,
        Cubic,
        Lanczos
    };
    enum class BorderStrategy
    {
        ConstantHigh,
        ConstantLow,
        Replicate,
        Reflect,
        Wrap
    };
    enum class ImageDepth
    {
        Byte = 8,
        Short = 16,
        Float = 32,
        Double = 64
    };

    PIXELPIPES_CONVERT_ENUM(Interpolation);
    PIXELPIPES_CONVERT_ENUM(BorderStrategy);
    PIXELPIPES_CONVERT_ENUM(ImageDepth);

    typedef std::function<void()> DescructorCallback;

    typedef struct Chunk
    {
        char *pointer;
        size_t length;

        bool operator==(const Chunk &rhs) const { return pointer == rhs.pointer && length == rhs.length; }
    } Chunk;

    class ImageDataIterator : public std::iterator<std::input_iterator_tag, Chunk>
    {
    public:
        ImageDataIterator &operator++()
        {
            increment();
            return *this;
        }
        /*ImageDataIterator operator++(int)
        {
            ImageDataIterator tmp(*this);
            operator++();
            return tmp;
        }*/
        bool operator==(const ImageDataIterator &rhs) const { return current() == rhs.current(); }
        bool operator!=(const ImageDataIterator &rhs) const { return ! (rhs.current() == current()); }
        Chunk &operator*() { return current(); }

    protected:
        virtual Chunk &current() const = 0;
        virtual void increment() = 0;

    };

    class ImageData : public std::enable_shared_from_this<ImageData>
    {
    public:
        virtual ~ImageData() = default;

        virtual ImageDepth depth() const = 0;

        virtual size_t shape(size_t i) const = 0;

        virtual const std::vector<size_t> shape() const = 0;

        virtual size_t stride(size_t i) const = 0;

        virtual size_t ndims() const = 0;

        virtual TypeIdentifier backend() const = 0;

        virtual ImageDataIterator& begin() = 0;

        virtual ImageDataIterator& end() = 0;
    };

    class BufferImage : public ImageData
    {
    public:
        BufferImage(detail::any_container<size_t> dims, ImageDepth depth);

        virtual ~BufferImage();

        virtual ImageDepth depth() const;

        virtual size_t shape(size_t i) const;

        virtual const std::vector<size_t> shape() const;

        virtual size_t stride(size_t i) const;

        virtual size_t ndims() const;

        virtual TypeIdentifier backend() const;

        virtual ImageDataIterator& begin();

        virtual ImageDataIterator& end();

    private:

        ImageDepth data_depth;

        std::vector<size_t> dimensions;

        unsigned char* buffer;

    };

    typedef std::shared_ptr<ImageData> Image;

    constexpr static TypeIdentifier ImageType = GetTypeIdentifier<Image>();

    PIXELPIPES_TYPE_NAME(Image, "image");

    class ImageVariable : public ContainerVariable<Image>
    {
    public:
        ImageVariable(Image value);
        ~ImageVariable() = default;

        virtual void describe(std::ostream &os) const;
    };

    typedef ContainerList<Image> ImageList;

    constexpr static TypeIdentifier ImageListType = Type<std::vector<Image>>::identifier;

    template <>
    inline Image extract(const SharedVariable v)
    {
        if (!ImageVariable::is(v))
            throw VariableException("Not an image type");

        return std::static_pointer_cast<ImageVariable>(v)->get();
    }

    template <>
    inline SharedVariable wrap(const Image v)
    {
        return std::make_shared<ImageVariable>(v);
    }

}