#pragma once

#include <functional>
#include <memory>
#include <iterator>

#include <pixelpipes/type.hpp>
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

    typedef struct Chunk
    {
        unsigned char *pointer;
        size_t length;

        bool operator==(const Chunk &rhs) const { return pointer == rhs.pointer && length == rhs.length; }
    } Chunk;

    static const Chunk end{0, 0};

    class ChunkCursor
    {
    public:
        virtual const Chunk current() const = 0;
        virtual void increment() = 0;
    };

    class ImageChunkIterator : public std::iterator<std::input_iterator_tag, Chunk>
    {
    public:
        ImageChunkIterator(std::shared_ptr<ChunkCursor> impl);

        inline ImageChunkIterator &operator++()
        {
            impl->increment();
            return *this;
        }
        inline ImageChunkIterator operator++(int)
        {
            ImageChunkIterator tmp(*this);
            operator++();
            return tmp;
        }
        inline bool operator==(const ImageChunkIterator &rhs) const { return impl->current() == rhs.impl->current(); }
        inline bool operator!=(const ImageChunkIterator &rhs) const { return !(rhs.impl->current() == impl->current()); }
        inline const Chunk operator*() { return impl->current(); }
        inline const Chunk operator->() { return impl->current(); }

    protected:
        std::shared_ptr<ChunkCursor> impl;
    };

    class ImageData;

    typedef std::shared_ptr<ImageData> Image;

    #define ImageType GetTypeIdentifier<Image>()

    class ImageData : public Token
    {
    public:
        inline static bool is(SharedToken v)
        {
            return (v->type_id() == ImageType);
        }

        virtual TypeIdentifier type_id() const;

        virtual void describe(std::ostream &os) const;

        virtual ~ImageData() = default;

        virtual ImageDepth depth() const = 0;

        virtual size_t element() const;

        virtual size_t width() const = 0;

        virtual size_t height() const = 0;

        virtual size_t channels() const = 0;

        virtual TypeIdentifier backend() const = 0;

        virtual ImageChunkIterator begin() const;

        virtual ImageChunkIterator end() const;

        virtual size_t rowstep() const = 0;

        virtual size_t colstep() const = 0;

        virtual unsigned char *data() const = 0;
    };

    class BufferImage : public ImageData
    {
    public:
        BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth);

        virtual ~BufferImage();

        virtual ImageDepth depth() const;

        virtual size_t width() const;

        virtual size_t height() const;

        virtual size_t channels() const;

        virtual TypeIdentifier backend() const;

        virtual size_t rowstep() const;

        virtual size_t colstep() const;

        virtual unsigned char *data() const;

    private:
        ImageDepth image_depth;

        size_t image_width;

        size_t image_height;

        size_t image_channels;

        unsigned char *buffer;
    };

    class ImageList : public List
    {
    public:
        ImageList(std::vector<Image> images);

        ~ImageList() = default;

        virtual size_t size() const;

        virtual TypeIdentifier element_type_id() const;

        virtual SharedToken get(int index) const;

    private:
        std::vector<Image> images;
    };

    #define ImageListType GetListIdentifier<Image>()

    template <>
    inline Image extract(const SharedToken v)
    {
        if (!ImageData::is(v))
            throw TypeException("Not an image type");

        return std::static_pointer_cast<ImageData>(v);
    }

    template <>
    inline std::vector<Image> extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        return ImageList::cast(v)->elements<Image>();
    }

    template <>
    inline SharedToken wrap(const std::vector<Image> v)
    {
        return std::make_shared<ImageList>(v);
    }

    void copy(const Image source, Image destination);

}