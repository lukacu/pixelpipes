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

    PIXELPIPES_CONVERT_ENUM(Interpolation)
    PIXELPIPES_CONVERT_ENUM(BorderStrategy)
    PIXELPIPES_CONVERT_ENUM(ImageDepth)

    typedef struct PIXELPIPES_API Chunk
    {
        unsigned char *pointer;
        size_t length;

        bool operator==(const Chunk &rhs) const { return pointer == rhs.pointer && length == rhs.length; }
    } Chunk;

    static const Chunk end{0, 0};

    class PIXELPIPES_API ChunkCursor
    {
    public:
        virtual const Chunk current() const = 0;
        virtual void increment() = 0;
    };

    class PIXELPIPES_API ImageChunkIterator : public std::iterator<std::input_iterator_tag, Chunk>
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

    class PIXELPIPES_API ImageData : public Token
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

    class PIXELPIPES_API BufferImage : public ImageData
    {
    public:
        typedef std::function<void()> DescructorCallback;

        BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth);

        BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth, unsigned char *buffer, DescructorCallback callback);

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

        size_t image_width;

        size_t image_height;

        size_t image_channels;

        ImageDepth image_depth;

        unsigned char *buffer;

        DescructorCallback callback;
    };

    class PIXELPIPES_API ImageList : public List
    {
    public:
        ImageList(Span<Image> images);

        ~ImageList();

        virtual size_t size() const;

        virtual TypeIdentifier element_type_id() const;

        virtual SharedToken get(size_t index) const;

/*        ImageList();*/
        ImageList(const ImageList &);
        ImageList(ImageList &&);
        ImageList& operator=(const ImageList &);
        ImageList& operator=(ImageList &&);

    private:
        struct ImageListState;
        Implementation<ImageListState> data;
    };

    #define ImageListType GetTypeIdentifier<Span<Image>>()

    template <>
    PIXELPIPES_API Image extract(const SharedToken v);

    template <>
    PIXELPIPES_API std::vector<Image> extract(const SharedToken v);

    template <>
    PIXELPIPES_API SharedToken wrap(const std::vector<Image> v);

    template <>
    PIXELPIPES_API Sequence<Image> extract(const SharedToken v);

    template <>
    PIXELPIPES_API SharedToken wrap(const Span<Image> v);


    void PIXELPIPES_API copy(const Image source, Image destination);

}