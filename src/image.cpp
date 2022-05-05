

#include <cstring>
#include <cmath>

#include <pixelpipes/image.hpp>
#include <pixelpipes/serialization.hpp>

namespace pixelpipes
{

    class EndCursor : public ChunkCursor
    {

    protected:
        virtual const Chunk current() const
        {
            return pixelpipes::end;
        }
        virtual void increment()
        {
        }
    };

    class SingleChunkCursor : public ChunkCursor
    {
    public:
        SingleChunkCursor(unsigned char *data, size_t length) : done(false), data(data), length(length) {}

        virtual const Chunk current() const
        {
            if (done)
            {
                return pixelpipes::end;
            }
            else
            {
                return Chunk{data, length};
            }
        }

        virtual void increment()
        {
            done = true;
        }

    private:
        bool done;
        unsigned char *data;
        size_t length;
    };

    class RowChunkCursor : public ChunkCursor
    {
    public:
        RowChunkCursor(unsigned char *data, size_t length, size_t stride, size_t rows) : data(data), length(length), stride(stride), position(0), total(rows) {}

        virtual const Chunk current() const
        {
            if (position >= total)
            {
                return pixelpipes::end;
            }
            else
            {
                return Chunk{data + stride * position, length};
            }
        }

        virtual void increment()
        {
            position++;
        }

    private:
        unsigned char *data;
        size_t length;
        size_t stride;
        size_t position;
        size_t total;
    };

    class ElementChunkCursor : public ChunkCursor
    {
    public:
        ElementChunkCursor(unsigned char *data, size_t length, size_t rowstride, size_t colstride, size_t rows, size_t cols) : data(data), length(length), rowstride(rowstride), colstride(colstride), row(0), col(0), rows(rows), cols(cols) {}

        virtual const Chunk current() const
        {
            if (row >= rows && col >= cols)
            {
                return pixelpipes::end;
            }
            else
            {
                return Chunk{data + rowstride * row + colstride * col, length};
            }
        }

        virtual void increment()
        {
            col++;

            if (col >= cols)
            {
                col = 0;
                row++;
            }
        }

    private:
        unsigned char *data;
        size_t length;
        size_t rowstride;
        size_t colstride;
        size_t row;
        size_t col;
        size_t rows;
        size_t cols;
    };

    void ImageData::describe(std::ostream &os) const
    {
        os << "[Image: width=" << width() << " height=" << height() << " channels=" << channels() << "]";
    }

    TypeIdentifier ImageData::type_id() const
    {
        return ImageType;
    }

    size_t ImageData::element() const
    {
        return ((size_t)depth() >> 3);
    }

    ImageChunkIterator ImageData::begin() const
    {

        if (width() * channels() * element() == rowstep())
        {
            if (channels() * element() == colstep())
            {
                size_t buffer_size = width() * height() * channels() * element();
                return ImageChunkIterator(std::make_shared<SingleChunkCursor>(data(), buffer_size));
            }
            else
            {
                return ImageChunkIterator(std::make_shared<ElementChunkCursor>(data(), element(), rowstep(), colstep(), height(), width()));
            }
        }
        else
        {
            if (channels() * element() == colstep())
            {
                size_t buffer_size = width() * channels() * element();
                return ImageChunkIterator(std::make_shared<RowChunkCursor>(data(), buffer_size, rowstep(), height()));
            }
            else
            {
                return ImageChunkIterator(std::make_shared<ElementChunkCursor>(data(), element(), rowstep(), colstep(), height(), width()));
            }
        }
    }

    ImageChunkIterator ImageData::end() const
    {
        return ImageChunkIterator(std::make_shared<EndCursor>());
    }

    ImageChunkIterator::ImageChunkIterator(std::shared_ptr<ChunkCursor> impl) : impl(impl){}

    BufferImage::BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth) : image_width(width), image_height(height), image_channels(channels), image_depth(depth), buffer(0), callback()
    {

        VERIFY(image_width > 0 && image_height > 0 && image_channels > 0, "Illegal input");

        size_t buffer_size = image_width * image_height * image_channels * ((size_t)image_depth >> 3);

        buffer = new unsigned char[buffer_size];
    }

    BufferImage::BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth, unsigned char *buffer, DescructorCallback callback) : image_width(width), image_height(height), image_channels(channels), image_depth(depth), buffer(buffer), callback(callback)
    {

        VERIFY(image_width > 0 && image_height > 0 && image_channels > 0, "Illegal input");

        VERIFY(buffer && callback, "Delegated memory must not be null and have a cleanup callback");

    }


    BufferImage::~BufferImage()
    {
        if (callback) {
            callback();
            buffer = nullptr;
        } else if (buffer)
            delete[] buffer;
    }

    ImageDepth BufferImage::depth() const
    {
        return image_depth;
    }

    size_t BufferImage::width() const
    {
        return image_width;
    }

    size_t BufferImage::height() const
    {
        return image_height;
    }

    size_t BufferImage::channels() const
    {
        return image_channels;
    }

    TypeIdentifier BufferImage::backend() const
    {
        return GetTypeIdentifier<BufferImage>();
    }

    size_t BufferImage::rowstep() const
    {
        return image_width * image_channels * element();
    }

    size_t BufferImage::colstep() const
    {
        return image_channels * element();
    }

    unsigned char *BufferImage::data() const
    {
        return buffer;
    }

    void copy(const Image source, Image destination)
    {

        VERIFY(source->width() == destination->width(), "Image width does not match");
        VERIFY(source->height() == destination->height(), "Image height does not match");
        VERIFY(source->channels() == destination->channels(), "Image channels do not match");
        VERIFY(source->depth() == destination->depth(), "Image depth does not match");

        ImageChunkIterator sit = source->begin();
        ImageChunkIterator dit = destination->begin();

        size_t soffset = 0;
        size_t doffset = 0;

        while (true)
        {
            size_t slen = (*sit).length - soffset;
            size_t dlen = (*dit).length - doffset;
            size_t length = (std::min)(slen, dlen);

            if (length == 0)
                break;

            std::memcpy((*dit).pointer + doffset, (*sit).pointer + soffset, length);

            doffset += length;
            soffset += length;

            if (slen == length)
            {
                soffset = 0;
                sit++;
            }

            if (dlen == length)
            {
                doffset = 0;
                dit++;
            }
        }
    }

    Type imate_type_constructor(const TypeParameters &type)
    {
        UNUSED(type);
        return Type(ImageType);
    }

    Type imate_type_denominator(const Type &me, const Type &other)
    {
        UNUSED(me);
        UNUSED(other);
        return Type(AnyType);
    }

    struct ImageList::ImageListState {
        std::vector<Image> data;

        ImageListState(Span<Image> inputs) : data(inputs.begin(), inputs.end()) {}
    };

    ImageList::ImageList(Span<Image> inputs) : data(inputs)
    {
    }

    ImageList::~ImageList() = default;
/*
    ImageList::ImageList() = default;*/
    ImageList::ImageList(const ImageList &) = default;
    ImageList::ImageList(ImageList &&) = default;

    ImageList& ImageList::operator=(const ImageList &) = default;
    ImageList& ImageList::operator=(ImageList &&) = default;

    size_t ImageList::size() const
    {
        return data->data.size();
    }

    TypeIdentifier ImageList::element_type_id() const
    {
        return ImageType;
    }

    SharedToken ImageList::get(size_t index) const
    {
        return data->data[index];
    }

    template <>
    Image extract(const SharedToken v)
    {
        if (!ImageData::is(v))
            throw TypeException("Not an image type");

        return std::static_pointer_cast<ImageData>(v);
    }

    template <>
    std::vector<Image> extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        return ImageList::cast(v)->elements<Image>();
    }

    template <>
    SharedToken wrap(const std::vector<Image> v)
    {
        return std::make_shared<ImageList>(make_span(v));
    }

    template <>
    Sequence<Image> extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        return ImageList::cast(v)->elements<Image>();
    }

    template <>
    SharedToken wrap(const Span<Image> v)
    {
        return std::make_shared<ImageList>(v);
    }




    PIXELPIPES_REGISTER_TYPE(ImageType, "image", imate_type_constructor, imate_type_denominator);

    void write_image(SharedToken v, std::ostream &target)
    {
        Image image = extract<Image>(v);
        write_t(target, image->height());
        write_t(target, image->width());
        write_t(target, image->channels());
        write_t(target, (unsigned short)image->depth());

        size_t total = 0;
        for (ImageChunkIterator it = image->begin(); it != image->end(); it++)
        {
            target.write((char *)(*it).pointer, (*it).length);
            total += (*it).length;
        }

        VERIFY(image->height() * image->width() * image->channels() * image->element() == total, "Image size mismatch");
    }

    PIXELPIPES_REGISTER_WRITER(ImageType, write_image);

    Image read_image(std::istream &source)
    {

        size_t height = read_t<size_t>(source);
        size_t width = read_t<size_t>(source);
        size_t channels = read_t<size_t>(source);
		unsigned short depth = read_t<unsigned short>(source);

        Image image = std::make_shared<BufferImage>(width, height, channels, (ImageDepth)depth);

        size_t length = image->height() * image->width() * image->channels() * image->element();

        source.read((char *)image->data(), length);

        return image;
    }

	PIXELPIPES_REGISTER_READER(ImageType, [](std::istream &source) -> SharedToken {return read_image(source); });

    PIXELPIPES_REGISTER_TYPE_DEFAULT(ImageListType, "image_list");

    PIXELPIPES_REGISTER_READER(ImageListType,
                               [](std::istream &source) -> SharedToken
                               {
                                   try
                                   {
                                       size_t len = read_t<size_t>(source);
                                       std::vector<Image> list;
                                       for (size_t i = 0; i < len; i++)
                                       {
                                           list.push_back(read_image(source));
                                       }
                                       return std::make_shared<ImageList>(make_span(list));
                                   }
                                   catch (std::bad_alloc const&)
                                   {
                                       throw SerializationException("Unable to allocate an array");
                                   }
                               }

    );

    PIXELPIPES_REGISTER_WRITER(ImageListType, [](SharedToken v, std::ostream &target)
                               {
        auto list = extract<std::vector<Image>>(v);
        write_t(target, list.size());
        for (size_t i = 0; i < list.size(); i++)
        {
            write_image(list[i], target);
        } });


    SharedToken ConstantImage(TokenList inputs, Image image)
    {
        VERIFY(inputs.size() == 0, "Incorrect number of parameters");
        return image;
    }

    // REGISTER_OPERATION_FUNCTION("image", ConstantImage, Image); TODO: support aliases
    REGISTER_OPERATION_FUNCTION("image_constant", ConstantImage, Image);

    class ConstantImages : public Operation
    {
    public:
        ConstantImages(Sequence<Image> images)
        {
            list = std::make_shared<ImageList>(images);
        }

        ~ConstantImages() = default;

        virtual SharedToken run(TokenList inputs)
        {
            VERIFY(inputs.size() == 0, "Incorrect number of parameters");
            return list;
        }

    protected:
        std::shared_ptr<ImageList> list;
    };

    REGISTER_OPERATION("image_list", ConstantImages, Sequence<Image>);

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    SharedToken GetImageProperties(TokenList inputs)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        Image image = extract<Image>(inputs[0]);

        return std::make_shared<IntegerList>(Sequence<int>({(int)image->width(), (int)image->height(), (int)image->channels(), (int)image->depth()}));
    }

    REGISTER_OPERATION_FUNCTION("image_properties", GetImageProperties);

}
