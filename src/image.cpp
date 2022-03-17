

#include <cstring>

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
        SingleChunkCursor(unsigned char *data, size_t length) : data(data), length(length), done(false) {}

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

    TypeIdentifier ImageData::type() const
    {
        return ImageType;
    }

    bool ImageData::is_scalar() const
    {
        return true;
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

    ImageChunkIterator::ImageChunkIterator(std::shared_ptr<ChunkCursor> impl) : impl(impl){};

    BufferImage::BufferImage(size_t width, size_t height, size_t channels, ImageDepth depth) : image_width(width), image_height(height), image_channels(channels), image_depth(depth), buffer(0)
    {

        VERIFY(image_width > 0 && image_height > 0 && image_channels > 0, "Illegal input");

        size_t buffer_size = image_width * image_height * image_channels * ((size_t)image_depth >> 3);

        buffer = new unsigned char[buffer_size];
    }

    BufferImage::~BufferImage()
    {
        if (buffer)
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
            size_t length = std::min(slen, dlen);

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

    ImageList::ImageList(std::vector<Image> inputs) : images(inputs.begin(), inputs.end())
    {
    }

    size_t ImageList::size() const
    {
        return images.size();
    }

    TypeIdentifier ImageList::element_type() const
    {
        return ImageType;
    }

    SharedVariable ImageList::get(int index) const
    {
        return images[index];
    }

    PIXELPIPES_REGISTER_WRITER(Image, 
        [](SharedVariable v, std::ostream &target)
        {
        Image image = extract<Image>(v);
        write_t(target, image->height());
        write_t(target, image->width());
        write_t(target, image->channels());
        write_t(target, (ushort) image->depth());
        
        size_t total = 0;
        for (ImageChunkIterator it = image->begin(); it != image->end(); it++) {
            target.write((char*)(*it).pointer, (*it).length);
            total += (*it).length;
        }

        VERIFY(image->height() * image->width() * image->channels() * image->element() == total, "Image size mismatch");

        }

    );

    PIXELPIPES_REGISTER_READER(Image, 
        [](std::istream &source)
        {

            size_t height = read_t<size_t>(source);
            size_t width = read_t<size_t>(source);
            size_t channels = read_t<size_t>(source);
            ushort depth = read_t<ushort>(source);

            Image image = std::make_shared<BufferImage>(width, height, channels, (ImageDepth) depth);

            size_t length = image->height() * image->width() * image->channels() * image->element();

            source.read((char*)image->data(), length);

            return image;

        }

    );

}
