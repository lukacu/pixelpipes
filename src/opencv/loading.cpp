
#include <pixelpipes/operation.hpp>

#include "common.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <turbojpeg.h>

#include <limits>

namespace
{

    class TurboJPEGDecompress
    {
    public:
        TurboJPEGDecompress() : _handle(tjInitDecompress())
        {
            if (!_handle)
                throw pixelpipes::TypeException(tjGetErrorStr());
        }

        ~TurboJPEGDecompress()
        {
            if (_handle)
                tjDestroy(_handle);
        }

        tjhandle get() const
        {
            return _handle;
        }

    private:
        tjhandle _handle;
    };

    [[noreturn]] void throw_turbojpeg_error(tjhandle handle, const char *operation)
    {
        throw pixelpipes::TypeException(
            pixelpipes::Formatter() << operation << ": "
                                     << (handle ? tjGetErrorStr2(handle) : tjGetErrorStr()));
    }

    int jpeg_pixel_format(pixelpipes::ImageChannels channels)
    {
        switch (channels)
        {
        case pixelpipes::ImageChannels::GRAY:
            return TJPF_GRAY;
        case pixelpipes::ImageChannels::RGB:
            return TJPF_RGB;
        case pixelpipes::ImageChannels::RGBA:
            return TJPF_RGBA;
        }
        throw pixelpipes::TypeException("Unsupported JPEG channel format");
    }

    int jpeg_channel_count(pixelpipes::ImageChannels channels)
    {
        switch (channels)
        {
        case pixelpipes::ImageChannels::GRAY:
            return 1;
        case pixelpipes::ImageChannels::RGB:
            return 3;
        case pixelpipes::ImageChannels::RGBA:
            return 4;
        }
        throw pixelpipes::TypeException("Unsupported JPEG channel format");
    }

    int jpeg_target_depth(pixelpipes::DataType depth)
    {
        switch (depth)
        {
        case pixelpipes::DataType::Char:
            return CV_8U;
        case pixelpipes::DataType::UnsignedShort:
            return CV_16U;
        case pixelpipes::DataType::Short:
            return CV_16S;
        case pixelpipes::DataType::Integer:
            return CV_32S;
        case pixelpipes::DataType::Float:
            return CV_32F;
        case pixelpipes::DataType::Boolean:
            break;
        }
        throw pixelpipes::TypeException("Unsupported JPEG image depth");
    }

}

namespace pixelpipes
{

    bool is_jpeg_buffer(const BufferReference &buffer)
    {
        const auto data = buffer->const_data();
        // JPEG files start with SOI followed by a marker.  Checking the
        // marker prefix keeps this dispatch cheap while avoiding treating
        // arbitrary buffers beginning with only FF D8 as JPEG data.
        return data.size() >= 3 && data.data()[0] == 0xff &&
               data.data()[1] == 0xd8 && data.data()[2] == 0xff;
    }

    class JpegImageDecode : public Operation
    {
    public:
        JpegImageDecode(DataType depth, ImageChannels channels, bool normalize)
            : _channels(channels), _depth(depth), _normalize(normalize)
        {
            VERIFY(_depth != DataType::Boolean, "Boolean type not supported");
        }

        TokenReference run(const TokenList &inputs, const TokenReference &output) override
        {
            UNUSED(output);
            VERIFY(inputs.size() == 1, "JpegImageDecode: expected 1 input");

            auto buffer = extract<BufferReference>(inputs[0]);
            VERIFY(buffer->size() <= std::numeric_limits<unsigned long>::max(),
                   "JPEG buffer is too large for libjpeg-turbo");

            TurboJPEGDecompress decoder;
            int width = 0;
            int height = 0;
            int subsampling = 0;
            int colorspace = 0;
            if (tjDecompressHeader3(decoder.get(), buffer->data().data(),
                                    static_cast<unsigned long>(buffer->size()),
                                    &width, &height, &subsampling, &colorspace) != 0)
                throw_turbojpeg_error(decoder.get(), "JPEG header decode failed");

            const int channels = jpeg_channel_count(_channels);
            cv::Mat image(height, width, CV_MAKETYPE(CV_8U, channels));
            VERIFY(image.step <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "JPEG image row stride is too large for libjpeg-turbo");
            if (tjDecompress2(decoder.get(), buffer->data().data(),
                              static_cast<unsigned long>(buffer->size()), image.data,
                              width, static_cast<int>(image.step), height,
                              jpeg_pixel_format(_channels), 0) != 0)
                throw_turbojpeg_error(decoder.get(), "JPEG image decode failed");

            const int target_depth = jpeg_target_depth(_depth);
            double offset = -minimum_value(image);
            double scaling = maximum_value(target_depth) /
                             static_cast<double>(maximum_value(image) - minimum_value(image));
            if (!_normalize)
            {
                scaling = 1.0;
                offset = 0.0;
            }

            if (target_depth != image.depth() || scaling != 1.0 || offset != 0.0)
                image.convertTo(image, CV_MAKETYPE(target_depth, image.channels()), scaling, offset);

            return wrap(image);
        }

        TokenReference evaluate(const TokenList &inputs) override
        {
            VERIFY(inputs.size() == 1, "JpegImageDecode: expected 1 input");

            if (any_placeholder(inputs))
            {
                const int channels = jpeg_channel_count(_channels);
                return create<Placeholder>(Shape(GetType<char>(),
                                                 {static_cast<size_t>(channels), unknown, unknown}));
            }
            return run(inputs, TokenReference());
        }

        OperationTrait trait() const override
        {
            return OperationTrait::Compute | OperationTrait::Foldable;
        }

        Type type() const override
        {
            return GetType<JpegImageDecode>();
        }

        Sequence<TokenReference> serialize() override
        {
            return Sequence<TokenReference>({wrap(_channels), wrap(_depth), wrap(_normalize)});
        }

    private:
        ImageChannels _channels;
        DataType _depth;
        bool _normalize;
    };

    PIXELPIPES_OPERATION_CLASS("image_decode_jpeg", JpegImageDecode,
                               DataType, ImageChannels, bool);

    class ImageDecode : public Operation
    {
    public:
        ImageDecode(DataType depth, ImageChannels channels, bool normalize) : _channels(channels), _depth(depth), _normalize(normalize)
        {
            VERIFY(_depth != DataType::Boolean, "Boolean type not supported");
        }
        ~ImageDecode() {}

        TokenReference run(const TokenList &inputs, const TokenReference &output) override
        {
            UNUSED(output);
            VERIFY(inputs.size() == 1, "ImageDecode: expected 1 input");

            auto buffer = extract<BufferReference>(inputs[0]);

            // Keep the public image decoder format-agnostic, but route JPEGs
            // through libjpeg-turbo.  Non-JPEG formats retain the existing
            // OpenCV path, so callers do not need to select a format-specific
            // operation or decoder.
            if (is_jpeg_buffer(buffer))
            {
                JpegImageDecode jpeg_decoder(_depth, _channels, _normalize);
                return jpeg_decoder.run(inputs, TokenReference());
            }

            cv::Mat wrapper(1, buffer->size(), CV_8UC1, buffer->data().data());

            int decode_flags = cv::IMREAD_UNCHANGED;
            int depth = 0;

            switch (_depth)
            {
            case DataType::Char:
            {
                depth = CV_8U;
                break;
            }
            case DataType::UnsignedShort:
            {
                depth = CV_16U;
                break;
            }
            case DataType::Short:
            {
                depth = CV_16S;
                break;
            }
            case DataType::Integer:
            {
                depth = CV_32S;
                break;
            }
            case DataType::Float:
            {
                depth = CV_32F;
                break;
            }
            case DataType::Boolean:
                break;
            }

            cv::Mat image = cv::imdecode(wrapper, decode_flags);
            VERIFY(!image.empty(), "Image decode error");

            // Convert to correct depth if necessary, also scales the values
            double offset = -minimum_value(image);
            double scaling = maximum_value(depth) / (maximum_value(image) - minimum_value(image));

            if (!_normalize)
            {
                scaling = 1.0;
                offset = 0.0;
            }

            if (depth != image.depth() || scaling != 1.0 || offset != 0.0)
                image.convertTo(image, CV_MAKETYPE(depth, image.channels()), scaling, offset);

            switch (_channels)
            {
            case ImageChannels::GRAY:
                if (image.channels() == 3)
                    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                else if (image.channels() == 4)
                    cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
                break;
            case ImageChannels::RGB:
                if (image.channels() == 3)
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                else if (image.channels() == 1)
                    cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
                else if (image.channels() == 4)
                    cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
                break;
            case ImageChannels::RGBA:
                if (image.channels() == 3)
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
                else if (image.channels() == 1)
                    cv::cvtColor(image, image, cv::COLOR_GRAY2RGBA);
                break;
            }

            return wrap(image);
        }

        TokenReference evaluate(const TokenList &inputs) override
        {
            VERIFY(inputs.size() == 1, "ImageDecode: expected 1 input");

            if (any_placeholder(inputs))
            {
                Size channels;
                Type depth = AnyType;

                switch (_channels)
                {
                case ImageChannels::GRAY:
                    channels = 1;
                    break;
                case ImageChannels::RGB:
                    channels = 3;
                    break;
                case ImageChannels::RGBA:
                    channels = 4;
                    break;
                }

                switch (_depth)
                {
                case DataType::Char:
                    depth = GetType<char>();
                    break;
                case DataType::UnsignedShort:
                    depth = GetType<uint16_t>();
                    break;
                case DataType::Short:
                    depth = GetType<int16_t>();
                    break;
                case DataType::Integer:
                    depth = GetType<int32_t>();
                    break;
                case DataType::Float:
                    depth = GetType<float>();
                    break;
                case DataType::Boolean:
                    break;
                }

                return create<Placeholder>(Shape(depth, {channels, unknown, unknown}));
            }
            else
            {
                return run(inputs, TokenReference());
            }
        }

        OperationTrait trait() const override
        {
            return OperationTrait::Compute | OperationTrait::Foldable;
        }

        virtual Type type() const override
        {
            return GetType<ImageDecode>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(_channels), wrap(_depth), wrap(_normalize)}); }

    private:
        ImageChannels _channels;
        DataType _depth;
        bool _normalize;
    };

    PIXELPIPES_OPERATION_CLASS("image_decode", ImageDecode, DataType, ImageChannels, bool);

}
