
#include <pixelpipes/operation.hpp>

#include "common.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace pixelpipes {

    class ImageDecode : public Operation
    {
    public:
        ImageDecode(ImageChannels channels, ImageDepth depth) : _channels(channels), _depth(depth) {}
        ~ImageDecode() {}

        TokenReference run(const TokenList& inputs) override
        {
            VERIFY(inputs.size() == 1, "ImageDecode: expected 1 input");

            auto buffer = extract<BufferReference>(inputs[0]);

            cv::Mat wrapper(1, buffer->size(), CV_8UC1, buffer->data().data());

            int decode_flags = 0;

            switch (_channels)
            {
            case ImageChannels::GRAY:
                decode_flags |= cv::IMREAD_GRAYSCALE;
                break;
            case ImageChannels::RGB:
                decode_flags |= cv::IMREAD_COLOR;
                break;
            case ImageChannels::RGBA:
                decode_flags |= cv::IMREAD_UNCHANGED;
                break;
            }

            switch (_depth)
            {
            case ImageDepth::Char:
                break;
            case ImageDepth::UShort:
                decode_flags |= cv::IMREAD_ANYDEPTH;
                break;
            case ImageDepth::Short:
                decode_flags |= cv::IMREAD_ANYDEPTH;
                break;
            case ImageDepth::Integer:
                decode_flags |= cv::IMREAD_ANYDEPTH;
                break;
            case ImageDepth::Float:
                decode_flags |= cv::IMREAD_ANYDEPTH;
                break;
            }

            cv::Mat image = cv::imdecode(wrapper, decode_flags);
            VERIFY(!image.empty(), "Image decode error");

            // Convert to correct depth if necessary, also scales the values
            double offset = -minimum_value(image);
            double scaling = 1.0 / (maximum_value(image) - minimum_value(image));

            switch (_depth)
            {
            case ImageDepth::Char:
                if (image.depth() != CV_8U)
                    image.convertTo(image, CV_8U, 255 * scaling, offset);
                break;
            case ImageDepth::UShort:
                if (image.depth() != CV_16U)
                    image.convertTo(image, CV_MAKETYPE(CV_16U, image.channels()), 65535 * scaling, offset);
                break;
            case ImageDepth::Short:
                if (image.depth() != CV_16S)
                    image.convertTo(image, CV_MAKETYPE(CV_16S, image.channels()), 65535 * scaling, offset);
                break;
            case ImageDepth::Integer:
                if (image.depth() != CV_32S)
                    image.convertTo(image, CV_MAKETYPE(CV_32S, image.channels()), 65535 * scaling, offset);
                break;
            case ImageDepth::Float:
                if (image.depth() != CV_32F)
                    image.convertTo(image, CV_MAKETYPE(CV_32F, image.channels()), 1.0 * scaling, offset);
                break;
            }

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

        TokenReference evaluate(const TokenList& inputs) override
        {
            VERIFY(inputs.size() == 1, "ImageDecode: expected 1 input");

            if (any_placeholder(inputs))
            {
                Size channels;
                TypeIdentifier depth = AnyType;

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
                case ImageDepth::Char:
                    depth = GetTypeIdentifier<uint8_t>();
                    break;
                case ImageDepth::UShort:
                    depth = GetTypeIdentifier<uint16_t>();
                    break;
                case ImageDepth::Short:
                    depth = GetTypeIdentifier<int16_t>();
                    break;
                case ImageDepth::Integer:
                    depth = GetTypeIdentifier<int32_t>();
                    break;
                case ImageDepth::Float:
                    depth = GetTypeIdentifier<float>();
                    break;
                }

                return create<Placeholder>(Shape(depth, {channels, unknown, unknown}));

            }
            else
            {
                return run(inputs);
            }
            
        }

        OperationTrait trait() const override
        {
            return OperationTrait::Compute;
        }

        virtual TypeIdentifier type() const override
        {
            return GetTypeIdentifier<ImageDecode>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(_channels), wrap(_depth)}); }

    private:
        ImageChannels _channels;
        ImageDepth _depth;
    
    };
    
    PIXELPIPES_OPERATION_CLASS("image_decode", ImageDecode, ImageChannels, ImageDepth);

}
