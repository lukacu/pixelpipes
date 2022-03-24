#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <pixelpipes/operation.hpp>
#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

#include "common.hpp"

namespace pixelpipes
{

    PIXELPIPES_MODULE(image);

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ocv_border_type(BorderStrategy b, int *value)
    {
        switch (b)
        {
        case BorderStrategy::ConstantHigh:
            *value = 1;
            return cv::BORDER_CONSTANT;
        case BorderStrategy::ConstantLow:
            *value = 0;
            return cv::BORDER_CONSTANT;
        case BorderStrategy::Replicate:
            return cv::BORDER_REPLICATE;
        case BorderStrategy::Reflect:
            return cv::BORDER_REFLECT;
        case BorderStrategy::Wrap:
            return cv::BORDER_WRAP;
        default:
            throw TypeException("Illegal border strategy value");
        }
    }

    ImageDepth decode_ocvtype(int cvtype)
    {

        int depth = CV_MAT_DEPTH(cvtype);

        ImageDepth dtype;

        switch (depth)
        {
        case CV_8U:
        {
            dtype = ImageDepth::Byte;
            break;
        }
        case CV_8S:
        {
            dtype = ImageDepth::Byte;
            break;
        }
        case CV_16U:
        {
            dtype = ImageDepth::Short;
            break;
        }
        case CV_16S:
        {
            dtype = ImageDepth::Short;
            break;
        }
        case CV_32F:
        {
            dtype = ImageDepth::Float;
            break;
        }
        case CV_64F:
        {
            dtype = ImageDepth::Double;
            break;
        }
        default:
        {
            throw TypeException("Unsupported data type");
        }
        }

        return dtype;
    }

    int encode_ocvtype(ImageDepth dtype)
    {

        int type = 0;

        switch (dtype)
        {
        case ImageDepth::Byte:
        {
            type = CV_8U;
            break;
        }
        case ImageDepth::Short:
        {
            type = CV_16U;
            break;
        }
        case ImageDepth::Float:
        {
            type = CV_32F;
            break;
        }
        case ImageDepth::Double:
        {
            type = CV_64F;
            break;
        }
        default:
        {
            throw TypeException("Unsupported data type");
        }
        }

        return type;
    }

    cv::Mat MatImage::make(const Image image)
    {

        int type = encode_ocvtype(image->depth());

        int ndims = image->channels() == 1 ? 2 : 3;

        int size[3] = {(int)image->height(), (int)image->width(), (int)image->channels()};

        if (ndims == 3 && size[2] <= CV_CN_MAX)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        cv::Mat mat(ndims, size, type);

        Image destination = std::make_shared<MatImage>(mat);

        copy(image, destination);

        return mat;
    }

    cv::Mat MatImage::wrap(const Image image)
    {

        int type = encode_ocvtype(image->depth());

        int ndims = image->channels() == 1 ? 2 : 3;

        int size[3] = {(int)image->height(), (int)image->width(), (int)image->channels()};
        size_t step[3] = {(size_t)image->rowstep(), (size_t)image->colstep(), 1};

        //bool transposed = false;

        /*if (image->channels() *image->element() != image->colstep())
        {
            size[2] = 1;
            step[2] = image->element();
            ndims++;
        }

        if (ndims >= 2 && step[0] < step[1])
        {
            std::swap(size[0], size[1]);
            std::swap(step[0], step[1]);
            transposed = true;
        }*/

        if (ndims == 3 && size[2] <= CV_CN_MAX)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        //std::cout << "gg " << (size_t)image->rowstep() << " " << (size_t)image->colstep() << " " << image->channels() << " " << image->element() << std::endl;

        //std::cout << (int)image->data()[0] << std::endl;



        cv::Mat mat(ndims, size, type, (void *)image->data(), step);

        //std::cout << "a  " << mat.step[0] << " " << mat.step[1] << " " << mat.step[2] << std::endl;

        return mat;
    }

    MatImage::MatImage(cv::Mat data) : mat(data)
    {

        VERIFY(data.dims == 2, "Only two dimension matrices supported");

        decode_ocvtype(data.type());
    }

    MatImage::~MatImage()
    {
    }

    ImageDepth MatImage::depth() const
    {
        return decode_ocvtype(mat.type());
    }

    size_t MatImage::element() const
    {
        return ((size_t) depth() >> 3);
    }

    size_t MatImage::width() const
    {
        return mat.cols;
    }

    size_t MatImage::height() const
    {
        return mat.rows;
    }

    size_t MatImage::channels() const
    {
        return CV_MAT_CN(mat.type());
    }

    TypeIdentifier MatImage::backend() const
    {
        return GetTypeIdentifier<cv::Mat>();
    }

    size_t MatImage::rowstep() const
    {
        return mat.step[0];
    }

    size_t MatImage::colstep() const
    {
        return mat.step[1];
    }

    unsigned char *MatImage::data() const
    {
        return mat.data;
    }

    class ImageFileList : public List
    {
    public:
        ImageFileList(std::vector<std::string> list, std::string prefix = std::string(), bool grayscale = false);

        ~ImageFileList() = default;

        virtual size_t size() const;

        virtual TypeIdentifier element_type() const;

        virtual SharedToken get(int index) const;

    private:
        std::string prefix;

        std::vector<std::string> list;

        bool grayscale;
    };

    ImageFileList::ImageFileList(std::vector<std::string> list, std::string prefix, bool grayscale) : prefix(prefix), list(list), grayscale(grayscale)
    {

        if (list.empty())
            throw TypeException("File list is empty");
    }

    size_t ImageFileList::ImageFileList::size() const
    {
        return list.size();
    }

    TypeIdentifier ImageFileList::element_type() const
    {
        return ImageType;
    }

    SharedToken ImageFileList::get(int index) const
    {

        if (index < 0 || index >= (int)list.size())
        {
            throw TypeException("Index out of range");
        }

        cv::Mat image = cv::imread(prefix + list[index], grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

        if (image.empty())
        {
            throw TypeException("Image not found: " + prefix + list[index]);
        }

        if (image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }

        return wrap(image);
    }

    SharedToken ConstantImage(std::vector<SharedToken> inputs, Image image)
    {
        VERIFY(inputs.size() == 0, "Incorrect number of parameters");
        return image;
    }

    // REGISTER_OPERATION_FUNCTION("image", ConstantImage, Image); TODO: support aliases
    REGISTER_OPERATION_FUNCTION("constant", ConstantImage, Image);

    class ConstantImages : public Operation
    {
    public:
        ConstantImages(std::vector<Image> images)
        {
            list = std::make_shared<ImageList>(images);
        }

        ~ConstantImages() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {
            VERIFY(inputs.size() == 0, "Incorrect number of parameters");
            return list;
        }

    protected:
        std::shared_ptr<ImageList> list;
    };

    REGISTER_OPERATION("images", ConstantImages, std::vector<Image>);

    class ImageFileListSource : public Operation
    {
    public:
        ImageFileListSource(std::vector<std::string> list, std::string prefix, bool grayscale)
        {
            filelist = std::make_shared<ImageFileList>(list, prefix, grayscale);
        }

        ~ImageFileListSource() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {
            VERIFY(inputs.size() == 0, "Incorrect number of parameters");
            return filelist;
        }

    protected:
        std::shared_ptr<ImageFileList> filelist;
    };

    REGISTER_OPERATION("filelist", ImageFileListSource, std::vector<std::string>, std::string, bool);

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    SharedToken GetImageProperties(std::vector<SharedToken> inputs)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        int dtype = -1;
        switch (image.depth())
        {
        case CV_8U:
            dtype = (int)ImageDepth::Byte;
            break;
        case CV_16S:
            dtype = (int)ImageDepth::Short;
            break;
        case CV_32F:
            dtype = (int)ImageDepth::Float;
            break;
        case CV_64F:
            dtype = (int)ImageDepth::Double;
            break;
        }

        return std::make_shared<IntegerList>(std::vector<int>({image.cols, image.rows, image.channels(), dtype}));
    }

    REGISTER_OPERATION_FUNCTION("properties", GetImageProperties);

    /**
     * @brief Converts depth of an image.
     *
     */
    SharedToken ConvertDepth(std::vector<SharedToken> inputs, ImageDepth depth) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        double maxin = maximum_value(image);
        int dtype;
        double maxout = 1;

        switch (depth)
        {
        case ImageDepth::Byte:
            dtype = CV_8U;
            maxout = 255;
            break;
        case ImageDepth::Short:
            dtype = CV_16S;
            maxout = 255 * 255;
            break;
        case ImageDepth::Float:
            dtype = CV_32F;
            maxout = 1;
            break;
        case ImageDepth::Double:
            dtype = CV_64F;
            maxout = 1;
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required
            return inputs[0];
        }

        cv::Mat result;
        image.convertTo(result, dtype, maxout / maxin);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("convert", ConvertDepth, ImageDepth);

    /**
     * @brief Converts color image to grayscale image.
     *
     */
    SharedToken Grayscale(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        cv::Mat result;
        cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("grayscale", Grayscale);

    /**
     * @brief Returns an image with selected values.
     *
     */
    SharedToken Equals(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        int value = Integer::get_value(inputs[1]);

        VERIFY(image.channels() == 1, "Image has more than one channel");
        VERIFY(image.depth() == CV_8U || image.depth() == CV_16S, "Only integer bit types supported");

        cv::Mat result = (image == value);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("equals", Equals);

    /**
     * @brief Extracts a single channel from multichannel image.
     *
     */
    SharedToken Channel(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        int index = Integer::get_value(inputs[1]);

        VERIFY(index >= 0 && index < image.channels(), "Wrong channel index, out of bounds");

        cv::Mat result;
        cv::extractChannel(image, result, index);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("channel", Channel);

    /**
     * @brief Combines 3 single channel images into color image.
     *
     */
    SharedToken Merge(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 3, "Incorrect number of parameters");

        cv::Mat image_0 = extract<cv::Mat>(inputs[0]);
        cv::Mat image_1 = extract<cv::Mat>(inputs[1]);
        cv::Mat image_2 = extract<cv::Mat>(inputs[2]);

        VERIFY(image_0.depth() == image_1.depth() && image_1.depth() == image_2.depth(), "Image types do not match");
        VERIFY(image_0.rows == image_1.rows && image_1.rows == image_2.rows, "Image sizes do not match");
        VERIFY(image_0.cols == image_1.cols && image_1.cols == image_2.cols, "Image sizes do not match");

        std::vector<cv::Mat> channels;
        cv::Mat result;

        channels.push_back(image_0);
        channels.push_back(image_1);
        channels.push_back(image_2);
        CV_EX_WRAP(cv::merge(channels, result));

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("merge", Merge);

    /**
     * @brief Calculates image moments.
     *
     */
    SharedToken Moments(std::vector<SharedToken> inputs, bool binary) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        VERIFY(image.channels() == 1, "Image has more than one channel");

        cv::Moments m = cv::moments(image, binary);

        std::vector<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

        return std::make_shared<FloatList>(data);
    }

    REGISTER_OPERATION_FUNCTION("moments", Moments, bool);

    /**
     * @brief Tabulates a function into a matrix of a given size
     *
     */
    SharedToken MapFunction(std::vector<SharedToken> inputs, int function, bool normalize) noexcept(false)
    {

        VERIFY(inputs.size() > 1, "Incorrect number of parameters");

        int size_x = Integer::get_value(inputs[0]);
        int size_y = Integer::get_value(inputs[1]);

        cv::Mat result(size_y, size_x, CV_32F);

        switch (function)
        {
        case 0:
        {
            VERIFY(inputs.size() == 6, "Incorrect number of parameters");

            float mean_x = Float::get_value(inputs[2]);
            float mean_y = Float::get_value(inputs[3]);
            float sigma_x = Float::get_value(inputs[4]);
            float sigma_y = Float::get_value(inputs[5]);

            // intialising standard deviation to 1.0
            float sigma = 1.0;
            float r, s = 2.0 * sigma * sigma;
            // sum is for normalization
            float sum = 0.0;

            // generating 5x5 kernel
            for (int x = 0; x < size_x; x++)
            {
                for (int y = 0; y < size_y; y++)
                {
                    float px = x - mean_x;
                    float py = y - mean_y;
                    r = (px * px) / (2 * sigma_x * sigma_x) + (py * py) / (2 * sigma_y * sigma_y);
                    float v = exp(-r);
                    sum += v;
                    result.at<float>(y, x) = v;
                }
            }

            if (normalize)
                result /= sum;
        }
        }

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("map", MapFunction, int, bool);

    /**
     * @brief Thresholds an image.
     *
     */
    SharedToken ImageThreshold(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 2, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);
        float threshold = Float::get_value(inputs[1]);

        VERIFY(image.channels() == 1, "Image has more than one channel");

        float maxval = maximum_value(image);

        cv::Mat result;
        cv::threshold(image, result, threshold, maxval, cv::THRESH_BINARY);

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("threshold", ImageThreshold);

    /**
     * @brief Inverts image pixel values.
     *
     */
    SharedToken Invert(std::vector<SharedToken> inputs) noexcept(false)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        cv::Mat image = extract<cv::Mat>(inputs[0]);

        float maxval = maximum_value(image);

        cv::Mat result = maxval - image;

        return wrap(result);
    }

    REGISTER_OPERATION_FUNCTION("invert", Invert);

}
