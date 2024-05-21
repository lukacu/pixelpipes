
#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wc11-extensions")
#pragma clang diagnostic ignored "-Wc11-extensions"
#endif
#endif

#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>

#include <opencv2/core.hpp>

#define CV_EX_WRAP(S)                              \
    try                                            \
    {                                              \
        S;                                         \
    }                                              \
    catch (cv::Exception & e)                      \
    {                                              \
        throw pixelpipes::TypeException(e.what()); \
    }

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

    enum class ColorConversion
    {
        RGB_GRAY,
        GRAY_RGB,
        RGB_HSV,
        HSV_RGB,
        RGB_YCRCB,
        YCRCB_RGB
    };

    enum class ImageChannels
    {
        GRAY = 1,
        RGB = 3,
        RGBA = 4
    };

    PIXELPIPES_CONVERT_ENUM(Interpolation)
    PIXELPIPES_CONVERT_ENUM(BorderStrategy)
    PIXELPIPES_CONVERT_ENUM(ImageChannels)
    PIXELPIPES_CONVERT_ENUM(ColorConversion)

    template <>
    inline cv::Point2f extract(const TokenReference &v)
    {
        Point2D p = extract<Point2D>(v);
        return cv::Point2f(p.x, p.y);
    }

    template <>
    inline TokenReference wrap(const cv::Point2f v)
    {
        return wrap(Point2D{v.x, v.y});
    }

    template <>
    inline std::vector<cv::Point2f> extract(const TokenReference &v)
    {
        Sequence<Point2D> original = extract<Sequence<Point2D>>(v);

        std::vector<cv::Point2f> convert;
        convert.reserve(original.size());
        for (auto p : original)
        {
            convert.push_back(cv::Point2f{p.x, p.y});
        }

        return convert;
    }

    template <>
    inline cv::Rect extract(const TokenReference &v)
    {
        auto r = extract<Rectangle>(v);
        return cv::Rect((int)r.left, (int)r.top, (int)(r.right - r.left), (int)(r.bottom - r.top));
    }

    template <>
    inline TokenReference wrap(const std::vector<cv::Point2f> &v)
    {

        Sequence<Point2D> convert(v.size());

        size_t i = 0;
        for (auto p : v)
        {
            convert[i++] = (Point2D{p.x, p.y});
        }

        return wrap(convert);
    }

    // TODO: remove this, replace with better OperationWrapper handling
    template <>
    inline TokenReference wrap(std::vector<cv::Point2f> v)
    {

        Sequence<Point2D> convert(v.size());

        size_t i = 0;
        for (auto p : v)
        {
            convert[i++] = (Point2D{p.x, p.y});
        }

        return wrap(convert);
    }

    template <>
    inline TokenReference wrap(View<cv::Point2f> v)
    {

        Sequence<Point2D> convert(v.size());

        size_t i = 0;
        for (auto p : v)
        {
            convert[i++] = (Point2D{p.x, p.y});
        }

        return wrap(convert);
    }

    template <>
    inline cv::Matx22f extract(const TokenReference &v)
    {
        View2D d = extract<View2D>(v);

        cv::Matx22f m{d.m00, d.m01, d.m02, d.m10, d.m11, d.m12, d.m20, d.m21, d.m22};
        return m;
    }

    template <>
    inline TokenReference wrap(const cv::Matx22f v)
    {
        View2D d{v(0, 0), v(0, 1), v(1, 0), v(1, 1)};

        return wrap(d);
    }

    template <>
    inline cv::Matx33f extract(const TokenReference &v)
    {
        View2D d = extract<View2D>(v);

        cv::Matx33f m{d.m00, d.m01, d.m02, d.m10, d.m11, d.m12, d.m20, d.m21, d.m22};
        return m;
    }

    template <>
    inline TokenReference wrap(const cv::Matx33f v)
    {
        View2D d{v(0, 0), v(0, 1), v(0, 2), v(1, 0), v(1, 1), v(1, 2), v(2, 0), v(2, 1), v(2, 2)};

        return wrap(d);
    }


    class MatImage : public Tensor
    {
    public:
        MatImage(cv::Mat data);

        virtual ~MatImage();

        virtual void describe(std::ostream &os) const override;

        virtual Shape shape() const override;

        virtual size_t length() const override;

        virtual size_t size() const override;

        virtual TokenReference get(size_t i) const override;

        virtual TokenReference get(const Sizes &i) const override;

        virtual ReadonlySliceIterator read_slices() const override;

        virtual WriteableSliceIterator write_slices() override;

        virtual ByteView const_data() const override;

        virtual ByteSpan data() override;

        virtual SizeSequence strides() const override;

        virtual cv::Mat get() const;

        virtual size_t cell_size() const override;

        virtual Type datatype() const override;

    private:
        cv::Mat _mat;
        SizeSequence _shape;
        SizeSequence _strides;
        Type _element;
    };

    cv::Mat wrap_tensor(const TensorReference &tensor);

    //TensorReference create_tensor(const cv::Mat &mat);

    template <>
    inline cv::Mat extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->is<MatImage>())
        {
            return v->cast<MatImage>()->get();
        }

        if (v->is<Tensor>())
        {
            return wrap_tensor(cast<Tensor>(v));
        }

        throw TypeException("Not an image value");
    }

    template <>
    inline TokenReference wrap(const cv::Mat v)
    {
        return create<MatImage>(v);
    }

    inline int maximum_value(int ocv_depth)
    {

        switch (ocv_depth)
        {
        case CV_8U:
            return 255;
        case CV_16U:
            return std::numeric_limits<uint16_t>::max();
        case CV_16S:
            return std::numeric_limits<int16_t>::max();
        case CV_32S:
            return std::numeric_limits<int32_t>::max();
        case CV_32F:
            return 1;
        default:
            throw TypeException("Unsupported image depth");
        }
    }

    inline int maximum_value(cv::Mat image)
    {
        return maximum_value(image.depth());
    }


    inline int minimum_value(int ocv_depth) 
    {

        switch (ocv_depth)
        {
        case CV_8U:
            return 0;
        case CV_16U:
        case CV_16S:
            return 0;
        case CV_32S:
            return std::numeric_limits<int32_t>::min();
        case CV_32F:
            return 0;
        default:
            throw TypeException("Unsupported image depth");
        }
    }

    inline int minimum_value(cv::Mat image)
    {
        return minimum_value(image.depth());
    }

   

    int ocv_border_type(BorderStrategy b, int *value);

    TokenReference forward_image_type(const TokenList &inputs);

    typedef struct ImageShape
    {
        Size width;
        Size height;
        Size channels;
        Type type;
    } ImageShape;

    inline ImageShape image_shape(const TokenReference &image)
    {
        auto shape = image->shape();
        if (shape.rank() < 3)
        {
            return {shape[1], shape[0], 1, shape.element()};
        }
        VERIFY(shape.rank() == 3, "Image has more than 3 dimensions");
        return {shape[2], shape[1], shape[0], shape.element()};
    }

    inline Size min(Size a, Size b) {
        if (a == unknown || b == unknown) return unknown;
        return std::min((size_t)a, (size_t)b);
    }

    inline Size max(Size a, Size b) {
        if (a == unknown || b == unknown) return unknown;
        return std::max((size_t)a, (size_t)b);
    }

    template <int W, int H, Type T, int C>
    TokenReference given_shape(const TokenList &inputs) noexcept(false)
    {
        VERIFY(inputs.size() >= W && inputs.size() >= H, "Incorrect number of parameters");
        return create<Placeholder>(Shape(T, {C, get_size(inputs[H]), get_size(inputs[W])}));
    }

    template <int I>
    TokenReference forward_shape(const TokenList &inputs) noexcept(false)
    {
        VERIFY(inputs.size() >= I, "Incorrect number of parameters");
        return create<Placeholder>(inputs[I]->shape());
    }

    template <int I, class T>
    TokenReference forward_shape_new_type(const TokenList &inputs) noexcept(false)
    {
        VERIFY(inputs.size() >= I, "Incorrect number of parameters");
        return create<Placeholder>(inputs[I]->shape().cast(GetType<T>()));
    }

    template <int W, int H, int I>
    TokenReference given_shape_type(const TokenList &inputs) noexcept(false)
    {
        VERIFY(inputs.size() >= W && inputs.size() >= H && inputs.size() >= I, "Incorrect number of parameters");
        auto shape = image_shape(inputs[I]);
        return create<Placeholder>(Shape(shape.type, {shape.channels, get_size(inputs[H]), get_size(inputs[W])}));
    }

    template <int I1, int I2>
    TokenReference common_shape(const TokenList &inputs) noexcept(false)
    {
        VERIFY(inputs.size() >= I1 && inputs.size() >= I2, "Incorrect number of parameters");

        auto s1 = inputs[I1]->shape();
        auto s2 = inputs[I2]->shape();

        return create<Placeholder>(s1 & s2);
    }

    TokenReference ensure_single_channel(const TokenList &inputs);

    

    #define rect_shape_int constant_shape<int, 4>
    #define rect_shape_float constant_shape<float, 4>

}
