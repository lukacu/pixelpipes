
#if defined(__clang__) && defined(__has_warning)
# if __has_warning( "-Wc11-extensions" )
#  pragma clang diagnostic ignored "-Wc11-extensions"
# endif
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

    enum class ImageDepth
    {
        Byte,
        Short,
        Integer,
        Float
    };

    PIXELPIPES_CONVERT_ENUM(Interpolation)
    PIXELPIPES_CONVERT_ENUM(BorderStrategy)
    PIXELPIPES_CONVERT_ENUM(ImageDepth)

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
        return cv::Rect((int) r.left, (int) r.top, (int) (r.right - r.left), (int) (r.bottom -  r.top));
    }

    template <>
    inline TokenReference wrap(const std::vector<cv::Point2f> v)
    {

        std::vector<Point2D> convert;
        convert.reserve(v.size());
        for (auto p : v)
        {
            convert.push_back(Point2D{p.x, p.y});
        }

        return wrap(convert);
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

        virtual ~MatImage() = default;

        virtual Shape shape() const;

        virtual size_t length() const;

        virtual size_t size() const;

        virtual TokenReference get(size_t i) const;

        virtual TokenReference get(const Sizes &i) const;

        virtual ReadonlySliceIterator read_slices() const;

        virtual WriteableSliceIterator write_slices();

        virtual const uchar *const_data() const;

        virtual uchar *data();

        virtual SizeSequence strides() const;

        virtual cv::Mat get() const;

        virtual size_t cell_size() const;

        static cv::Mat copy(const TensorReference &tensor);

        static cv::Mat wrap(const TensorReference &tensor);

    private:
        cv::Mat _mat;
        SizeSequence _shape;
        SizeSequence _strides;
        TypeIdentifier _element;
    };

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
            return MatImage::wrap(cast<Tensor>(v));
        }

        throw TypeException("Not an image value");
    }

    template <>
    inline TokenReference wrap(const cv::Mat v)
    {
        return create<MatImage>(v);
    }

    inline int maximum_value(cv::Mat image)
    {

        switch (image.depth())
        {
        case CV_8U:
            return 255;
        case CV_16U:
            return 255 * 255;
        case CV_32S:
            return 255 * 255 * 255 * 128;
        case CV_32F:
            return 1;
        default:
            throw TypeException("Unsupported image depth");
        }
    }

    int ocv_border_type(BorderStrategy b, int *value);

}
