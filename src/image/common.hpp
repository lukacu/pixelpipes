
#pragma once 

#include <pixelpipes/token.hpp>
#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

#include <opencv2/core.hpp>

#define CV_EX_WRAP(S) try { S ; } catch ( cv::Exception& e ) { throw pixelpipes::TypeException(e.what()); }

namespace pixelpipes
{

    class MatImage : public ImageData
    {
    public:
        MatImage(cv::Mat data);

        virtual ~MatImage();

        virtual ImageDepth depth() const;

        virtual size_t width() const;

        virtual size_t height() const;

        virtual size_t channels() const;

        virtual TypeIdentifier backend() const;

        virtual size_t rowstep() const;

        virtual size_t colstep() const;

        virtual size_t element() const;

        virtual unsigned char* data() const;

        static cv::Mat make(const Image image);

        static cv::Mat wrap(const Image image);

        cv::Mat mat;
    };

    template <>
    inline cv::Mat extract(const SharedToken v)
    {
        if (!ImageData::is(v))
            throw TypeException("Not an image value");

        Image image = std::static_pointer_cast<ImageData>(v);

        if (image->backend() == GetTypeIdentifier<cv::Mat>())
        {
            return std::static_pointer_cast<MatImage>(v)->mat;
        }

        return MatImage::wrap(image);
    }

    template <>
    inline cv::Point2f extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type() == FloatType)
        {
            float value = Float::get_value(v);
            return cv::Point2f(value, value);
        }

        if (v->type() == IntegerType)
        {
            int value = Integer::get_value(v);
            return cv::Point2f(value, value);
        }

        if (v->type() != Point2DType)
            throw TypeException("Not a point value");

        Point2D p = std::static_pointer_cast<Point2DVariable>(v)->get();

        return cv::Point2f(p.x, p.y);
    }

    template <>
    inline std::vector<cv::Point2f> extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (!Point2DList::is(v))
            throw TypeException("Not a point list");

        std::vector<Point2D> original = extract<std::vector<Point2D>>(v);

        std::vector<cv::Point2f> convert; convert.reserve(original.size());
        for (auto p : original) {
            convert.push_back(cv::Point2f{p.x, p.y});
        }

        return convert;
    }

    template <>
    inline cv::Matx33f extract(const SharedToken v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->type() != View2DType)
            throw TypeException("Not a view value");

        View2D d = std::static_pointer_cast<View2DVariable>(v)->get();

        cv::Matx33f m{d.m00, d.m01, d.m02, d.m10, d.m11, d.m12, d.m20, d.m21, d.m22};
        return m;
    }

    template <>
    inline SharedToken wrap(const cv::Matx33f v)
    {
        View2D d{v(0, 0), v(0, 1), v(0, 2), v(1, 0), v(1, 1), v(1, 2), v(2, 0), v(2, 1), v(2, 2)};

        return std::make_shared<View2DVariable>(d);
    }

    template <>
    inline SharedToken wrap(const cv::Mat v)
    {
        return std::make_shared<MatImage>(v);
    }

    inline int maximum_value(cv::Mat image)
    {

        switch (image.depth())
        {
        case CV_8U:
            return 255;
        case CV_16S:
            return 255 * 255;
        case CV_32F:
        case CV_64F:
            return 1;
        default:
            throw TypeException("Unsupported image depth");
        }
    }

    int ocv_border_type(BorderStrategy b, int* value);

}