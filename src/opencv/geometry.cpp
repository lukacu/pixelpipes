#include "common.hpp"

#include <opencv2/imgproc.hpp>
 
namespace pixelpipes
{

    inline int interpolate_convert(Interpolation interpolation)
    {

        switch (interpolation)
        {
        case Interpolation::Linear:
            return cv::INTER_LINEAR;
        case Interpolation::Area:
            return cv::INTER_AREA;
        case Interpolation::Cubic:
            return cv::INTER_CUBIC;
        case Interpolation::Lanczos:
            return cv::INTER_LANCZOS4;
        default:
            return cv::INTER_NEAREST;
        }
    }

    cv::Mat transpose(const cv::Mat &image) noexcept(false)
    {
        cv::Mat result;
        cv::transpose(image, result);
        return result;
    }

    TokenReference _transpose_eval(const TokenList& tokens)
    {
        VERIFY(tokens.size() == 1, "Transpose requires one argument");
        auto shape = tokens[0]->shape();
        return create<Placeholder>(Shape(shape.element(), {shape[0], shape[1]}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("transpose", transpose, _transpose_eval);

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    cv::Mat view(const cv::Mat& image, const cv::Matx33f transform, int width, int height, Interpolation interpolation, BorderStrategy border)
    {
        int border_value = 0;
        int border_const = 0;

        border_const = ocv_border_type(border, &border_value);

        cv::Mat output;
        int bvalue = maximum_value(image) * border_value;

        CV_EX_WRAP(cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate_convert(interpolation), border_const, bvalue));
        
        return output;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("view", view, (given_shape<2, 3, FloatType>));

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    cv::Mat remap(const cv::Mat& image, const cv::Mat& x, const cv::Mat& y, Interpolation interpolation, BorderStrategy border)
    {

        int border_value = 0;
        int border_const = 0;

        border_const = ocv_border_type(border, &border_value);

        cv::Mat output;

        int bvalue = maximum_value(image) * border_value;

        CV_EX_WRAP(cv::remap(image, output, x, y, interpolate_convert(interpolation), border_const, bvalue));
        
        return output;
    }

    TokenReference _remap_eval(const TokenList& tokens)
    {
        VERIFY(tokens.size() == 5, "Remap requires five arguments");
        auto ishape = tokens[0]->shape();
        auto xshape = tokens[1]->shape();
        return create<Placeholder>(Shape(ishape.element(), {xshape[0], xshape[1]}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("remap", remap, _remap_eval);

    template <typename T>
    Rectangle bounds(const cv::Mat& image)
    {

        int top = std::numeric_limits<int>::max();
        int bottom = std::numeric_limits<int>::lowest();
        int left = std::numeric_limits<int>::max();
        int right = std::numeric_limits<int>::lowest();

        for (int y = 0; y < image.rows; y++)
        {
            const T *p = image.ptr<T>(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (p[x])
                {
                    top = std::min(top, y);
                    left = std::min(left, x);
                    bottom = std::max(bottom, y);
                    right = std::max(right, x);
                }
            }
        }

        if (top > bottom)
            return {(float)0, (float)0, (float)image.cols, (float)image.rows};
        else
            return {(float)left, (float)top, (float)right + 1, (float)bottom + 1};
    }

    Rectangle mask_bounds(const cv::Mat& image) noexcept(false)
    {
        VERIFY(image.channels() == 1, "Image has more than one channel");

        // float maxval = maximum_value(image); TOD: what to do with this?

        switch (image.depth())
        {
        case CV_8U:
            return bounds<uchar>(image);
        case CV_8S:
            return bounds<uchar>(image);
        case CV_16S:
            return bounds<short>(image);
        case CV_16U:
            return bounds<ushort>(image);
        case CV_32F:
            return bounds<float>(image);
        case CV_64F:
            return bounds<double>(image);
        default:
            throw TypeException("Unsupported depth");
        }
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("mask_bounds", mask_bounds, rect_shape_int);

    /**
     * @brief Performs image resize.
     *
     * TODO: split into two operations
     *
     */
    cv::Mat resize(const cv::Mat &image, int width, int height, Interpolation interpolation)
    {

        cv::Mat result;
        cv::resize(image, result, cv::Size(width, height), 0, 0, interpolate_convert(interpolation));

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("resize", resize, (given_shape_type<1, 2, 0>));

    cv::Mat rescale(const cv::Mat &image, float scale, Interpolation interpolation)
    {

        cv::Mat result;
        cv::resize(image, result, cv::Size(), scale, scale, interpolate_convert(interpolation));

        return result;
    }

    TokenReference _rescale_eval(const TokenList& tokens)
    {
        VERIFY(tokens.size() == 3, "Rescale requires three arguments");
        auto shape = tokens[0]->shape();
        auto scale = extract<float>(tokens[1]);
        return create<Placeholder>(Shape(shape.element(), {shape[0] * scale, shape[1] * scale}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("rescale", rescale, _rescale_eval);

    /**
     * @brief Flips a 2D array around vertical, horizontal, or both axes.
     *
     */
    TokenReference flip(const cv::Mat& image, bool horizontal, bool vertical) noexcept(false)
    {

        TensorReference result = create_tensor(image);

        cv::Mat out = wrap_tensor(result);

        if (horizontal)
        {
            if (vertical)
            {
                cv::flip(image, out, -1);
            }
            else
            {
                cv::flip(image, out, 1);
            }
        }
        else
        {
            if (!vertical)
            {
                cv::copyTo(image, out, cv::Mat());
            }
            else
            {
                cv::flip(image, out, 0);
            }
        }

        return result;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("flip", flip, (forward_shape<0>));

    /**
     * @brief Returns a bounding box of custom size.
     *
     */
    cv::Mat crop(const cv::Mat &image, int x, int y, int width, int height) noexcept(false)
    {
        cv::Rect bbox(x, y, width, height);
        // Compute intersection of the bounding box with the image.
        bbox &= cv::Rect(0, 0, image.cols, image.rows);
        try                                            
        {               
            // Crop image to the bounding box. The output is an image of the same size as the bounding box.                               
            return image(bbox).clone();                                  
        }                                              
        catch (cv::Exception & e)                      
        {                                              
            throw pixelpipes::TypeException(e.what()); 
        }

    }

    TokenReference _crop_eval(const TokenList& tokens)
    {
        VERIFY(tokens.size() == 5, "Crop requires five arguments");
        auto shape = tokens[0]->shape();
        auto x = extract<int>(tokens[1]);
        auto y = extract<int>(tokens[2]);
        auto width = extract<int>(tokens[3]);
        auto height = extract<int>(tokens[4]);
        cv::Rect bbox(x, y, width, height);
        bbox &= cv::Rect(0, 0, shape[1], shape[0]);
        return create<Placeholder>(Shape(shape.element(), {bbox.width, bbox.height}));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("crop", crop, _crop_eval);

    cv::Mat crop_safe(const cv::Mat &image, int x, int y, int width, int height, BorderStrategy border) noexcept(false)
    { 
        int border_value = 0;
        int border_const = 0;

        border_const = ocv_border_type(border, &border_value);

        VERIFY(width > 0 && height > 0, "Width and height must be positive");

        cv::Rect bbox(x, y, width, height);
        try                                            
        {    
            // If the bounding box is in the image, just crop it.
            if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= image.cols && bbox.y + bbox.height <= image.rows)
            {
                return image(bbox).clone();
            }

            // Crop image to the bounding box. The output is an image of the same size as the bounding box. 
            // If the bounding box is larger than the image, the output is padded according to the border strategy.
            cv::Mat result;

            // Calculate overlap of the bounding box with the image.
            cv::Rect overlap = bbox & cv::Rect(0, 0, image.cols, image.rows);

            // Copy the content of the overlap to the result, the rest is padded.
            cv::copyMakeBorder(image(overlap), result, overlap.y, bbox.y + bbox.height - overlap.y - image.rows, overlap.x, bbox.x + bbox.width - overlap.x - image.cols, border_const, border_value);

            return result;
        }
        catch (cv::Exception & e)                      
        {                                              
            throw pixelpipes::TypeException(e.what()); 
        }
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("crop_safe", crop_safe, (given_shape_type<3, 4, 0>));

}
