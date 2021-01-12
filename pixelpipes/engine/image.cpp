
#include <opencv2/imgproc.hpp>

#include "types.hpp"
#include "python.hpp"

namespace pixelpipes {

inline int maximum_value(cv::Mat image) {

    switch (image.depth()) {
    case CV_8U:
        return 255;
    case CV_16S:
        return 255 * 255;
    case CV_32F:
    case CV_64F:
        return 1;
    default:
        throw VariableException("Unsupported image depth");
    }
}

inline int interpolate_convert(Interpolation interpolation) {

    switch (interpolation) {
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

/**
 * @brief Apply view linear transformation to an image.
 * 
 */
SharedVariable ViewImage(std::vector<SharedVariable> inputs, ContextHandle context, Interpolation interpolation, BorderStrategy border) {

    if (inputs.size() != 4) 
        throw VariableException("Incorrect number of parameters");

    int border_value = 0;
    int border_const = 0;

    switch (border) {
    case BorderStrategy::ConstantHigh:
        border_const = cv::BORDER_CONSTANT;
        border_value = 1;
        break;
    case BorderStrategy::ConstantLow:
        border_const = cv::BORDER_CONSTANT;
        border_value = 0;
        break;
    case BorderStrategy::Replicate:
        border_const = cv::BORDER_REPLICATE;
        break;
    case BorderStrategy::Reflect:
        border_const = cv::BORDER_REFLECT;
        break;
    case BorderStrategy::Wrap:
        border_const = cv::BORDER_WRAP;
        break;
    default:
        throw VariableException("Illegal border strategy value");
    }

    cv::Mat image = Image::get_value(inputs[0]);
    cv::Matx33f transform = View::get_value(inputs[1]);

    int width = Integer::get_value(inputs[2]);
    int height = Integer::get_value(inputs[3]);

    cv::Mat output;

    int bvalue = maximum_value(image) * border_value;

    try {
        cv::warpPerspective(image, output, transform, cv::Size(width, height), interpolate_convert(interpolation), border_const, bvalue);
    } catch (cv::Exception cve) {
        throw VariableException(cve.what());
    }

    return std::make_shared<Image>(output);

}

REGISTER_OPERATION_FUNCTION(ViewImage, Interpolation, BorderStrategy);


/**
 * @brief Converts depth of an image.
 * 
 */
SharedVariable ConvertDepth(std::vector<SharedVariable> inputs, ContextHandle context, ImageDepth depth) noexcept(false) {

    if (inputs.size() != 1) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);

    double maxin = maximum_value(image);
    int dtype;
    double maxout;

    switch (depth) {
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

    if (image.depth() == dtype) {
        // No conversion required
        return inputs[0];
    }

    cv::Mat result;
    image.convertTo(result, dtype, maxout / maxin);

    return std::make_shared<Image>(result);

}

REGISTER_OPERATION_FUNCTION(ConvertDepth, ImageDepth);


/**
 * @brief Converts color image to grayscale image.
 * 
 */
SharedVariable Grayscale(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);

    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);

    return std::make_shared<Image>(result);

}

REGISTER_OPERATION_FUNCTION(Grayscale);


/**
 * @brief Thresholds an image.
 * 
 */
SharedVariable ThresholdImage(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float threshold = Float::get_value(inputs[1]);

    if (image.channels() != 1)
        throw VariableException("Image has more than one channel");

    float maxval = maximum_value(image);

    cv::Mat result;
    cv::threshold(image, result, threshold, maxval, cv::THRESH_BINARY);

    return std::make_shared<Image>(result);

}

REGISTER_OPERATION_FUNCTION(ThresholdImage);


/**
 * @brief Inverts image pixel values.
 * 
 */
SharedVariable Invert(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 1) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);

    float maxval = maximum_value(image);

    cv::Mat result = maxval - image;

    return std::make_shared<Image>(result);
   
}

REGISTER_OPERATION_FUNCTION(Invert);


/**
 * @brief Returns an image with selected values.
 * 
 */
SharedVariable Equals(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int value = Integer::get_value(inputs[1]);

    if (image.channels() != 1)
        throw VariableException("Image has more than one channel");

    if (image.depth() != CV_8U && image.depth() != CV_16S)
            throw VariableException("Only integer bit types supported");

    cv::Mat result = (image == value);

    return std::make_shared<Image>(result);

}

REGISTER_OPERATION_FUNCTION(Equals);


/**
 * @brief Extracts a single channel from multichannel image.
 * 
 */
SharedVariable Channel(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int index = Integer::get_value(inputs[1]);

    if (index < 0 || index >= image.channels())
        throw VariableException("Wrong channel index, out of bounds");

    cv::Mat result;
    cv::extractChannel(image, result, index);

    return std::make_shared<Image>(result);

}

REGISTER_OPERATION_FUNCTION(Channel);


/**
 * @brief Combines 3 single channel images into color image.
 * 
 */
SharedVariable Merge(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 3) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image_0 = Image::get_value(inputs[0]);
    cv::Mat image_1 = Image::get_value(inputs[1]);
    cv::Mat image_2 = Image::get_value(inputs[2]);

    /*
    SOMETHING WRONG WITH THIS CHECK
    if (image_0.rows == image_1.rows && image_1.rows == image_2.rows)
        throw OperationException("Size does not match", shared_from_this());
    if (image_0.cols == image_1.cols && image_1.cols == image_2.cols)
        throw OperationException("Size does not match", shared_from_this());
    if (image_0.depth() == image_1.depth() && image_1.depth() == image_2.depth())
        throw OperationException("Depth does not match", shared_from_this());
    */
   
    std::vector<cv::Mat> channels;
    cv::Mat result;

    channels.push_back(image_0);
    channels.push_back(image_1);
    channels.push_back(image_2);
    cv::merge(channels, result);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(Merge);


/**
 * @brief Calculates image moments.
 * 
 */
SharedVariable Moments(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);

    if (image.channels() != 1)
        throw VariableException("Image has more than one channel");


    cv::Moments m = cv::moments(image);

    std::vector<float> data{(float)m.m00, (float)m.m01, (float)m.m10, (float)m.m11};

    return std::make_shared<FloatList>(data);
   
}

REGISTER_OPERATION_FUNCTION(Moments);


/**
 * @brief Draw a polygon to a canvas of a given size.
 * 
 */
SharedVariable Polygon(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {
  

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], VariableType::Point), "Not a list of points");

    int width = Integer::get_value(inputs[1]);
    int height = Integer::get_value(inputs[2]);

    std::vector<cv::Point2f> points = List::cast(inputs[0])->elements<cv::Point2f, Point>();

    try {

        std::vector<cv::Point> v(points.begin(), points.end());

        cv::Mat mat = cv::Mat::zeros(cv::Size(height, width), CV_8UC1);

        cv::fillPoly(mat, std::vector<std::vector<cv::Point>>({v}), cv::Scalar(255,255,255));

        return std::make_shared<Image>(mat);

    } catch (cv::Exception cve) {
        throw VariableException(cve.what());
    }

}

REGISTER_OPERATION_FUNCTION(Polygon);


template<typename T>
std::vector<float> bounds(cv::Mat image) {

    int top = std::numeric_limits<int>::max();
    int bottom = std::numeric_limits<int>::lowest();
    int left = std::numeric_limits<int>::max();
    int right = std::numeric_limits<int>::lowest();

    for (int y = 0; y < image.rows; y++) {
        T *p = image.ptr<T>(y);
        for (int x = 0; x < image.cols; x++) {
            if (p[x]) {
                top = std::min(top, y);
                left = std::min(left, x);
                bottom = std::max(bottom, y);
                right = std::max(right , x);
            }
        }
    }

    if (top > bottom)
        return {(float)0, (float)0, (float)image.cols, (float)image.rows};
    else
        return {(float)left, (float)top, (float)right, (float)bottom};

}


/**
 * @brief Creates bounding box with a size of an image.
 * 
 */
SharedVariable MaskBoundingBox(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 1) {
        throw VariableException("Incorrect number of parameters");
    }

    cv::Mat image = Image::get_value(inputs[0]);

    if (image.channels() != 1)
        throw VariableException("Image has more than one channel");

    switch (image.depth()) {
        case CV_8U:
            return std::make_shared<FloatList>(bounds<uint8_t>(image));
        case CV_16S:
            return std::make_shared<FloatList>(bounds<uint16_t>(image));
        case CV_32F:
            return std::make_shared<FloatList>(bounds<float>(image));
        case CV_64F:
            return std::make_shared<FloatList>(bounds<double>(image));
        default:
            throw VariableException("Unsupported image depth");
    }

}

REGISTER_OPERATION_FUNCTION(MaskBoundingBox);


/**
 * @brief Performs image resize.
 * 
 */
SharedVariable ImageResize(std::vector<SharedVariable> inputs, ContextHandle context, Interpolation interpolation) {

    if (inputs.size() == 3) {

        cv::Mat image = Image::get_value(inputs[0]);

        int width = Integer::get_value(inputs[1]);
        int height = Integer::get_value(inputs[2]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(height, width), 0, 0, interpolate_convert(interpolation));

        return std::make_shared<Image>(result);

    }

    if (inputs.size() == 2) {

        cv::Mat image = Image::get_value(inputs[0]);

        int scale = Float::get_value(inputs[1]);

        cv::Mat result;
        cv::resize(image, result, cv::Size(), scale, scale, interpolate_convert(interpolation));

        return std::make_shared<Image>(result);

    }

    throw VariableException("Incorrect number of parameters");
}

REGISTER_OPERATION_FUNCTION(ImageResize, Interpolation);


/**
 * @brief Adds two images.
 * 
 */
SharedVariable ImageAdd(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image_0 = Image::get_value(inputs[0]);
    cv::Mat image_1 = Image::get_value(inputs[1]);

    VERIFY(image_0.channels() == image_1.channels(), "Images must have same number of channels");
    VERIFY(image_0.rows == image_1.rows && image_0.cols == image_1.cols, "Image sizes do not match");

    cv::Mat result;
    cv::add(image_0, image_1, result);
    // cv::add saturates data type

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageAdd);


/**
 * @brief Subtracts two images.
 * 
 */
SharedVariable ImageSubtract(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image_0 = Image::get_value(inputs[0]);
    cv::Mat image_1 = Image::get_value(inputs[1]);

    if (image_0.channels() != image_1.channels())
        throw VariableException("Images must have same number of channels");

    if (image_0.rows != image_1.rows || image_0.cols != image_1.cols)
        throw VariableException("Image sizes do not match");

    cv::Mat result;
    cv::subtract(image_0, image_1, result);
    // cv::subtract saturates data type

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageSubtract);

/**
 * @brief Multiplies image with a multiplier (number).
 * 
 */
SharedVariable ImageMultiply(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float multiplier = Float::get_value(inputs[1]);

    cv::Mat result;
    result = multiplier * image;

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageMultiply);

/**
 * @brief Blends two images using alpha.
 * 
 */
SharedVariable ImageBlend(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 3) {
        throw VariableException("Incorrect number of parameters");
    }

    cv::Mat image_0 = Image::get_value(inputs[0]);
    cv::Mat image_1 = Image::get_value(inputs[1]);
    float alpha = Float::get_value(inputs[2]);  
    float beta = (1 - alpha);

    cv::Mat result;

    cv::addWeighted(image_0, alpha, image_1, beta, 0.0, result);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageBlend);

/*
FILTER AND BLURING OPERATIONS
*/

/**
 * @brief Blurs an image using a gaussian filter.
 * 
 */
SharedVariable GaussianBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 5) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size_x = Integer::get_value(inputs[1]);
    int size_y = Integer::get_value(inputs[2]);
    float sigma_x = Float::get_value(inputs[3]);
    float sigma_y = Float::get_value(inputs[4]);

    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(size_y, size_x), sigma_x, sigma_y, cv::BORDER_REPLICATE);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(GaussianBlur);

/**
 * @brief Blurs an image using a median filter.
 * 
 */
SharedVariable MedianBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    cv::medianBlur(image, result, size);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(MedianBlur);


/**
 * @brief Convolving an image with a normalized box filter.
 * 
 */
SharedVariable AverageBlur(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int size = Integer::get_value(inputs[1]);

    cv::Mat result;
    cv::blur(image, result, cv::Size(size, size));

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(AverageBlur);

/**
 * @brief Applies the bilateral filter to an image.
 * 
 */
SharedVariable BilateralFilter(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 4) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    int d = Integer::get_value(inputs[1]);
    float sigma_color = Float::get_value(inputs[2]);
    float sigma_space = Float::get_value(inputs[3]);

    cv::Mat result;
    cv::bilateralFilter(image, result, d, (double)sigma_color, (double)sigma_space);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(BilateralFilter);


/**
 * @brief Convolves an image with custom kernel.
 * 
 */
SharedVariable ImageFilter(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    cv::Mat kernel = Image::get_value(inputs[1]);

    if (kernel.channels() != 1)
        throw VariableException("Kernel has more than one channel");

    cv::Mat result;
    cv::filter2D(image, result, image.depth(), kernel);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageFilter);

/*
NOISE GENERATION
*/

/**
 * @brief Creates a single channel image with values sampled from normal distribution.
 * 
 */
SharedVariable NormalNoise(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 4) 
        throw VariableException("Incorrect number of parameters");

    int width = Integer::get_value(inputs[0]);
    int height = Integer::get_value(inputs[1]);
    float mean = Float::get_value(inputs[2]);
    float std = Float::get_value(inputs[3]);

	cv::RNG generator(context->random());
	cv::Mat noise(height, width, CV_64F);
	generator.fill(noise, cv::RNG::NORMAL, mean, std);

    return std::make_shared<Image>(noise);
}

REGISTER_OPERATION_FUNCTION(NormalNoise);

/**
 * @brief Creates a single channel image with values sampled from uniform distribution.
 * 
 */
SharedVariable UniformNoise(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {
    
    if (inputs.size() != 4) 
        throw VariableException("Incorrect number of parameters");

    int width = Integer::get_value(inputs[0]);
    int height = Integer::get_value(inputs[1]);
    float min = Float::get_value(inputs[2]);
    float max = Float::get_value(inputs[3]);

	cv::RNG generator(context->random());
	cv::Mat noise(height, width, CV_64F);
	generator.fill(noise, cv::RNG::UNIFORM, min, max);

    return std::make_shared<Image>(noise);
}

REGISTER_OPERATION_FUNCTION(UniformNoise);

/*
OTHER OPERATIONS
*/

/**
 * @brief Sets image pixels to zero with probability P.
 * 
 */
SharedVariable ImageDropout(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 2) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float dropout_p = Float::get_value(inputs[1]);

	cv::RNG generator(context->random());
    cv::Mat result = image.clone();

    if (result.channels() == 1) {
        for(int y = 0; y < result.rows; y++){
            for(int x = 0; x < result.cols; x++){  
                if (generator.uniform(0.0, 1.0) < dropout_p) {                  
                    if (result.depth() == CV_8U) {
                        result.at<uchar>(y,x) = 0;                   
                    }
                    else if (result.depth() == CV_32F) {
                        result.at<float>(y,x) = 0.0;                   
                    }
                    else if (result.depth() == CV_64F) {
                        result.at<double>(y,x) = 0.0;                   
                    }                    
                }                      
            }
        }
    }
    else if (result.channels() == 3) {
        for(int y = 0; y < result.rows; y++){
            for(int x = 0; x < result.cols; x++){  
                if (generator.uniform(0.0, 1.0) < dropout_p) {                  
                    if (result.depth() == CV_8U) {
                        cv::Vec3b & color = result.at<cv::Vec3b>(y,x);
                        color[0] = 0;
                        color[1] = 0;
                        color[2] = 0;                  
                    }
                    else if (result.depth() == CV_32F) {
                        cv::Vec3f & color = result.at<cv::Vec3f>(y,x);
                        color[0] = 0.0;
                        color[1] = 0.0;
                        color[2] = 0.0;                    
                    }
                    else if (result.depth() == CV_64F) {
                        cv::Vec3d & color = result.at<cv::Vec3d>(y,x);
                        color[0] = 0.0;
                        color[1] = 0.0;
                        color[2] = 0.0;                    
                    }                    
                }                      
            }
        }
    }

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageDropout);


/**
 * @brief Divides image to pacthes and sets patch pixels to zero with probability P.
 * 
 */
SharedVariable ImageCoarseDropout(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    if (inputs.size() != 3) 
        throw VariableException("Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float dropout_p = Float::get_value(inputs[1]);
    float dropout_size = Float::get_value(inputs[2]);

    cv::Mat result = image.clone();

    int patch_size_x = (int) (result.cols * dropout_size);
    int patch_size_y = (int) (result.rows * dropout_size);
    int num_patches_x = (int) (1 / dropout_size);
    int num_patches_y = (int) (1 / dropout_size);

    cv::RNG generator(context->random());

    if (result.channels() == 1) {
        for (int yp = 0; yp < num_patches_y; yp++) {
            for (int xp = 0; xp < num_patches_x; xp++) {
                if (generator.uniform(0.0, 1.0) < dropout_p){
                    for (int y = 0; y < patch_size_y; y++) {
                        for (int x = 0; x < patch_size_x; x++) {

                            int iy = y + yp * patch_size_y;
                            int ix = x + xp * patch_size_x;

                            if (result.depth()  == CV_8U) {
                                result.at<uchar>(iy,ix) = 0;                   
                            }
                            else if (result.depth() == CV_32F) {
                                result.at<float>(iy,ix) = 0.0;                   
                            }
                            else if (result.depth() == CV_64F) {
                                result.at<double>(iy,ix) = 0.0;                   
                            }   
                        }
                    }
                }
            }
        }
    }

    else if (result.channels() == 3) {
        for (int yp = 0; yp < num_patches_y; yp++) {
            for (int xp = 0; xp < num_patches_x; xp++) {
                if (generator.uniform(0.0, 1.0) < dropout_p){
                    for (int y = 0; y < patch_size_y; y++) {
                        for (int x = 0; x < patch_size_x; x++) {

                            int iy = y + yp * patch_size_y;
                            int ix = x + xp * patch_size_x; 

                            if (result.depth() == CV_8U) {
                                cv::Vec3b & color = result.at<cv::Vec3b>(iy,ix);
                                color[0] = 0;
                                color[1] = 0;
                                color[2] = 0;                  
                            }
                            else if (result.depth() == CV_32F) {
                                cv::Vec3f & color = result.at<cv::Vec3f>(iy,ix);
                                color[0] = 0.0;
                                color[1] = 0.0;
                                color[2] = 0.0;                    
                            }
                            else if (result.depth() == CV_64F) {
                                cv::Vec3d & color = result.at<cv::Vec3d>(iy,ix);
                                color[0] = 0.0;
                                color[1] = 0.0;
                                color[2] = 0.0;                    
                            }    
                        }
                    }           
                }                           
            }
        }
    }

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageCoarseDropout);


/**
 * @brief Returns a bounding box of custom size.
 * 
 */
SharedVariable RegionBoundingBox(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() != 4) {
        throw VariableException("Incorrect number of parameters");
    }

    float top = Float::get_value(inputs[1]);    
    float bottom = Float::get_value(inputs[2]);  
    float left = Float::get_value(inputs[3]);    
    float right = Float::get_value(inputs[4]);

    if (top != bottom){
        if (top > bottom){
            float temp = top;
            top = bottom;
            bottom = temp;
        }
    }
    else
        throw VariableException("Invalid bounding box coordinates");

    if (left != right){
        if (left > right){
            float temp = left;
            left = right;
            right = left;
        }
    }
    else
        throw VariableException("Invalid bounding box coordinates");
    
    std::vector<float> b_box = { (float)left, (float)top, (float)right, (float)bottom};

    std::vector<float> bbox = {left, top, right, bottom};

    return std::make_shared<FloatList>(bbox);
}

REGISTER_OPERATION_FUNCTION(RegionBoundingBox);


/**
 * @brief Cuts region form an image defined by the bounding box.
 * 
 */
SharedVariable ImageCut(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[1], VariableType::Float), "Not a float list");

    cv::Mat image = Image::get_value(inputs[0]);
    auto bbox = std::static_pointer_cast<List>(inputs[1]);

    float left = Float::get_value(bbox->get(0));
    float top = Float::get_value(bbox->get(1));
    float right = Float::get_value(bbox->get(2));
    float bottom = Float::get_value(bbox->get(3));

    cv::Mat result = image.clone();

    if (result.channels() == 1) {
        for (int y = (int) top; y < (int) bottom; y++) {
            for (int x = (int) left; x < (int) right; x++) {
                if (result.depth() == CV_8U) {
                    result.at<uchar>(y,x) = 0;                   
                }
                else if (result.depth() == CV_32F) {
                    result.at<float>(y,x) = 0.0;                   
                }
                else if (result.depth() == CV_64F) {
                    result.at<double>(y,x) = 0.0;                   
                }  
            }
        }
    }

    else if (result.channels() == 3) {
        for (int y = (int) top; y < (int) bottom; y++) {
            for (int x = (int) left; x < (int) right; x++) {
                if (result.depth() == CV_8U) {
                    cv::Vec3b & color = result.at<cv::Vec3b>(y,x);
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 0;                    
                }  
                else if (result.depth() == CV_32F) {
                    cv::Vec3f & color = result.at<cv::Vec3f>(y,x);
                    color[0] = 0.0;
                    color[1] = 0.0;
                    color[2] = 0.0;                    
                }  
                else if (result.depth() == CV_64F) {
                    cv::Vec3d & color = result.at<cv::Vec3d>(y,x);
                    color[0] = 0.0;
                    color[1] = 0.0;
                    color[2] = 0.0;                    
                }  
            }
        }
    }

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageCut);


/**
 * @brief Inverts all values above a threshold in image.
 * 
 */
SharedVariable ImageSolarize(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    
    VERIFY(image.channels() == 1, "Image has more than one channel");

    float threshold = Float::get_value(inputs[1]);  
    float max = maximum_value(image);

    cv::Mat result = image.clone();

    if (result.channels() == 1) {
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                if (result.depth() == CV_8U) {
                    if (result.at<uchar>(y,x) > (int) threshold){
                        result.at<uchar>(y,x) = max - result.at<uchar>(y,x);
                    }
                }
                else if (result.depth() == CV_32F) {
                    if (result.at<float>(y,x) > threshold){
                        result.at<float>(y,x) = max - result.at<float>(y,x);
                    }               
                }
                else if (result.depth() == CV_64F) {
                    if (result.at<double>(y,x) > threshold){
                        result.at<double>(y,x) = max - result.at<double>(y,x);
                    }               
                }
            }
        }
    }

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION(ImageSolarize);


}