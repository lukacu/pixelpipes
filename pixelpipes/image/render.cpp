
#include <opencv2/imgproc.hpp>

#include <pixelpipes/image.hpp>
#include <pixelpipes/geometry.hpp>

namespace pixelpipes {

/**
 * @brief Draw a polygon to a canvas of a given size.
 * 
 */
SharedVariable Polygon(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {
  

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[0], Point2DType), "Not a list of points");

    int width = Integer::get_value(inputs[1]);
    int height = Integer::get_value(inputs[2]);

    std::vector<cv::Point2f> points = List::cast(inputs[0])->elements<cv::Point2f>();

    try {

        std::vector<cv::Point> v(points.begin(), points.end());

        cv::Mat mat = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);

        cv::fillPoly(mat, std::vector<std::vector<cv::Point>>({v}), cv::Scalar(255,255,255));

        return std::make_shared<Image>(mat);

    } catch (cv::Exception& cve) {
        throw VariableException(cve.what());
    }

}

REGISTER_OPERATION_FUNCTION("polygon", Polygon);


/*
NOISE GENERATION
*/

/**
 * @brief Creates a single channel image with values sampled from normal distribution.
 * 
 */
SharedVariable NormalNoise(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {

    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int width = Integer::get_value(inputs[0]);
    int height = Integer::get_value(inputs[1]);
    float mean = Float::get_value(inputs[2]);
    float std = Float::get_value(inputs[3]);

	cv::RNG generator(context->random());
	cv::Mat noise(height, width, CV_64F);
	generator.fill(noise, cv::RNG::NORMAL, mean, std);

    return std::make_shared<Image>(noise);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("normal_noise", NormalNoise, StohasticOperation);

/**
 * @brief Creates a single channel image with values sampled from uniform distribution.
 * 
 */
SharedVariable UniformNoise(std::vector<SharedVariable> inputs, ContextHandle context) noexcept(false) {
    
    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int width = Integer::get_value(inputs[0]);
    int height = Integer::get_value(inputs[1]);
    float min = Float::get_value(inputs[2]);
    float max = Float::get_value(inputs[3]);

	cv::RNG generator(context->random());
	cv::Mat noise(height, width, CV_64F);
	generator.fill(noise, cv::RNG::UNIFORM, min, max);

    return std::make_shared<Image>(noise);
}

REGISTER_OPERATION_FUNCTION_WITH_BASE("uniform_noise", UniformNoise, StohasticOperation);


SharedVariable Linear(std::vector<SharedVariable> inputs, ContextHandle context, bool flip) noexcept(false) {
    
    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    int width = Integer::get_value(inputs[0]);
    int height = Integer::get_value(inputs[1]);

    float min = Float::get_value(inputs[2]);
    float max = Float::get_value(inputs[3]);

	cv::Mat result(height, width, CV_64F);

    double *data = &result.at<double>(0, 0);

    if (flip) {
        for(int y = 0; y < result.rows; y++){
            float v = ((float) y / (float) (result.rows-1)) * (max - min) + min;
            for(int x = 0; x < result.cols; x++){  
                *data = v;
                data++;
            }
        }

    } else {

        for(int y = 0; y < result.rows; y++){
            for(int x = 0; x < result.cols; x++){  
                *data = ((float) x / (float) (result.cols-1)) * (max - min) + min;
                data++;
            }
        }
    }

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("linear", Linear, bool);




}