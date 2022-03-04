#include <pixelpipes/image.hpp>

#include "common.hpp"

namespace pixelpipes {

/**
 * @brief Blends two images using alpha.
 * 
 */
SharedVariable ImageBlend(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    cv::Mat image_0 = Image::get_value(inputs[0]);
    cv::Mat image_1 = Image::get_value(inputs[1]);
    float alpha = Float::get_value(inputs[2]);  
    float beta = (1 - alpha);

    cv::Mat result;

    cv::addWeighted(image_0, alpha, image_1, beta, 0.0, result);

    return std::make_shared<Image>(result);
}

REGISTER_OPERATION_FUNCTION("blend", ImageBlend);

/**
 * @brief Sets image pixels to zero with probability P.
 * 
 */
SharedVariable ImageDropout(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 3, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float dropout_p = Float::get_value(inputs[1]);

	cv::RNG generator(Integer::get_value(inputs[2]));
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

REGISTER_OPERATION_FUNCTION("dropout", ImageDropout);


/**
 * @brief Divides image to pacthes and sets patch pixels to zero with probability P.
 * 
 */
SharedVariable ImageCoarseDropout(std::vector<SharedVariable> inputs) noexcept(false) {

    VERIFY(inputs.size() == 4, "Incorrect number of parameters");

    cv::Mat image = Image::get_value(inputs[0]);
    float dropout_p = Float::get_value(inputs[1]);
    float dropout_size = Float::get_value(inputs[2]);

    cv::Mat result = image.clone();

    int patch_size_x = (int) (result.cols * dropout_size);
    int patch_size_y = (int) (result.rows * dropout_size);
    int num_patches_x = (int) (1 / dropout_size);
    int num_patches_y = (int) (1 / dropout_size);

    cv::RNG generator(Integer::get_value(inputs[3]));

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

REGISTER_OPERATION_FUNCTION("coarse_dropout", ImageCoarseDropout);

/**
 * @brief Cuts region form an image defined by the bounding box.
 * 
 */
SharedVariable ImageCut(std::vector<SharedVariable> inputs) {

    VERIFY(inputs.size() == 2, "Incorrect number of parameters");
    VERIFY(List::is_list(inputs[1], FloatType), "Not a float list");

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

REGISTER_OPERATION_FUNCTION("cut", ImageCut);


/**
 * @brief Inverts all values above a threshold in image.
 * 
 */
SharedVariable ImageSolarize(std::vector<SharedVariable> inputs) {

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

REGISTER_OPERATION_FUNCTION("solarize", ImageSolarize);


}