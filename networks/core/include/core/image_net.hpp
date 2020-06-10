// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file image_net.hpp
 * @ingroup NETWORKS_CORE
 */
#pragma once
#include "base_net.hpp"
#include <functional>

namespace qnn {
namespace vision {

/**
 * @brief Structure of Mat buffer
 *
 * Note: cv::Mat on bm1880 uses ion memory.
 */
struct MatBuffer {
    MatBuffer(std::string name) { this->name = name; }
    std::string name; /**< A short purpose description */
    cv::Mat img;      /**< The cv::Mat buffer */
};

/**
 * @brief Structure of int8 image quantize parameters
 *
 */
struct ImageQuantizeParameters {
    ImageQuantizeParameters(){};
    /**
     * @brief Construct a new ImageQuantizeParameters object. The short vector will be padded
     * using the last element to match the length of the longer one.
     *
     * @param mean The mean vector
     * @param scales The scale vector
     */
    ImageQuantizeParameters(const std::vector<float> mean, const std::vector<float> scales)
        : input_scales(scales), mean_values(mean) {
        if (input_scales.size() == 0 || mean_values.size() == 0) {
            LOGE << "Vectors must not be empty! Aborting...";
            exit(-1);
        } else if (mean_values.size() != input_scales.size()) {
            LOGW << "Warning! Mean size is not the same as scale size. Will duplicate the last "
                    "element of the short vector to match the length of the long one.";

            if (mean_values.size() > input_scales.size()) {
                PadToSameLength(mean_values, input_scales);
            } else {
                PadToSameLength(input_scales, mean_values);
            }
        }
    }
    std::vector<float> input_scales; /**< The input threshold */
    std::vector<float> mean_values;  /**< The mean values */
    private:
    /**
     * @brief Inline function that pads the short vector to match the length of the longer one with
     * last element.
     *
     * @param longer Longer vector
     * @param shorter Shorter vector
     */
    inline void PadToSameLength(std::vector<float>& longer, std::vector<float>& shorter) {
        auto value = shorter[shorter.size() - 1];
        for (size_t i = shorter.size(); i < longer.size(); i++) {
            shorter.push_back(value);
        }
    }
};

typedef std::function<void(OutTensors&, std::vector<float>&, int, int)> ParseFunc;

/**
 * @brief The base class of image related networks
 *
 */
class ImageNet : public BaseNet {
    public:
    /**
     * @brief Resize image with right and bottom padded
     *
     * @param src The input image
     * @param dst The output image
     * @param width The desired output width
     * @param height The desired output height
     * @param buf The MatBuffer structure for cv::resize and copymakeborder from acquiring new
     * space from ion
     * @param resize_policy The resize method for OpenCV
     * @param preserve_ratio Preserve image ratio
     * @return float ratio = output_img / input_img
     */
    static float ResizeImage(const cv::Mat& src, cv::Mat& dst, int width, int height,
                             std::vector<MatBuffer>& buf, int resize_policy,
                             bool preserve_ratio = true);

    /**
     * @brief Resize image with 4 direction padded
     *
     * @param src The input image
     * @param dst The output image
     * @param width The desired output width
     * @param height The desired output height
     * @param buf The MatBuffer structure for cv::resize and copymakeborder from acquiring new
     * space from ion
     * @param resize_policy The resize method for OpenCV
     * @param preserve_ratio Preserve image ratio
     * @return float ratio = output_img / input_img
     */
    static float ResizeImageCenter(const cv::Mat& src, cv::Mat& dst, int width, int height,
                                   std::vector<MatBuffer>& buf, int resize_policy,
                                   bool preserve_ratio = true, cv::Scalar scalar = cv::Scalar(0));

    /**
     * @brief Set the resize image width, height.
     *
     * @param w The image width
     * @param h The image height
     */
    void SetNetWH(int w, int h) {
        net_width = w;
        net_height = h;
    }

    protected:
    // FIXME: This might change if the bmtap2's API changed
    /**
     * @brief Construct a new ImageNet object
     *
     * @param model_path The bmodel path
     * @param supported_input_shapes The supported input shapes of the bmodel
     * @param channel The net channel
     * @param width The resize image width
     * @param height The resize image height
     * @param preserve_ratio Preserve ratio when resizing images
     * @param resize_policy Change the resize policy. e.g. INTER_LINEAR, INTER_NEAREST.
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager.
     */
    ImageNet(const std::string& model_path, const std::vector<NetShape>& supported_input_shapes,
             int channel, int width, int height, bool preserve_ratio, int resize_policy,
             QNNCtx* qnn_ctx = nullptr);

    /**
     * @brief Detect images
     *
     * @param images The input images
     * @param parse Function parser (std::function)
     */
    void Detect(const std::vector<cv::Mat>& images, ParseFunc parse);

    /**
     * @brief Image preprocessing operations. Note: This should be able to override cause tranpose
     * only can be done
     * after resize.
     *
     * @param image The input image
     * @param prepared The output image
     * @param ratio The output ratio
     */
    virtual void PrepareImage(const cv::Mat& image, cv::Mat& prepared, float& ratio);
    /**
     * @brief Set the quantize parameter values.
     *
     * @param params The input ImageQuantizeParameters struct
     */
    void SetQuanParams(const ImageQuantizeParameters& params) { quan_parms = params; }

    /**
     * @brief Get the private QuanParams reference object
     *
     * @return const ImageQuantizeParameters& return const quan_parms in private member
     */
    const ImageQuantizeParameters& GetRefQuanParams() const { return quan_parms; }

    // These parameters are moved to protected due to virtual function PrepareImages.
    int net_channel;     /**< The net input channel */
    int net_width;       /**< The resize width */
    int net_height;      /**< The resize height */
    bool preserve_ratio; /**< Preserve ratio when resize */
    int m_resize_policy; /**< The resize policy. e.g. INTER_LINEAR, INTER_NEAREST */

    std::vector<MatBuffer>
        m_ibuf; /**< The internal buffer vector for function ResizeImage/ ResizeImageCenter */

    private:
    /**
     * @brief Where images are quantized and splitted into r, g, b Mats.
     *
     * @param image The input image
     * @param channels The output images
     */
    void PreProcessImage(cv::Mat& image, std::vector<cv::Mat>& channels);

    ImageQuantizeParameters quan_parms; /**< The quantize parameters */
};

}  // namespace vision
}  // namespace qnn
