// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file openpose.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include "core/image_net.hpp"
#include "face.hpp"
#include "openpose/core/array.hpp"
#include "openpose/net/bodyPartConnectorBase.hpp"
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace qnn {
namespace vision {

/**
 * @brief cv::Point2f pair for recording body parts from Openpose.
 *
 */
struct BodyStructure {
    /**
     * @brief Construct a new BodyStructure object.
     *
     * @param first The end point of a body part.
     * @param second Second end point of a body part.
     */
    BodyStructure(cv::Point2f &first, cv::Point2f &second) : a(first), b(second) {}
    cv::Point2f a; /**< End point A */
    cv::Point2f b; /**< End point B */
};

/**
 * @brief Info structure for saving the parameters used by Openpose.
 *
 */
struct PoseInfo {
    op::PoseModel pose_model = op::PoseModel::COCO_18; /**< The type of model */
    int midx, npairs, nparts; /**< The pair index values related to model type */

    float multi_scalenet_to_output = 1.f;  /**< The recovery scale size from net output to image */
    bool multi_maximize_positives = false; /**< Get the maximum possible bodies */
    bool detect_multi = true;
    /**
     * @brief Openpose properties.
     * The following properties werer used by qnn.
     * 1. op::PoseProperty::NMSThreshold
     * 2. op::PoseProperty::ConnectInterMinAboveThreshold
     * 3. op::PoseProperty::ConnectInterThreshold
     * 4. op::PoseProperty::ConnectMinSubsetCnt
     * 5. op::PoseProperty::ConnectMinSubsetScore
     *
     */
    std::array<std::atomic<float>, (int)op::PoseProperty::Size> properties;
};

/**
 * @brief Pose buffer used by Openpose.
 *
 */
struct PostBuffer {
    unique_aptr<float> heatmap_uptr; /**< Heatmap buffer */
    unique_aptr<int> kernel_uptr;    /**< Kernel buffer used in nms */
    unique_aptr<float> peakmap_uptr; /**< Peakmap buffer */
};

/**
 * @brief Find human body poses in an image by Openpose
 *
 */
class Openpose : public ImageNet {
    public:
    /**
     * @brief Construct a new Openpose object
     *
     * @param model_path The bmodel path
     * @param dataset_type The dataset used to train the model
     * @param opt Choose to detect multi-poses or a single body.
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    Openpose(const std::string &model_path, const std::string dataset_type, const bool opt,
             QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Set the threshold used by Openpose
     *
     * @param threshold The confidence threshold for Openpose. [default: 0.05]
     */
    void SetThreshold(const float threshold);

    /**
     * @brief Detect the body poses with input a input image.
     *
     * @param image The input image
     * @param keypoints The output body keypoints
     */
    void Detect(const cv::Mat &image, std::vector<std::vector<cv::Point2f>> &keypoints);

    /**
     * @brief Get the skeleton vectors from the output result.
     *
     * @param keypoints Total keypoins from a image
     * @param skeleton The output skeleton vector (0 is kept for easy indexing)
     */
    void GetSkeletonVectors(const std::vector<std::vector<cv::Point2f>> &keypoints,
                            std::vector<std::vector<BodyStructure>> &skeleton);

    /**
     * @brief A simple helper function to draw the body pose on an image.
     *
     * @param keypoints The result from Openpose::Detect
     * @param img The input image
     */
    void Draw(const std::vector<std::vector<cv::Point2f>> &keypoints, cv::Mat &img);

    private:
    /**
     * @brief Get bodies from Inference output
     *
     * @param heatmap_arr The inference output
     * @param W The net width from inference
     * @param H The net height from inference
     * @param channels The channel from inference
     * @param img The image from Openpose::Detect
     * @param keypoints The output body keypoints
     */
    void GetBodies(const float *heatmap_arr, const int W, const int H, const int channels,
                   const cv::Mat &img, std::vector<std::vector<cv::Point2f>> &keypoints);

    /**
     * @brief Resize the heatmap output net size to input net size
     *
     * @param heat_uptr The uptr from PostBuffer
     * @param heatmap_arr The inference output
     * @param origin_w The net width from inference
     * @param origin_h The net height from inference
     * @param origin_c The channel from inference
     * @param target_w The resize target width
     * @param target_h The resize target height
     */
    static void ResizeToNetInput(unique_aptr<float> &heat_uptr, const float *heatmap_arr,
                                 const int &origin_w, const int &origin_h, const int &origin_c,
                                 const int &target_w, const int &target_h);

    /**
     * @brief Get peak map from resized heatmap
     *
     * @param peakmap_uptr The uptr from PostBuffer
     * @param kernel_uptr The uptr from PostBuffer
     * @param multi_scalenet_to_output The scaling ratio for the pose keypoint
     * @param heat_uptr The uptr from PostBuffer
     * @param threshold The confidence threshold
     * @param nparts Nums of body parts exist
     * @param channels The heatmap channel
     * @param max_peaks Equals to heatmap height + 1
     * @param img_w The input image width
     * @param img_h The input image height
     * @param heap_w  The heapmap width
     * @param heap_h The heapmap height
     */
    static void GetPeakMap(unique_aptr<float> &peakmap_uptr, unique_aptr<int> &kernel_uptr,
                           float &multi_scalenet_to_output, const unique_aptr<float> &heat_uptr,
                           const float &threshold, const int &nparts, const int &channels,
                           const int &max_peaks, const int &img_w, const int &img_h,
                           const int &heap_w, const int &heap_h);

    PoseInfo m_info;   /**< Info for Openpose */
    PostBuffer m_pbuf; /**< Post-processing buffers used by GetBodies */
};

}  // namespace vision
}  // namespace qnn
