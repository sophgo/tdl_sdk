// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file anti_facespoofing.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include "anti_facespoofing/afs_classify.hpp"
#include "anti_facespoofing/afs_classify_hsv_ycbcr.hpp"
#include "anti_facespoofing/afs_cv.hpp"
#include "anti_facespoofing/afs_depth.hpp"
#include "anti_facespoofing/afs_optflow.hpp"
#include "anti_facespoofing/afs_patch.hpp"
#include "anti_facespoofing/afs_canny.hpp"
#include "anti_facespoofing/afs_utils.hpp"
#include "mtcnn.hpp"
#include <chrono>

namespace qnn {
namespace vision {

/**
 * @brief The AntiFaceSpoofing class
 *
 */
class AntiFaceSpoofing {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofing object
     *
     * @param depth_model_path model path for depth network
     * @param flow_model_path model path for flow network
     * @param patch_model_path model path for patch network
     * @param classify_model_path model path for classify network
     * @param classify_hsv_ycbcr_model_path model path for classify hsv ycbcr network
     * @param canny_model_path model path for canny network
     * @param qnn_ctx The QNNCtx is an internal cross-model memory manager
     */
    explicit AntiFaceSpoofing(const std::string &depth_model_path,
                              const std::string &flow_model_path,
                              const std::string &patch_model_path,
                              const std::string &classify_model_path,
                              const std::string &classify_hsv_ycbcr_model_path,
                              const std::string &canny_model_path,
                              QNNCtx *qnn_ctx = nullptr);

    /**
     * @brief Use Tracker queue for voting algorithm instead of one single frame.
     *
     * @param flag true to use queue
     */
    void UseTrackerQueue(const bool flag);

    /**
     * @brief Use Weighting to combine the confidence value from each algorithm.
     *
     * @param flag true to use weighting confidence
     */
    void UseWeightingResult(const bool flag, const float threshold) {
        use_weighting_results = flag;
        m_weighting_threshold = threshold;
    }

    /**
     * @brief Turn on and off the filters inside Detect
     *
     * @param cv_bgfilter_flag CV background based filter
     * @param cv_classify_flag NN model based on classification
     * @param cv_sharpfilter_flag CV laplacian based filter
     * @param depth_filter_flag NN model based on depth
     * @param flow_filter_flag NN model based on optical flow
     * @param patch_filter_flag NN model based on patch
     */
    void SetDoFilter(bool cv_bgfilter_flag, bool cv_classify_flag, bool classify_hsv_ycbcr_flag,
                     bool cv_sharpfilter_flag, bool cv_lbpsvm_flag, bool depth_filter_flag,
                     bool flow_filter_flag, bool patch_filter_flag, bool canny_filter_flag);

    /**
     * @brief A function to pass threshold into AntiFaceSpoofingCVsharpness instance
     *
     * @param threshold The threshold for AntiFaceSpoofingCVsharpness to check mtion blurrness
     */
    void SetSharpnessFilterThreshold(uint threshold);

    /**
     * @brief A function to pass LbpSVM model and mean/std value into AntiFaceSpoofingLbpSVM instance
     *
     * @param model path for AntiFaceSpoofingLbpSVM to do Lbp SVM
     */
    void LoadLbpSVMModel(const std::string &model_path, const std::string &mean_path,
                         const std::string &std_path);

    /**
     * @brief Lbp SVM need the landmark to do the face_align
     *
     * @param FaceInfo
     */
    void SetFaceInfo(const face_info_t face_info) { m_face_info = face_info; };

    void SetLbpSVMFilterThreshold(float threshold);
    /**
     * @brief A function to pass threshold into AntiFaceSpoofingClassify instance
     *
     * @param threshold The threshold for AntiFaceSpoofingClassify to check real face
     */
    void SetClassifyFilterThreshold(float threshold);

    /**
     * @brief A function to pass threshold into AntiFaceSpoofingClassifyHSVYCbCr instance
     *
     * @param threshold The threshold for AntiFaceSpoofingCanny to check real face
     */
    void SetClassifyHSVYCbCrFilterThreshold(float threshold);

    /**
     * @brief A function to pass threshold into AntiFaceSpoofingDepth instance
     *
     * @param threshold The threshold for AntiFaceSpoofingDepth to check real face
     */
    void SetDepthFilterThreshold(size_t threshold);

    /**
     * @brief A function to pass threshold into AntiFaceSpoofingOpticalFlow instance
     *
     * @param threshold The threshold for AntiFaceSpoofingOpticalFlow to check real face
     */
    void SetFlowFilterThreshold(float threshold);

    /**
     * @brief A function to pass threshold into AntiFaceSpoofingCanny instance
     *
     * @param threshold The threshold for AntiFaceSpoofingCanny to check real face
     */
    void SetCannyFilterThreshold(float threshold);

    /** @brief Sets of function to pass confidence weighting for cocktail each algo's confidence
     *
     * @param The weight for calculate the weighting, some algo don't have confidence value
     * so didn't open the interface to apply the weighting now. MODIFY it if needed in the future.
     */
    void SetLbpSVMWeighting(float weight) { m_lbpsvmfilter.weighting = weight; };
    void SetClassifyWeighting(float weight) { m_classifyfilter.weighting = weight; };
    void SetClassifyHSVYCbCrWeighting(float weight) { m_classify_hsv_ycbcr_filter.weighting = weight; };
    void SetOptFlowWeighting(float weight) { m_flowfilter.weighting = weight; };
    // void SetBGWeighting(float weight) { m_bgfilter.weighting = weight; };
    // void SetSharpnessWeighting(float weight) { m_sharpfilter.weighting = weight; };
    // void SetPatchWeighting(float weight) { m_patchfilter.weighting = weight; };
    // void SetDepthWeighting(float weight) { m_depthfilter.weighting = weight; };

    /**
     * @brief Detect if the face is a spoofing face. Currently only outputs the first one.
     *
     * @param image The input image
     * @param face_rect The input face info from fd
     * @param spoofing_flags Return false if is a spoofing face
     */
    void Detect(const cv::Mat &image, const std::vector<cv::Rect> &face_rect,
                std::vector<bool> &spoofing_flags);

    /**
     * @brief Get the Depth NN filter visual map output
     *
     * @return cv::Mat The face depth map. Fake faces will be darker.
     */
    cv::Mat GetDepthVisualMat();
    const size_t GetDepthCounterVal();
    const size_t GetPatchCounterVal();
    const uint GetSharpnessVal();
    const float GetConfidenceVal();
    const float GetClassifyHSVYCbCrConfidenceVal();
    const float GetWeightedConfidenceVal();
    const float GetCannyConfidenceVal();
    const float GetLbpSVMResult();

    private:
    cv::Rect squareRectScale(const cv::Rect &rect, const float scale_val);
    cv::Mat cropImage(const cv::Mat &image, const cv::Rect &face_rect);
    bool use_simple_flag_decision = true;
    bool use_weighting_results = false;
    float m_weighting_threshold = 0.0;
    float m_weighted_confidence = 0.0;
    face_info_t m_face_info;
    TrackerQueue m_queue;

    SpoofingFilter<AntiFaceSpoofingCVBG> m_bgfilter;
    SpoofingFilter<AntiFaceSpoofingCVSharpness> m_sharpfilter;
    SpoofingFilter<AntiFaceSpoofingLBPSVM> m_lbpsvmfilter;
    SpoofingFilter<AntiFaceSpoofingClassify> m_classifyfilter;
    SpoofingFilter<AntiFaceSpoofingClassifyHSVYCbCr> m_classify_hsv_ycbcr_filter;
    SpoofingFilter<AntiFaceSpoofingDepth> m_depthfilter;
    SpoofingFilter<AntiFaceSpoofingOpticalFlow> m_flowfilter;
    SpoofingFilter<AntiFaceSpoofingPatch> m_patchfilter;
    SpoofingFilter<AntiFaceSpoofingCanny> m_cannyfilter;
};
}  // namespace vision
}  // namespace qnn