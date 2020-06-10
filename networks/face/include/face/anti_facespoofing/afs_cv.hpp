// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/**
 * @file anti_facespoofing.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "face/face.hpp"
#include "LibLinear.hpp"

#include <vector>
#include <bitset>
#include <chrono>

namespace qnn {
namespace vision {

/**
 * @brief Background motion detection parameters
 *
 */
struct BGDetectParam {
    BGDetectParam() {
        kerenl_erode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        kerenl_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    }
    cv::Mat prev_frame;
    bool motion = false;
    bool skip = false;
    float active_period = 0.5;
    int bg_update_state = 0;
    std::chrono::time_point<std::chrono::steady_clock> bg_start_freeze_time;
    float bg_resume_update_time = 5;  // sec

    int face_vanish_count = 0;
    int face_vanish_threshold = 5;

    cv::Mat kerenl_erode;
    cv::Mat kerenl_dilate;
};

class AntiFaceSpoofingCVBG {
    public:
    /**
     * @brief Construct a new AntiFaceSpoofingCVBG object
     *
     */
    AntiFaceSpoofingCVBG();

    /**
     * @brief OpenCV based background detect filter
     *
     * @param image The input frame
     * @param face_rect The input face info from fd
     * @return int Return 1 if is a real face
     */
    int BGDetect(const cv::Mat &image, const std::vector<cv::Rect> &face_rect);

    private:
    /**
     * @brief Update background and produce frame diff from updated background
     *
     * @param img Input frame
     * @param lr_rate Learning rate
     * @param blur_img Blurred image
     * @param morph_img Morphology diff image
     */
    void FrameDiff(const cv::Mat &img, const float &lr_rate, cv::Mat &blur_img, cv::Mat &morph_img);

    /**
     * @brief Detect motion of right, left, and top of a face. Return false if 2 or more blocks have
     * motion detected
     *
     * @param ds_img Scaled image
     * @param morph_img Morphology image from FrameDiff
     * @param ds_rect The input face info from fd
     * @return int Return 1 if is a real face
     */
    int HeadForegroundDetect(const cv::Mat &ds_img, const cv::Mat &morph_img,
                             const std::vector<cv::Rect> &ds_rect);

    /**
     * @brief Detect motions with contours, contours must smaller than face_rect
     *
     * @param ds_img Scaled image
     * @param blur_img Blurred image from FrameDiff
     * @param morph_img Morphology image from FrameDiff
     * @param ds_rect The input face info from fd
     * @return int Return 1 if is a real face
     */
    int FrameRectDetect(const cv::Mat &ds_img, const cv::Mat &blur_img, const cv::Mat &morph_img,
                        const std::vector<cv::Rect> &ds_rect);

    BGDetectParam m_bparam; /**< BG params */
};

class AntiFaceSpoofingCVSharpness {
    public:
    /**
     * @brief Detect face image motion (blurrness).
     *
     * @param img The input face
     * @return true Motion is lower than threshold
     * @return false Motion is higher than threhold
     */
    bool Detect(cv::Mat &img);

    /**
     * @brief Set the blurrness threshold. Smaller, more blur.
     *
     * @param value The input threshold
     */
    void SetThreshold(uint value);

    /**
     * @brief Get the blurrness threshold.
     *
     * @return threshold
     */
    const uint GetThreshold();

    /**
     * @brief Get the raw compute value.
     *
     * @return const uint The blurr value
     */
    const uint GetSharpnessVal();

    private:
    /**
     * @brief Get the Blurrness value from given image with Laplacian
     *
     * @param img Input image
     * @return const uint Return blurrness value
     */
    const uint GetSharpnessLevel(cv::Mat &img);

    uint m_sharpness_threshold = 200; /**< The sharpness threshold */
    uint m_sharpness_val = 0;         /**< The sharpness value of the processed frame */
};

class AntiFaceSpoofingLBPSVM {
    public:
    AntiFaceSpoofingLBPSVM() {}
    ~AntiFaceSpoofingLBPSVM();
    AntiFaceSpoofingLBPSVM(const std::string& model_path,
                           const std::string& mean_path,
                           const std::string& std_path);

    void LoadLBPSVMModel(const std::string& model_path,
                         const std::string& mean_path,
                         const std::string& std_path);

    void SetThreshold(float value) { m_lbpsvm_threshold = value; };
    /**
     * @brief Use SVM classify the LBP feature
     *
     * @param img The input face
     * @return true SVM is equal to threshold
     * @return false SVM is lower than threhold
     */

    bool Detect(const cv::Mat &image, const face_info_t &face_info);

    float GetLbpSVMResult() { return m_result; };

    private:
    const uint GetSharpnessLevel(cv::Mat &img);
    const uint GetDimFeat(uint radius, uint neightbors) { return (radius*neightbors+2)*3*2; };
    cv::Mat GetFeatureVector(cv::Mat crop_rgb_face, bool bNorm=true);
    void GetCircularLBPFeatureOptimization(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors);
	void GetUniformPatternLBPFeature(cv::InputArray _src, cv::OutputArray _dst, int radius, int neighbors);
	cv::Mat GetLocalRegionLBPH(const cv::Mat& src, int minValue, int maxValue, bool normed);
	cv::Mat GetLBPH(cv::InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed);
	int GetDimFeat(int radius, int neighbors);

    bool m_bModelLoaded = false;
    LibLinear *m_pLinear = nullptr;
    uint m_radius = 9, m_neighbors = 8, m_dim_feature;
    float *m_pMean = nullptr, *m_pStd = nullptr;
    float m_result = 0.0;
    float m_lbpsvm_threshold = 0.0;
};
}  // namespace vision
}  // namespace qnn