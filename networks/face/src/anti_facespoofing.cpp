// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>

#include "anti_facespoofing.hpp"
#include "utils/function_tracer.h"
#include "utils/image_utils.hpp"

namespace qnn {
namespace vision {

// TODO: FIXME: Uncomment these lines when spoofing model is ready.
AntiFaceSpoofing::AntiFaceSpoofing(const std::string &depth_model_path,
                                   const std::string &flow_model_path,
                                   const std::string &patch_model_path,
                                   const std::string &classify_model_path,
                                   const std::string &classify_hsv_ycbcr_model_path,
                                   const std::string &canny_model_path,
                                   QNNCtx *qnn_ctx)
    : m_classifyfilter(classify_model_path, qnn_ctx),
      m_classify_hsv_ycbcr_filter(classify_hsv_ycbcr_model_path, qnn_ctx),
      m_depthfilter(depth_model_path, qnn_ctx),
      m_flowfilter(flow_model_path, qnn_ctx),
      m_patchfilter(patch_model_path, qnn_ctx),
      m_cannyfilter(canny_model_path, qnn_ctx) {
    // TODO: FIXME: Temporarily for experimentss
    std::ofstream myFile("spoofing.txt");
    myFile << "";
    myFile.close();
}

void AntiFaceSpoofing::UseTrackerQueue(const bool flag) { use_simple_flag_decision = !flag; }

void AntiFaceSpoofing::SetDoFilter(bool cv_bgfilter_flag, bool cv_classify_flag,
                                   bool classify_hsv_ycbcr_flag, bool cv_sharpfilter_flag,
                                   bool cv_lbpsvm_flag, bool depth_filter_flag,
                                   bool flow_filter_flag, bool patch_filter_flag,
                                   bool canny_filter_flag) {
    m_bgfilter.do_filter = cv_bgfilter_flag;
    m_classifyfilter.do_filter = cv_classify_flag;
    m_classify_hsv_ycbcr_filter.do_filter = classify_hsv_ycbcr_flag;
    m_sharpfilter.do_filter = cv_sharpfilter_flag;
    m_lbpsvmfilter.do_filter = cv_lbpsvm_flag;
    m_depthfilter.do_filter = depth_filter_flag;
    m_flowfilter.do_filter = flow_filter_flag;
    m_patchfilter.do_filter = patch_filter_flag;
    m_cannyfilter.do_filter = canny_filter_flag;
}

void AntiFaceSpoofing::SetSharpnessFilterThreshold(uint threshold) {
    m_sharpfilter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::SetClassifyFilterThreshold(float threshold) {
    m_classifyfilter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::SetClassifyHSVYCbCrFilterThreshold(float threshold) {
    m_classify_hsv_ycbcr_filter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::LoadLbpSVMModel(const std::string &model_path,
                                       const std::string &mean_path,
                                       const std::string &std_path) {
    m_lbpsvmfilter.inst.LoadLBPSVMModel(model_path, mean_path, std_path);
}

void AntiFaceSpoofing::SetLbpSVMFilterThreshold(float threshold) {
    m_lbpsvmfilter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::SetDepthFilterThreshold(size_t threshold) {
    m_depthfilter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::SetFlowFilterThreshold(float threshold) {
    m_flowfilter.inst.SetThreshold(threshold);
}

void AntiFaceSpoofing::SetCannyFilterThreshold(float threshold) {
    m_cannyfilter.inst.SetThreshold(threshold);
}

cv::Mat AntiFaceSpoofing::GetDepthVisualMat() {
    if (!m_depthfilter.do_filter) return cv::Mat();
    return m_depthfilter.inst.GetDepthVisualMat();
}

const size_t AntiFaceSpoofing::GetDepthCounterVal() { return m_depthfilter.inst.GetCounterVal(); }

const size_t AntiFaceSpoofing::GetPatchCounterVal() { return m_patchfilter.inst.GetCounterVal(); }

const uint AntiFaceSpoofing::GetSharpnessVal() { return m_sharpfilter.inst.GetSharpnessVal(); }

cv::Rect AntiFaceSpoofing::squareRectScale(const cv::Rect &rect, const float scale_val) {
    cv::Rect bbox = rect;
    cv::Rect square_bbox;
    // Convert to square rect
    if (bbox.width != bbox.height) {
        int max_length = std::max(bbox.width, bbox.height);
        square_bbox = cv::Rect(bbox.x + bbox.width/2 - max_length/2,
                               bbox.y + bbox.height/2 - max_length/2,
                               max_length, max_length);
    } else {
        square_bbox = bbox;
    }
    float new_length = square_bbox.width * scale_val;
    return cv::Rect(square_bbox.x + square_bbox.width/2 - new_length/2,
                    square_bbox.y + square_bbox.height/2 - new_length/2,
                    new_length, new_length);
}

cv::Mat AntiFaceSpoofing::cropImage(const cv::Mat &image, const cv::Rect &face_rect) {
    int offset = 0;
    cv::Rect bbox = face_rect;

    // Prevent image distortion
    if ((bbox.width > image.cols) || (bbox.height > image.rows)) {
        int new_left = std::max(0, bbox.x);
        int new_top = std::max(0, bbox.y);
        int new_right = std::min(image.cols, bbox.x + bbox.width);
        int new_bottom = std::min(image.rows, bbox.y + bbox.height);

        int new_width = new_right - new_left;
        int new_height = new_bottom - new_top;

        if (new_width > new_height) {
            int center_x = (new_left + new_right) / 2;
            new_left = center_x - new_height / 2;
            new_right = center_x + new_height / 2;
        } else if (new_height > new_width) {
            int center_y = (new_bottom + new_top) / 2;
            new_top = center_y - new_width / 2;
            new_bottom = center_y + new_width / 2;
        }
        bbox = cv::Rect(new_left, new_top, new_right - new_left, new_bottom - new_top);
    } else {
        if ((bbox.x + bbox.width) >= image.cols) {
            offset = bbox.x + bbox.width - image.cols - 1;
            bbox.x = std::max(0, bbox.x - offset);
            bbox.width = (image.cols - 1) - bbox.x;
        }
        if (bbox.x < 0) {
            bbox.x = 0;
            bbox.width = std::min(image.cols, bbox.width);
        }
        if ((bbox.y + bbox.height) >= image.rows) {
            offset = (bbox.y + bbox.height) - (image.rows - 1);
            bbox.y = std::max(0, bbox.y - offset);
            bbox.height = (image.rows - 1) - bbox.y;
        }
        if (bbox.y < 0) {
            bbox.y = 0;
            bbox.height = std::min(image.rows, bbox.height);
        }
    }

    return image(bbox).clone();
}

const float AntiFaceSpoofing::GetConfidenceVal() { return m_classifyfilter.inst.GetConfidence(); }

const float AntiFaceSpoofing::GetClassifyHSVYCbCrConfidenceVal() { return m_classify_hsv_ycbcr_filter.inst.GetConfidence(); }

const float AntiFaceSpoofing::GetWeightedConfidenceVal() { return m_weighted_confidence; }

const float AntiFaceSpoofing::GetCannyConfidenceVal() { return m_cannyfilter.inst.GetConfidence(); }

const float AntiFaceSpoofing::GetLbpSVMResult() {return m_lbpsvmfilter.inst.GetLbpSVMResult(); }

void AntiFaceSpoofing::Detect(const cv::Mat &image, const std::vector<cv::Rect> &face_rect,
                              std::vector<bool> &spoofing_flags) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    if (image.empty()) {
        return;
    }
    spoofing_flags.clear();
    short working_filter = 0;
    float sum_base_weighting = 0.0;
    float sum_weighting = 0.0;
    // Must run every frame even face_rect size is 0
    std::vector<TrackerPair> filter_values;
    if (m_bgfilter.do_filter) {
        bool bg_result = false;
        bg_result = (bool)m_bgfilter.inst.BGDetect(image, face_rect);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::CV_BACKGROUND, bg_result));
    }

    if (face_rect.size() == 0) return;

    if (m_lbpsvmfilter.do_filter) {
        bool lbpsvm_result = false;
        lbpsvm_result = m_lbpsvmfilter.inst.Detect(image, m_face_info);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::CV_LBPSVM, lbpsvm_result));

        if (use_weighting_results) {
            float confidence = m_lbpsvmfilter.inst.GetLbpSVMResult() * m_lbpsvmfilter.weighting;
            sum_base_weighting += m_lbpsvmfilter.weighting;
            sum_weighting += confidence;
        }
    }

    if (m_classifyfilter.do_filter) {
        bool classify_result = false;
        // Preprocess
        cv::Rect scale_bbox = squareRectScale(face_rect[0], 0.6);
        cv::Mat crop_image = cropImage(image, scale_bbox);
        // Resize image
        cv::Mat resized_image;
        cv::resize(crop_image, resized_image, {224, 224}, 0, 0);
        if (!resized_image.empty()) m_classifyfilter.inst.Detect(resized_image, classify_result);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_CLASSIFY, classify_result));

        if (use_weighting_results) {
            float confidence = m_classifyfilter.inst.GetConfidence() * m_classifyfilter.weighting;
            sum_base_weighting += m_classifyfilter.weighting;
            sum_weighting += confidence;
        }
    }

    if (m_classify_hsv_ycbcr_filter.do_filter) {
        bool classify_hsv_ycbcr_result = false;
        // Preprocess
        cv::Rect scale_bbox = squareRectScale(face_rect[0], 0.7);
        cv::Mat crop_image = cropImage(image, scale_bbox);
        // Resize image
        cv::Mat resized_image;
        cv::resize(crop_image, resized_image, {224, 224}, 0, 0);
        if (!resized_image.empty()) m_classify_hsv_ycbcr_filter.inst.Detect(resized_image, classify_hsv_ycbcr_result);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_CLASSIFY_HSV_YCBCR, classify_hsv_ycbcr_result));

        if (use_weighting_results) {
            float confidence = m_classify_hsv_ycbcr_filter.inst.GetConfidence() * m_classify_hsv_ycbcr_filter.weighting;
            sum_base_weighting += m_classify_hsv_ycbcr_filter.weighting;
            sum_weighting += confidence;
        }
    }

    if (m_patchfilter.do_filter) {
        bool patch_result = false;
        cv::Mat crop_img = image(RectScale(face_rect[0], image.cols, image.rows, 1.5)).clone();
        if (!crop_img.empty()) m_patchfilter.inst.Detect(crop_img, patch_result);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_PATCH, patch_result));
    }

    if (m_cannyfilter.do_filter) {
        bool canny_result = false;

        m_cannyfilter.inst.Detect(image, face_rect[0], canny_result);
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_CANNY, canny_result));
    }

    if(m_flowfilter.do_filter) {
        bool flow_result = false;
        m_flowfilter.inst.Detect(image, face_rect[0], flow_result);
        working_filter++;
        filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_OPTFLOW, flow_result));

        if (use_weighting_results) {
            float confidence = m_flowfilter.inst.GetConfidence() * m_flowfilter.weighting;
            sum_base_weighting += m_flowfilter.weighting;
            sum_weighting += confidence;
        }
    }

    if (m_depthfilter.do_filter || m_sharpfilter.do_filter) {
        cv::Mat crop_img =
            image(RectScale(face_rect[0], image.cols, image.rows, 0.9, 0, 10)).clone();
        if (m_depthfilter.do_filter) {
            bool depth_result = false;
            m_depthfilter.inst.Detect(crop_img, depth_result);
            working_filter++;
            filter_values.push_back(TrackerPair(SPOOFINGMODULES::NN_DEPTH, depth_result));
        }

        if (m_sharpfilter.do_filter) {
            bool sharpness_result = false;
            sharpness_result = m_sharpfilter.inst.Detect(crop_img);
            working_filter++;
            filter_values.push_back(TrackerPair(SPOOFINGMODULES::CV_SHARPNESS, sharpness_result));
        }
    }

    std::string res_name = "";
    std::string res_res = "";
    for (size_t i = 0; i < filter_values.size(); i++) {
        res_name += std::to_string((int)filter_values[i].first);
        res_res += std::to_string((int)filter_values[i].second);
        if (i != filter_values.size() - 1) {
            res_name += ", ";
            res_res += ", ";
        }
    }
    LOGI << "Result: (" << res_name << ") = (" << res_res << ")";

    if (use_weighting_results) {
        float avg_weighting = sum_weighting / sum_base_weighting;
        std::cout << "Using Weighting Result!, Weighting Sum: " << sum_weighting
                  << " ,Base Weighting: " << sum_base_weighting
                  << " ,Avg Weighting: " << avg_weighting << std::endl;
        m_weighted_confidence = avg_weighting;
        if (avg_weighting > m_weighting_threshold) {
            spoofing_flags.push_back(true);
        }
        else {
            spoofing_flags.push_back(false);
        }
        for (size_t i = 1; i < face_rect.size(); i++)
            spoofing_flags.push_back(false);

        return;
    }
    // Final result
    bool is_real = false;
    m_queue.GetTrakerResult(face_rect[0], working_filter, filter_values, is_real,
                            use_simple_flag_decision);

    spoofing_flags.push_back(is_real);
    for (size_t i = 1; i < face_rect.size(); i++) {
        spoofing_flags.push_back(false);
    }
    return;
}
}  // namespace vision
}  // namespace qnn