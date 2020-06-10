// Copyright 2018 Bitmain Inc.
// License
// Author Yangwen Huang <yangwen.huang@bitmain.com>
/** @file */
#pragma once
#include "config.h"
#include "face/face.hpp"
#include "opencv2/opencv.hpp"
/**
 * @file face_debug.hpp
 * @ingroup NETWORKS_DEBUG
 */
namespace qnn {
namespace vision {

#if DEBUG_NETWORK_IMG
/** Save OpenCV image to file */
#    define BITMAIN_CVIMG_SAVE(filename, img) SaveCVImg(filename, img)
/** Draw output face rect to image and save to file. */
#    define BITMAIN_DRAWFDRECT_SAVE(filename, img, rect_vec) \
        DrawRectnSaveCVImg(filename, img, rect_vec)
/** Draw output face rect in info to image and save to file. */
#    define BITMAIN_DRAWFDINFO_SAVE(filename, img, info_vec) \
        DrawFaceInfonSaveCVImg(filename, img, info_vec)
#else
/** Save OpenCV image to file */
#    define BITMAIN_CVIMG_SAVE(filename, img)
/** Draw output face rect to image and save to file. */
#    define BITMAIN_DRAWFDRECT_SAVE(filename, img, rect_vec)
/** Draw output face rect in info to image and save to file. */
#    define BITMAIN_DRAWFDINFO_SAVE(filename, img, info_vec)
#endif

/**
 * @brief Save OpenCV image to file.
 *
 * @param name File path
 * @param src Mat input
 */
void SaveCVImg(const char* name, const cv::Mat& src);

/**
 * @brief Draw output face rect to image and save to file.
 *
 * @param name File name.
 * @param src Mat input
 * @param rects The input rectangles
 */
void DrawRectnSaveCVImg(const char* name, const cv::Mat& src,
                        std::vector<face_detect_rect_t>& rects);

/**
 * @brief Template function of draw output face rect to image and save to file.
 *
 * @tparam T face_info_regression_t or face_info_t
 * @param name File name
 * @param src Mat input
 * @param rects The input face infos
 */
template <class T>
void DrawFaceInfonSaveCVImg(const char* name, const cv::Mat& src, std::vector<T>& rects) {
    cv::Mat temp;
    src.copyTo(temp);
    for (size_t i = 0; i < rects.size(); i++) {
        face_detect_rect_t& rect = rects[i].bbox;
        float x = rect.x1;
        float y = rect.y1;
        float w = rect.x2 - rect.x1 + 1;
        float h = rect.y2 - rect.y1 + 1;
        cv::rectangle(temp, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 0.5);
    }
    cv::imwrite(name, temp);
}

/**
 * @brief Save images in the format of txt.
 *
 * @param name File name
 * @param src Mat input
 */
void SaveMat2Txt(const std::string name, const cv::Mat& src);

/**
 * @brief Save tensor to file in the format of txt.
 *
 * @param name File name
 * @param in_tensor The input tensor
 * @param out_tensors The output tensors in std::vector
 * @param is_app Appends to the existing file
 */
void SaveTensor2Txt(const std::string name, InputTensor& in_tensor,
                    std::vector<OutputTensor>& out_tensors, bool is_app = false);

/**
 * @brief Save face infos to file in the format of txt.
 *
 * @param name File name
 * @param infos The input face infos
 * @param is_app Appends to the existing file
 */
void SaveFDInfo2Txt(const std::string name, std::vector<face_info_regression_t>& infos,
                    bool is_app = false);

}  // namespace vision
}  // namespace qnn