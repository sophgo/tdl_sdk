// Copyright 2018 Bitmain Inc.
// License
// Author
/**
 * @file face.hpp
 * @ingroup NETWORKS_FACE
 */
#pragma once

#include "core/image_net.hpp"
#include "core/net_types.hpp"
#include "face_types.hpp"
#include <array>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#define MIN_FACE_WIDTH (16)
#define MIN_FACE_HEIGHT (16)

namespace qnn {
namespace vision {

class FaceDetector {
    public:
    virtual void Detect(const cv::Mat &image, std::vector<face_info_t> &results) = 0;
    virtual void Detect(const std::vector<cv::Mat> &images,
                        std::vector<std::vector<face_info_t>> &results) = 0;
    virtual ~FaceDetector(){};
};

class FaceRecognizer {
    public:
    virtual void Detect(const cv::Mat &image, FaceAttributeInfo &result) = 0;
    virtual void Detect(const std::vector<cv::Mat> &images,
                        std::vector<FaceAttributeInfo> &results) = 0;
    virtual ~FaceRecognizer(){};
};

/**
 * @brief Dump the bbox of face_detect_rect_t.
 *
 * @param faceInfo Input face rects
 */
void dump_faceinfo(const std::vector<face_detect_rect_t> &faceInfo);

/**
 * @brief Dump the bbox inside face_info_t.
 *
 * @param faceInfo Input face infos
 */
void dump_faceinfo(const std::vector<face_info_t> &faceInfo);

void SaveFaceInfoToJson(const std::string &output_name, const std::string &imgname,
                        const std::vector<face_info_t> &faces);

void DrawFaceInfo(const std::vector<face_info_t> &faceInfo, cv::Mat &image);

std::vector<std::vector<float>> FaceInfoToVector(const std::vector<face_info_t> &faceInfo);

net_err_t face_align(const cv::Mat &image, cv::Mat &aligned, const face_info_t &face_info,
                     int width, int height);

// method : u is IoU(Intersection Over Union)
// method : m is IoM(Intersection Over Maximum)
/**
 * @brief Do nms on output face rects.
 *
 * @param bboxes The input face rects
 * @param bboxes_nms The output face rects
 * @param threshold The nms threshold
 * @param method Nms method: u is IoU(Intersection Over Union), m is IoM(Intersection Over Maximum)
 */
void NonMaximumSuppression(std::vector<face_detect_rect_t> &bboxes,
                           std::vector<face_detect_rect_t> &bboxes_nms, float threshold,
                           char method);

/**
 * @brief Do nms on output face rects.
 *
 * @tparam T face_info_regression_t or face_info_t
 * @param bboxes The input face rects
 * @param bboxes_nms The output face rects
 * @param threshold The nms threshold
 * @param method Nms method: u is IoU(Intersection Over Union), m is IoM(Intersection Over Maximum)
 */
template <typename T>
void NonMaximumSuppression(std::vector<T> &bboxes, std::vector<T> &bboxes_nms, float threshold,
                           char method);
}  // namespace vision
}  // namespace qnn
