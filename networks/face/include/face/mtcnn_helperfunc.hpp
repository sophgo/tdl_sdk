/**
 * @file mtcnn_helperfunc.hpp
 * @ingroup NETWORKS_FD
 */
#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

#include "face.hpp"
#include "opencv2/opencv.hpp"
#include "utils/log_common.h"

namespace qnn {
namespace vision {
/**
 * @brief Regress rect bounding box with nn regression output.
 *
 * @param boxes The input rect boxes
 * @param regression The regression data
 * @param stage Stage for p - 1, r - 2, o - 3
 * @param output_rects Output rects
 */
void BoxRegress(const std::vector<face_detect_rect_t> &boxes,
                const std::vector<std::array<float, 4>> &regression, const int stage,
                std::vector<face_detect_rect_t> *output_rects);

/**
 * @brief Regress face infos' bounding box with nn regression output.
 *
 * @param boxes The input face infos
 * @param stage Stage for p - 1, r - 2, o - 3
 * @param output_rects Output rects
 */
void BoxRegress(const std::vector<face_info_regression_t> &boxes, const int stage,
                std::vector<face_detect_rect_t> *output_rects);

/**
 * @brief Regress face infos' bounding box with nn regression output.
 *
 * @param boxes The input face infos
 * @param stage Stage for p - 1, r - 2, o - 3
 * @param output_rects Output face infos
 */
void BoxRegress(const std::vector<face_info_regression_t> &boxes, const int stage,
                std::vector<face_info_t> *output_rects);

/**
 * @brief Adjust face rects to square boxes.
 *
 * @param bboxes The input / output box
 */
void AdjustBbox2Square(std::vector<face_detect_rect_t> &bboxes);

/**
 * @brief Combine bbox from origin_bboxes and face landmarks from face_infos.
 *
 * @param origin_bboxes Original tensor calculated boxes
 * @param face_infos Face infos contains face landmarks (only landmarks will pass to output)
 * @param results Output face infos
 */
void RestoreBbox(const std::vector<face_detect_rect_t> &origin_bboxes,
                 const std::vector<face_info_t> &face_infos, std::vector<face_info_t> &results);

/**
 * @brief Restore bbox from resized image scale to original image scale.
 *
 * @param face_infos Input face infos
 * @param ratio Scaling ratio
 * @param results Output face infos
 */
void RestoreBboxWithRatio(const std::vector<face_info_t> &face_infos, const float &ratio,
                          std::vector<face_info_t> &results);

/**
 * @brief Get recommended batch number from the input candidates.
 *
 * @param box_number Total candidates
 * @return int Recommended batch size
 */
int GetBoxPerBatch(int box_number);

/**
 * @brief Pad the squared rects into the size mtcnn needs.
 *
 * @param origin_bboxes The original rects
 * @param squared_bboxes The squared rects from origin_bboxes
 * @param pad_bboxes The output padded boxes
 * @param image Input image that makes sure no padded boxes are outside the image
 */
void Padding(std::vector<face_detect_rect_t> &origin_bboxes,
             std::vector<face_detect_rect_t> &squared_bboxes,
             std::vector<face_detect_rect_t> &pad_bboxes, const cv::Mat &image);

template <typename T>
void NonMaximumSuppression(std::vector<T> &bboxes, std::vector<T> &bboxes_nms, float threshold,
                           char method) {
    std::sort(bboxes.begin(), bboxes.end(), [](T &a, T &b) { return a.bbox.score > b.bbox.score; });

    int select_idx = 0;
    int num_bbox = bboxes.size();
    std::vector<int> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.emplace_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        face_detect_rect_t select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                         (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1) continue;

            face_detect_rect_t &bbox_i(bboxes[i].bbox);
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0) {
                continue;
            }

            float area2 =
                static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;
            if (method == 'u' &&
                static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
                continue;
            }
            if (method == 'm' &&
                static_cast<float>(area_intersect) / std::min(area1, area2) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }
}

}  // namespace vision
}  // namespace qnn
