#include "face.hpp"
#include "utils/log_common.h"
#include "utils/function_tracer.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

namespace qnn {
namespace vision {

using namespace std;
using cv::Mat;
using cv::Point2f;
using cv::Rect;
using json = nlohmann::json;

void dump_faceinfo(const vector<face_detect_rect_t> &rects) {
    ostringstream msg;
    msg << endl;
    for (face_detect_rect_t fd_rect : rects) {
        float x = fd_rect.x1;
        float y = fd_rect.y1;
        float h = fd_rect.x2 - fd_rect.x1 + 1;
        float w = fd_rect.y2 - fd_rect.y1 + 1;
        msg << fd_rect.id << " (x y h w): (" << x << " " << y << " " << h << " " << w << ")"
            << endl;
    }
    LOGD << msg.str();
}

void dump_faceinfo(const vector<face_info_t> &faceInfo) {
    ostringstream msg;
    msg << endl;
    for (size_t i = 0; i < faceInfo.size(); i++) {
        float x = faceInfo[i].bbox.x1;
        float y = faceInfo[i].bbox.y1;
        float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
        float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
        msg << faceInfo[i].bbox.id << " (x y h w): (" << x << " " << y << " " << h << " " << w
            << ")" << std::endl;
    }
    LOGD << msg.str();
}

void SaveFaceInfoToJson(const string &output_name, const string &imgname,
                        const vector<face_info_t> &faces) {
    json output_json = json::array();

    for (const face_info_t &face : faces) {
        json j;

        j["image_name"] = imgname;
        j["category_name"] = "face";
        j["bbox_x"] = face.bbox.x1;
        j["bbox_y"] = face.bbox.y1;
        j["bbox_w"] = face.bbox.x2 - face.bbox.x1 + 1;
        j["bbox_h"] = face.bbox.y2 - face.bbox.y1 + 1;
        j["score"] = face.bbox.score;
        j["face_pts"] = json::array({{{"name", "left_eye"}},
                                     {{"name", "right_eye"}},
                                     {{"name", "nose"}},
                                     {{"name", "left_mouth"}},
                                     {{"name", "right_mouth"}}});
        for (int i = 0; i < FACE_PTS; i++) {
            j["face_pts"][i]["x"] = face.face_pts.x[i];
            j["face_pts"][i]["y"] = face.face_pts.y[i];
        }

        output_json.push_back(j);
    }

    std::ofstream o(output_name);
    o << std::setw(4) << output_json << std::endl;
}

void DrawFaceInfo(const std::vector<face_info_t> &faceInfo, cv::Mat &image) {
    for (size_t i = 0; i < faceInfo.size(); i++) {
        float x = faceInfo[i].bbox.x1;
        float y = faceInfo[i].bbox.y1;
        float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
        float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
        cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);

        face_pts_t facePts = faceInfo[i].face_pts;
        for (int j = 0; j < 5; j++) {
            cv::circle(image, cv::Point(facePts.x[j], facePts.y[j]), 1, cv::Scalar(255, 255, 0), 2);
        }
    }
}

std::vector<std::vector<float>> FaceInfoToVector(const std::vector<face_info_t> &faceInfo) {
    vector<vector<float>> ans;
    ans.push_back(std::vector<float>(1, faceInfo.size()));

    for (size_t i = 0; i < faceInfo.size(); i++) {
        std::vector<float> candidate;
        float x = faceInfo[i].bbox.x1;
        float y = faceInfo[i].bbox.y1;
        float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
        float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;

        candidate.push_back(x);
        candidate.push_back(y);
        candidate.push_back(w);
        candidate.push_back(h);
        face_pts_t facePts = faceInfo[i].face_pts;
        for (int j = 0; j < 5; j++) {
            candidate.push_back(facePts.x[j]);
            candidate.push_back(facePts.y[j]);
        }
        ans.push_back(candidate);
    }

    return ans;
}

// TODO: Don't directly put this function in utils or it'll cause circular dependencies.
// method : u is IoU(Intersection Over Union)
// method : m is IoM(Intersection Over Maximum)
void NonMaximumSuppression(std::vector<face_detect_rect_t> &bboxes,
                           std::vector<face_detect_rect_t> &bboxes_nms, float threshold,
                           char method) {
    BITMAIN_FUNCTION_TRACE(__PRETTY_FUNCTION__);
    std::sort(bboxes.begin(), bboxes.end(), [](auto &a, auto &b) { return a.score > b.score; });

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

        face_detect_rect_t select_bbox = bboxes[select_idx];
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
                                         (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1) continue;

            face_detect_rect_t &bbox_i(bboxes[i]);
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

template <>
void NonMaximumSuppression<face_info_t>(std::vector<face_info_t> &bboxes,
                                        std::vector<face_info_t> &bboxes_nms, float threshold,
                                        char method);
template <>
void NonMaximumSuppression<face_info_regression_t>(std::vector<face_info_regression_t> &bboxes,
                                                   std::vector<face_info_regression_t> &bboxes_nms,
                                                   float threshold, char method);

}  // namespace vision
}  // namespace qnn
