#include "mtcnn_helperfunc.hpp"
#include <cmath>

namespace qnn {
namespace vision {

using std::string;
using std::vector;

void BoxRegress(const std::vector<face_detect_rect_t> &boxes,
                const std::vector<std::array<float, 4>> &regression, const int stage,
                std::vector<face_detect_rect_t> *output_rects) {
    for (size_t bboxId = 0; bboxId < boxes.size(); bboxId++) {
        face_detect_rect_t faceRect;
        float regw = boxes[bboxId].x2 - boxes[bboxId].x1;
        float regh = boxes[bboxId].y2 - boxes[bboxId].y1;
        regw += (stage == 1) ? 0 : 1;
        regh += (stage == 1) ? 0 : 1;
        faceRect.x1 = boxes[bboxId].x1 + regw * regression[bboxId][0];
        faceRect.y1 = boxes[bboxId].y1 + regh * regression[bboxId][1];
        faceRect.x2 = boxes[bboxId].x2 + regw * regression[bboxId][2];
        faceRect.y2 = boxes[bboxId].y2 + regh * regression[bboxId][3];
        faceRect.score = boxes[bboxId].score;
        faceRect.id = boxes[bboxId].id;
        output_rects->emplace_back(faceRect);
    }
    return;
}

void BoxRegress(const std::vector<face_info_regression_t> &boxes, const int stage,
                std::vector<face_detect_rect_t> *output_rects) {
    for (size_t bboxId = 0; bboxId < boxes.size(); bboxId++) {
        face_detect_rect_t faceRect;
        float regw = boxes[bboxId].bbox.x2 - boxes[bboxId].bbox.x1;
        float regh = boxes[bboxId].bbox.y2 - boxes[bboxId].bbox.y1;
        regw += (stage == 1) ? 0 : 1;
        regh += (stage == 1) ? 0 : 1;
        faceRect.x1 = boxes[bboxId].bbox.x1 + regw * boxes[bboxId].regression[0];
        faceRect.y1 = boxes[bboxId].bbox.y1 + regh * boxes[bboxId].regression[1];
        faceRect.x2 = boxes[bboxId].bbox.x2 + regw * boxes[bboxId].regression[2];
        faceRect.y2 = boxes[bboxId].bbox.y2 + regh * boxes[bboxId].regression[3];
        faceRect.score = boxes[bboxId].bbox.score;
        faceRect.id = boxes[bboxId].bbox.id;

        output_rects->emplace_back(faceRect);
    }
    return;
}

void BoxRegress(const std::vector<face_info_regression_t> &boxes, const int stage,
                std::vector<face_info_t> *output_rects) {
    for (size_t bboxId = 0; bboxId < boxes.size(); bboxId++) {
        face_detect_rect_t faceRect;
        float regw = boxes[bboxId].bbox.x2 - boxes[bboxId].bbox.x1;
        float regh = boxes[bboxId].bbox.y2 - boxes[bboxId].bbox.y1;
        regw += (stage == 1) ? 0 : 1;
        regh += (stage == 1) ? 0 : 1;
        faceRect.x1 = boxes[bboxId].bbox.x1 + regw * boxes[bboxId].regression[0];
        faceRect.y1 = boxes[bboxId].bbox.y1 + regh * boxes[bboxId].regression[1];
        faceRect.x2 = boxes[bboxId].bbox.x2 + regw * boxes[bboxId].regression[2];
        faceRect.y2 = boxes[bboxId].bbox.y2 + regh * boxes[bboxId].regression[3];
        faceRect.score = boxes[bboxId].bbox.score;
        faceRect.id = boxes[bboxId].bbox.id;

        output_rects->emplace_back(face_info_t{faceRect, boxes[bboxId].face_pts});
    }
    return;
}

void AdjustBbox2Square(std::vector<face_detect_rect_t> &bboxes) {
    for (size_t i = 0; i < bboxes.size(); i++) {
        float w = bboxes[i].x2 - bboxes[i].x1;
        float h = bboxes[i].y2 - bboxes[i].y1;
        float side = h > w ? h : w;
        bboxes[i].x1 += (w - side) * 0.5;
        bboxes[i].y1 += (h - side) * 0.5;

        bboxes[i].x2 = static_cast<int>((bboxes[i].x1 + side));
        bboxes[i].y2 = static_cast<int>((bboxes[i].y1 + side));
        bboxes[i].x1 = static_cast<int>((bboxes[i].x1));
        bboxes[i].y1 = static_cast<int>((bboxes[i].y1));
    }
}

void RestoreBbox(const vector<face_detect_rect_t> &origin_bboxes,
                 const vector<face_info_t> &face_infos, vector<face_info_t> &results) {
    // adjust results according to resize ratio
    vector<face_info_t> faces;
    for (uint32_t j = 0; j < face_infos.size(); j++) {
        face_info_t face;
        face.bbox.x1 = origin_bboxes[j].x1;
        face.bbox.y1 = origin_bboxes[j].y1;
        face.bbox.x2 = origin_bboxes[j].x2;
        face.bbox.y2 = origin_bboxes[j].y2;
        face.bbox.score = origin_bboxes[j].score;
        face.face_pts = face_infos[j].face_pts;
        results.emplace_back(face);
    }
    dump_faceinfo(results);
}

void RestoreBboxWithRatio(const vector<face_info_t> &face_infos, const float &ratio,
                          vector<face_info_t> &results) {
    // adjust results according to resize ratio
    vector<face_info_t> faces;
    for (uint32_t j = 0; j < face_infos.size(); j++) {
        face_info_t face;
        face.bbox.x1 = face_infos[j].bbox.x1 * ratio;
        face.bbox.y1 = face_infos[j].bbox.y1 * ratio;
        face.bbox.x2 = face_infos[j].bbox.x2 * ratio;
        face.bbox.y2 = face_infos[j].bbox.y2 * ratio;
        face.bbox.score = face_infos[j].bbox.score;
        face.face_pts.x[0] = face_infos[j].face_pts.x[0] * ratio;
        face.face_pts.x[1] = face_infos[j].face_pts.x[1] * ratio;
        face.face_pts.x[2] = face_infos[j].face_pts.x[2] * ratio;
        face.face_pts.x[3] = face_infos[j].face_pts.x[3] * ratio;
        face.face_pts.x[4] = face_infos[j].face_pts.x[4] * ratio;
        face.face_pts.y[0] = face_infos[j].face_pts.y[0] * ratio;
        face.face_pts.y[1] = face_infos[j].face_pts.y[1] * ratio;
        face.face_pts.y[2] = face_infos[j].face_pts.y[2] * ratio;
        face.face_pts.y[3] = face_infos[j].face_pts.y[3] * ratio;
        face.face_pts.y[4] = face_infos[j].face_pts.y[4] * ratio;
        results.emplace_back(face);
    }
    dump_faceinfo(results);
}

int GetBoxPerBatch(int box_number) {
    if (box_number >= 128) {
        return 128;
    }

    // get the left most bit
    int index = 6;
    while (index >= 0) {
        if (box_number & (1 << index)) {
            return 1 << index;
        }
        index--;
    }
    return 0;
}

void Padding(vector<face_detect_rect_t> &origin_bboxes, vector<face_detect_rect_t> &squared_bboxes,
             vector<face_detect_rect_t> &pad_bboxes, const cv::Mat &image) {
    // backup squared_bboxes
    LOGD << "[MTCNN] padding squared faces:" << squared_bboxes.size();
    vector<face_detect_rect_t> origin_backup;
    vector<face_detect_rect_t> squared_backup;

    int skip_faces = 0;
    for (size_t i = 0; i < squared_bboxes.size(); i++) {
        face_detect_rect_t tmp;
        tmp = squared_bboxes[i];
        tmp.y2 = (squared_bboxes[i].y2 >= image.rows) ? image.rows - 1 : squared_bboxes[i].y2;
        tmp.x2 = (squared_bboxes[i].x2 >= image.cols) ? image.cols - 1 : squared_bboxes[i].x2;
        tmp.y1 = (squared_bboxes[i].y1 < 1) ? 1 : squared_bboxes[i].y1;
        tmp.x1 = (squared_bboxes[i].x1 < 1) ? 1 : squared_bboxes[i].x1;

        if (tmp.x2 - tmp.x1 < MIN_FACE_WIDTH || tmp.y2 - tmp.y1 < MIN_FACE_HEIGHT ||
            tmp.x1 - 1 > image.cols || tmp.y1 - 1 > image.rows) {
            ++skip_faces;
            continue;
        }

        origin_backup.emplace_back(origin_bboxes[i]);
        squared_backup.emplace_back(squared_bboxes[i]);
        pad_bboxes.emplace_back(tmp);
    }

    if (skip_faces > 0) {
        std::cerr << "[MTCNN] padding: skip " << skip_faces << " small faces" << std::endl;
    }

    // restore squared_bboxes
    LOGD << "[MTCNN] restore squared_bboxes:" << squared_bboxes.size();
    origin_backup.swap(origin_bboxes);
    squared_backup.swap(squared_bboxes);
}

}  // namespace vision
}  // namespace qnn