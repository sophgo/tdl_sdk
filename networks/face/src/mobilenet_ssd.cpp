#include "mobilenet_ssd.hpp"
#include "utils/face_debug.hpp"
#include "utils/log_common.h"
#include "utils/math_utils.hpp"

#define MBSSD_MEAN (-128)
#define MBSSD_SCALE float(1) / 0.937958  // TODO: FIXME: Use get input threshold.
#define MBSSD_DEFAULT_THRESHOLD2 (0.9)
#define MBSSD_CLASSNUM 21
#define MBSSD_HUMANIDX 15
#define MBSSD_C 3
#define MBSSD_SZ 300

namespace qnn {
namespace vision {

using cv::Mat;
using std::max;
using std::min;
using std::string;
using std::vector;

static const vector<NetShape> kPossibleInputShapes = {NetShape(1, 3, MBSSD_SZ, MBSSD_SZ)};

MobileNetSSD::MobileNetSSD(const string &model_path, const string &anchor_path, QNNCtx *qnn_ctx)
    : ImageNet(model_path, kPossibleInputShapes, MBSSD_C, MBSSD_SZ, MBSSD_SZ, false,
               cv::INTER_NEAREST, qnn_ctx) {
    // Read predefined anchor from json file
    cv::FileStorage fs(anchor_path.c_str(), 0);
    cv::FileNode fn = fs.root();
    m_total_anchor_num = 0;  // Use for assert check
    for (cv::FileNodeIterator fnit = fn.begin(); fnit != fn.end(); ++fnit) {
        AnchorSSD unit;
        unit.name = (*fnit).name();
        cv::FileNode item = (*fnit)["anchors"];
        cv::FileNode item2 = (*fnit)["var"];
        for (size_t i = 0; i < item.size(); i += 4) {
            AnchorShape shape;
            float x1 = item[i];
            float y1 = item[i + 1];
            float x2 = item[i + 2];
            float y2 = item[i + 3];
            // Calculate w, h, cx, cy ratio from json
            shape.width = x2 - x1;
            shape.height = y2 - y1;
            shape.center_x = (x2 + x1) / 2;
            shape.center_y = (y2 + y1) / 2;
            // Output variance
            shape.variance[0] = item2[i];
            shape.variance[1] = item2[i + 1];
            shape.variance[2] = item2[i + 2];
            shape.variance[3] = item2[i + 3];
            unit.shapes.push_back(shape);
            m_total_anchor_num++;
        }
        m_anchors.push_back(unit);
    }
    // Init quantize value
    ImageQuantizeParameters qp(std::vector<float>{MBSSD_MEAN, MBSSD_MEAN, MBSSD_MEAN},
                               std::vector<float>{MBSSD_SCALE});
    SetQuanParams(qp);

    // TODO: now is 21, in the future padded to 32?
    m_confidence_data = allocate_aligned<float>(16, MBSSD_CLASSNUM);
}

void MobileNetSSD::Detect(const vector<Mat> &images, vector<vector<face_detect_rect_t>> &results) {
    if (images.size() == 0) {
        return;
    }
    int batch_num = images.size();
    results.clear();
    results.reserve(batch_num);

    ImageNet::Detect(images, [&](OutTensors &out, vector<float> &ratios, int start, int end) {
        assert(start >= 0 && start < end && end <= (int)images.size());
        auto confidence = out["mbox_conf"];
        auto reg_ptr = out["mbox_loc"].data;
        assert(m_total_anchor_num == (int)(confidence.count / MBSSD_CLASSNUM));
        float *confidence_data = m_confidence_data.get();
        size_t npts_count = 0;
        std::vector<face_detect_rect_t> candiates;
        for (size_t npts_idx = 0; npts_idx < m_anchors.size(); npts_idx++) {
            auto &shapes = m_anchors[npts_idx].shapes;
            for (size_t idx = 0; idx < shapes.size(); idx++) {
                size_t &&anchor_idx = npts_count + idx;
                qnn::math::SoftMaxForBuffer(confidence.data + anchor_idx * MBSSD_CLASSNUM,
                                            confidence_data, MBSSD_CLASSNUM);
                for (size_t i = 1; i < MBSSD_CLASSNUM; i++){
                    if (*(confidence_data + i) > MBSSD_DEFAULT_THRESHOLD2) {
                        auto &shape = shapes[idx];
                        float &reg0 = reg_ptr[4 * anchor_idx];
                        float &reg1 = reg_ptr[4 * anchor_idx + 1];
                        float &reg2 = reg_ptr[4 * anchor_idx + 2];
                        float &reg3 = reg_ptr[4 * anchor_idx + 3];
                        float &&center_x = shape.variance[0] * reg0 * shape.width + shape.center_x;
                        float &&center_y = shape.variance[1] * reg1 * shape.height + shape.center_y;
                        float &&width = exp(shape.variance[2] * reg2) * shape.width;
                        float &&height = exp(shape.variance[3] * reg3) * shape.height;
                        face_detect_rect_t faceRect;
                        faceRect.x1 = (center_x - width / 2) * images[start].cols;
                        faceRect.y1 = (center_y - height / 2) * images[start].rows;
                        faceRect.x2 = (center_x + width / 2) * images[start].cols;
                        faceRect.y2 = (center_y + height / 2) * images[start].rows;
                        faceRect.score = *(confidence_data + MBSSD_HUMANIDX);
                        faceRect.id = i;
                        candiates.push_back(faceRect);
                    }
                }
            }
            npts_count += shapes.size();
        }
        std::vector<face_detect_rect_t> bboxes_nms;
        NonMaximumSuppression(candiates, bboxes_nms, 0.7, 'm');
        results.push_back(bboxes_nms);
        BITMAIN_DRAWFDRECT_SAVE("mobilenetssd_output.jpg", images[start], bboxes_nms);
    });
}

}  // namespace vision
}  // namespace qnn
