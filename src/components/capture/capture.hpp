#pragma once

#include <memory>
#include <vector>
#include <map>
#include <cstdint>
#include <string>
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"
#include "capture/object_capture.hpp"

typedef enum { AREA_RATIO = 0, EYES_DISTANCE=1 } quality_assessment_e;

// 派生类
class FaceCapture : public ObjectCapture {
public:
    FaceCapture(const std::string& capture_dir,
               std::shared_ptr<BaseModel> face_ld_model,
               const std::string& model_path = "./face_quality_cv186x.bmodel");

    void updateCaptureData(
        std::shared_ptr<BaseImage> image,
        uint64_t frame_id,
        const std::vector<ObjectBoxInfo>& boxes,
        const std::vector<TrackerInfo>& tracks
    ) override;

    void getCaptureData(
        uint64_t frame_id,  // 注意：这里与基类参数不同
        std::vector<ObjectCaptureInfo>& captures
    );

private:
    float computeFaceQuality(
        std::shared_ptr<BaseImage> image,
        const std::vector<ObjectBoxInfo>& boxes,
        const std::vector<TrackerInfo>& tracks,
        quality_assessment_e qa_method
    );
    float get_score(ObjectBoxInfo bbox, 
                    std::shared_ptr<ModelLandmarksInfo> landmarks_meta, 
                    TrackerInfo tracker,
                    uint32_t img_w, uint32_t img_h, bool fl_model);

    std::shared_ptr<BaseModel> face_ld_model;
    float obj_quality = 0.0f;
    std::unordered_map<uint64_t, ObjectCaptureInfo> capture_infos_;
    const size_t MAX_CAPTURE_NUM = 20;       // 最大存储数量
    const float QUALITY_THRESHOLD = 0.7f;    // 质量阈值
    const int MISS_TIME_LIMIT = 5;           // 丢失容忍帧数
};
