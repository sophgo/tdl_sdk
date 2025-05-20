#ifndef TDL_SDK_OBJECT_CAPTURE_HPP
#define TDL_SDK_OBJECT_CAPTURE_HPP

#include <memory>
#include "common/model_output_types.hpp"
#include "common/object_type_def.hpp"
#include "model/base_model.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"

struct ObjectCaptureInfo {
  float quality;
  std::shared_ptr<BaseImage> image;
  uint64_t last_capture_frame_id;
  int miss_counter = 0;
};
// 基类
class ObjectCapture {
 public:
  ObjectCapture(const std::string& capture_dir,
                ModelType model_id,
                const std::string& model_path);
  virtual ~ObjectCapture() = default;

  virtual void updateCaptureData(std::shared_ptr<BaseImage> image,
                                 uint64_t frame_id,
                                 const std::vector<ObjectBoxInfo>& boxes,
                                 const std::vector<TrackerInfo>& tracks);

  virtual void getCaptureData(std::vector<ObjectCaptureInfo>& captures);

 protected:
  std::map<uint64_t, std::shared_ptr<BaseImage>> capture_images;
  std::map<uint64_t, ObjectCaptureInfo> capture_object_infos;
  std::string capture_dir;
  uint64_t capture_interval = 30;
  std::shared_ptr<BaseModel> quality_model;
};

#endif /* TDL_SDK_OBJECT_CAPTURE_HPP */