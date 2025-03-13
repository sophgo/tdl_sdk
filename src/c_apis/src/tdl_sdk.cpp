#include "tdl_sdk.h"

#include "common/common_types.hpp"
#include "tdl_type_internal.hpp"
#include "tdl_utils.h"
#include "utils/tdl_log.hpp"
cvtdl_handle_t CVI_TDL_CreateHandle(const int32_t tpu_device_id) {
  tdl_context_t *context = new tdl_context_t();
  return (cvtdl_handle_t)context;
}

int32_t CVI_TDL_DestroyHandle(cvtdl_handle_t handle) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  for (auto &model : context->models) {
    CVI_TDL_CloseModel(handle, model.first);
  }
  delete context;
  return 0;
}

cvtdl_image_t CVI_TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory) {
  if (vpss_frame == nullptr) {
    return nullptr;
  }

  // TODO(fuquan.ke): use own_memory to create VPSSFrame
  tdl_image_context_t *image_context = new tdl_image_context_t();
  image_context->image = ImageFactory::wrapVPSSFrame(vpss_frame, own_memory);
  return (cvtdl_image_t)image_context;
}

cvtdl_image_t CVI_TDL_ReadImage(const char *path) {
  tdl_image_context_t *image_context = new tdl_image_context_t();
  image_context->image = ImageFactory::readImage(path, false, InferencePlatform::CVITEK);
  return (cvtdl_image_t)image_context;
}

int32_t CVI_TDL_DestroyImage(cvtdl_image_t image_handle) {
  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;
  if (image_context == nullptr) {
    return -1;
  }
  delete image_context;
  return 0;
}

int32_t CVI_TDL_OpenModel(cvtdl_handle_t handle, const cvtdl_model_e model_id,
                          const char *model_path) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) != context->models.end()) {
    return 0;
  }
  ModelType model_type = convertModelType(model_id);
  std::shared_ptr<BaseModel> model =
      context->model_factory->getModel(model_type, model_path);
  if (model == nullptr) {
    return -1;
  }
  context->models[model_id] = model;
  return 0;
}

int32_t CVI_TDL_CloseModel(cvtdl_handle_t handle,
                           const cvtdl_model_e model_id) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) == context->models.end()) {
    LOGW("model %d not found", model_id);
    return -1;
  }
  context->models.erase(model_id);
  return 0;
}

int32_t CVI_TDL_ObjectDetection(cvtdl_handle_t handle,
                                const cvtdl_model_e model_id,
                                cvtdl_image_t image_handle,
                                cvtdl_object_t *object_meta) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) == context->models.end()) {
    return -1;
  }
  std::shared_ptr<BaseModel> model = context->models[model_id];
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }

  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  ModelBoxInfo *object_detection_output = (ModelBoxInfo *)output.get();
  CVI_TDL_InitObjectMeta(object_meta, object_detection_output->bboxes.size());
  object_meta->width = object_detection_output->image_width;
  object_meta->height = object_detection_output->image_height;
  for (int i = 0; i < object_detection_output->bboxes.size(); i++) {
    object_meta->info[i].box.x1 = object_detection_output->bboxes[i].x1;
    object_meta->info[i].box.y1 = object_detection_output->bboxes[i].y1;
    object_meta->info[i].box.x2 = object_detection_output->bboxes[i].x2;
    object_meta->info[i].box.y2 = object_detection_output->bboxes[i].y2;
    object_meta->info[i].class_id = object_detection_output->bboxes[i].class_id;
    object_meta->info[i].score = object_detection_output->bboxes[i].score;
  }
  return 0;
}

int32_t CVI_TDL_FaceDetection(cvtdl_handle_t handle,
                              const cvtdl_model_e model_id,
                              cvtdl_image_t image_handle,
                              cvtdl_face_t *face_meta) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) == context->models.end()) {
    return -1;
  }
  std::shared_ptr<BaseModel> model = context->models[model_id];
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    ModelBoxLandmarkInfo *box_landmark_output =
        (ModelBoxLandmarkInfo *)output.get();
    uint32_t num_landmark_per_face =
        box_landmark_output->box_landmarks[0].landmarks_x.size();
    CVI_TDL_InitFaceMeta(face_meta, box_landmark_output->box_landmarks.size(),
                         num_landmark_per_face);
    for (size_t i = 0; i < box_landmark_output->box_landmarks.size(); i++) {
      face_meta->info[i].box.x1 = box_landmark_output->box_landmarks[i].x1;
      face_meta->info[i].box.y1 = box_landmark_output->box_landmarks[i].y1;
      face_meta->width = box_landmark_output->box_landmarks[i].x2 -
                                     box_landmark_output->box_landmarks[i].x1;
      face_meta->height = box_landmark_output->box_landmarks[i].y2 -
                                      box_landmark_output->box_landmarks[i].y1;
      face_meta->info[i].score =
          box_landmark_output->box_landmarks[i].score;
      for (size_t j = 0; j < num_landmark_per_face; j++) {
        face_meta->info[i].landmarks.x[j] =
            box_landmark_output->box_landmarks[i].landmarks_x[j];
        face_meta->info[i].landmarks.y[j] =
            box_landmark_output->box_landmarks[i].landmarks_y[j];
      }
      face_meta->info[i].landmarks.score =
          box_landmark_output->box_landmarks[i].landmarks_score[0];
    }
    face_meta->width = box_landmark_output->image_width;
    face_meta->height = box_landmark_output->image_height;
    face_meta->size = box_landmark_output->box_landmarks.size();
  } else if (output->getType() == ModelOutputType::OBJECT_DETECTION) {
    ModelBoxInfo *object_detection_output = (ModelBoxInfo *)output.get();
    CVI_TDL_InitFaceMeta(face_meta, object_detection_output->bboxes.size(), 0);
    for (size_t i = 0; i < object_detection_output->bboxes.size(); i++) {
      face_meta->info[i].box.x1 = object_detection_output->bboxes[i].x1;
      face_meta->info[i].box.y1 = object_detection_output->bboxes[i].y1;
      face_meta->width = object_detection_output->bboxes[i].x2 -
                                     object_detection_output->bboxes[i].x1;
      face_meta->height = object_detection_output->bboxes[i].y2 -
                                      object_detection_output->bboxes[i].y1;
      face_meta->info[i].score = object_detection_output->bboxes[i].score;
    }
    face_meta->width = object_detection_output->image_width;
    face_meta->height = object_detection_output->image_height;
    face_meta->size = object_detection_output->bboxes.size();
  }

  return 0;
}
