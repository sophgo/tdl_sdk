#include "tdl_sdk.h"

#include <opencv2/opencv.hpp>
#include "common/common_types.hpp"
#include "tdl_type_internal.hpp"
#include "tdl_utils.h"
#include "tracker/tracker_types.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
#include "video_decoder/video_decoder_type.hpp"

std::shared_ptr<BaseModel> get_model(TDLHandle handle,
                                     const TDLModel model_id) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return nullptr;
  }
  if (context->models.find(model_id) == context->models.end()) {
    LOGW("model %d not found", model_id);
    return nullptr;
  }
  return context->models[model_id];
}

TDLHandle TDL_CreateHandle(const int32_t tpu_device_id) {
  TDLContext *context = new TDLContext();
  return (TDLHandle)context;
}

int32_t TDL_DestroyHandle(TDLHandle handle) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  for (auto &model : context->models) {
    TDL_CloseModel(handle, model.first);
  }
  delete context;
  context = nullptr;
  return 0;
}

TDLImage TDL_WrapFrame(void *frame, bool own_memory) {
  if (frame == nullptr) {
    return nullptr;
  }

  // TODO(fuquan.ke): use own_memory to create VPSSFrame
  TDLImageContext *image_context = new TDLImageContext();
  image_context->image = ImageFactory::wrapVPSSFrame(frame, own_memory);
  return (TDLImage)image_context;
}

#if !defined(__BM168X__) && !defined(__CMODEL_CV181X__)
int32_t TDL_InitCamera(TDLHandle handle) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }

  context->video_decoder =
      VideoDecoderFactory::createVideoDecoder(VideoDecoderType::VI);

  if (context->video_decoder == nullptr) {
    LOGE("create video decoder failed\n");
    return -1;
  }

  return 0;
}

TDLImage TDL_GetCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;

  TDLImageContext *image_context = new TDLImageContext();

  context->video_decoder->read(image_context->image, chn);

  return (TDLImage)image_context;
}

int32_t TDL_ReleaseCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder->release(chn) != 0) {
    LOGE("release camera frame failed\n");
    return -1;
  }
  return 0;
}

int32_t TDL_DestoryCamera(TDLHandle handle) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder != nullptr) {
    context->video_decoder.reset();
    context->video_decoder = nullptr;
  }
  return 0;
}
#endif

TDLImage TDL_ReadImage(const char *path) {
  TDLImageContext *image_context = new TDLImageContext();
  image_context->image = ImageFactory::readImage(path, ImageFormat::BGR_PACKED,
                                                 InferencePlatform::AUTOMATIC);
  return (TDLImage)image_context;
}

TDLImage TDL_ReadBin(const char *path, TDLDataTypeE data_type) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    return 0;
  }
  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  TDLImageContext *image_context = new TDLImageContext();
  TDLDataType tdl_data_type = convertDataTypeE(data_type);
  int data_size = CommonUtils::getDataTypeSize(tdl_data_type);
  int num_data = file_size / data_size;
  if (file_size % data_size != 0) {
    LOGE("file size %d is not aligned with data type size %d", file_size,
         data_size);
    fclose(file);
    return nullptr;
  }
  image_context->image = ImageFactory::createImage(
      num_data, 1, ImageFormat::GRAY, tdl_data_type, true);
  uint8_t *data_buffer = image_context->image->getVirtualAddress()[0];
  fread(data_buffer, 1, file_size, file);
  fclose(file);
  return (TDLImage)image_context;
}

int32_t TDL_DestroyImage(TDLImage image_handle) {
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  if (image_context == nullptr) {
    return -1;
  }
  if (image_context->image) {
    image_context->image.reset();
  }
  delete image_context;
  image_context = nullptr;
  return 0;
}

int32_t TDL_SetModelThreshold(TDLHandle handle, const TDLModel model_id,
                              float threshold) {
  if (threshold < 0.0 || threshold > 1.0) {
    LOGE("Invalid threshold value: %f", threshold);
    return -1;
  }

  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }

  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  float oldthreshold = model->getModelThreshold();
  if (oldthreshold == threshold) {
    return 0;
  }
  LOGI("Set threshold from %f to %f", oldthreshold, threshold);

  model->setModelThreshold(threshold);
  return 0;
}
int32_t TDL_LoadModelConfig(TDLHandle handle, const char *model_config_json) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  TDLModelFactory &factory = TDLModelFactory::getInstance();
  return factory.loadModelConfig(model_config_json);
}
int32_t TDL_SetModelDir(TDLHandle handle, const char *model_dir) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  TDLModelFactory &factory = TDLModelFactory::getInstance();
  factory.setModelDir(model_dir);
  return 0;
}
int32_t TDL_OpenModel(TDLHandle handle, const TDLModel model_id,
                      const char *model_path, const char *model_config_json) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) != context->models.end()) {
    return 0;
  }
  ModelType model_type = convertModelType(model_id);
  TDLModelFactory &factory = TDLModelFactory::getInstance();
  std::string str_model_path = model_path;
  if (str_model_path.empty()) {
    str_model_path = factory.getModelPath(model_type);
  }
  ModelConfig model_config = factory.getModelConfig(model_type);
  if (model_config_json != nullptr) {
    try {
      nlohmann::json json_config = nlohmann::json::parse(model_config_json);
      model_config = factory.parseModelConfig(json_config);
    } catch (const std::exception &e) {
      LOGE("Failed to parse model config: %s", e.what());
      return -1;
    }
  }
  std::shared_ptr<BaseModel> model =
      factory.getModel(model_type, str_model_path, model_config);
  if (model == nullptr) {
    return -1;
  }
  context->models[model_id] = model;
  return 0;
}

int32_t TDL_CloseModel(TDLHandle handle, const TDLModel model_id) {
  TDLContext *context = (TDLContext *)handle;
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

int32_t TDL_Detection(TDLHandle handle, const TDLModel model_id,
                      TDLImage image_handle, TDLObject *object_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }

  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    ModelBoxLandmarkInfo *object_Landmark_output =
        (ModelBoxLandmarkInfo *)output.get();
    if (object_Landmark_output->box_landmarks.size() <= 0) {
      LOGI("TDL_Detection: None to detect\n");
      return 0;
    }
    TDL_InitObjectMeta(
        object_meta, object_Landmark_output->box_landmarks.size(),
        object_Landmark_output->box_landmarks[0].landmarks_x.size());
    for (int i = 0; i < object_Landmark_output->box_landmarks.size(); i++) {
      object_meta->info[i].box.x1 = object_Landmark_output->box_landmarks[i].x1;
      object_meta->info[i].box.y1 = object_Landmark_output->box_landmarks[i].y1;
      object_meta->info[i].box.x2 = object_Landmark_output->box_landmarks[i].x2;
      object_meta->info[i].box.y2 = object_Landmark_output->box_landmarks[i].y2;
      object_meta->info[i].class_id =
          object_Landmark_output->box_landmarks[i].class_id;
      object_meta->info[i].score =
          object_Landmark_output->box_landmarks[i].score;
      for (int j = 0;
           j < object_Landmark_output->box_landmarks[i].landmarks_x.size();
           j++) {
        object_meta->info[i].landmark_properity[j].x =
            object_Landmark_output->box_landmarks[i].landmarks_x[j];
        object_meta->info[i].landmark_properity[j].y =
            object_Landmark_output->box_landmarks[i].landmarks_y[j];
        object_meta->info[i].landmark_properity[j].score =
            object_Landmark_output->box_landmarks[i].landmarks_score[j];
      }
    }
  } else if (output->getType() == ModelOutputType::OBJECT_DETECTION) {
    ModelBoxInfo *object_detection_output = (ModelBoxInfo *)output.get();
    TDL_InitObjectMeta(object_meta, object_detection_output->bboxes.size(), 0);
    for (int i = 0; i < object_detection_output->bboxes.size(); i++) {
      object_meta->info[i].box.x1 = object_detection_output->bboxes[i].x1;
      object_meta->info[i].box.y1 = object_detection_output->bboxes[i].y1;
      object_meta->info[i].box.x2 = object_detection_output->bboxes[i].x2;
      object_meta->info[i].box.y2 = object_detection_output->bboxes[i].y2;
      object_meta->info[i].class_id =
          object_detection_output->bboxes[i].class_id;
      object_meta->info[i].score = object_detection_output->bboxes[i].score;
    }
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_FaceDetection(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLFace *face_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
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
    if (box_landmark_output->box_landmarks.size() == 0) {
      return 0;
    }
    uint32_t num_landmark_per_face =
        box_landmark_output->box_landmarks[0].landmarks_x.size();
    TDL_InitFaceMeta(face_meta, box_landmark_output->box_landmarks.size(),
                     num_landmark_per_face);
    for (size_t i = 0; i < box_landmark_output->box_landmarks.size(); i++) {
      face_meta->info[i].box.x1 = box_landmark_output->box_landmarks[i].x1;
      face_meta->info[i].box.y1 = box_landmark_output->box_landmarks[i].y1;
      face_meta->info[i].box.x2 = box_landmark_output->box_landmarks[i].x2;
      face_meta->info[i].box.y2 = box_landmark_output->box_landmarks[i].y2;
      face_meta->width = box_landmark_output->box_landmarks[i].x2 -
                         box_landmark_output->box_landmarks[i].x1;
      face_meta->height = box_landmark_output->box_landmarks[i].y2 -
                          box_landmark_output->box_landmarks[i].y1;
      face_meta->info[i].score = box_landmark_output->box_landmarks[i].score;
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
  } else if (output->getType() == ModelOutputType::OBJECT_DETECTION) {
    ModelBoxInfo *object_detection_output = (ModelBoxInfo *)output.get();
    TDL_InitFaceMeta(face_meta, object_detection_output->bboxes.size(), 0);
    for (size_t i = 0; i < object_detection_output->bboxes.size(); i++) {
      face_meta->info[i].box.x1 = object_detection_output->bboxes[i].x1;
      face_meta->info[i].box.y1 = object_detection_output->bboxes[i].y1;
      face_meta->info[i].box.x2 = object_detection_output->bboxes[i].x2;
      face_meta->info[i].box.y2 = object_detection_output->bboxes[i].y2;
      face_meta->width = object_detection_output->bboxes[i].x2 -
                         object_detection_output->bboxes[i].x1;
      face_meta->height = object_detection_output->bboxes[i].y2 -
                          object_detection_output->bboxes[i].y1;
      face_meta->info[i].score = object_detection_output->bboxes[i].score;
    }
    face_meta->width = object_detection_output->image_width;
    face_meta->height = object_detection_output->image_height;
    face_meta->size = object_detection_output->bboxes.size();
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_Classfification(TDLHandle handle, const TDLModel model_id,
                            TDLImage image_handle, TDLClassInfo *class_info) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::CLASSIFICATION) {
    ModelClassificationInfo *classification_output =
        (ModelClassificationInfo *)output.get();
    class_info->class_id = classification_output->topk_class_ids[0];
    class_info->score = classification_output->topk_scores[0];
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_ObjectClassification(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle, TDLObject *object_meta,
                                 TDLClass *class_info) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  // TODO(fuquan.ke): crop object from image
  // TODO(fuquan.ke): inference
  // TODO(fuquan.ke): set class info
  return 0;
}

int32_t TDL_FaceAttribute(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLFace *face_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  TDL_InitFaceMeta(face_meta, 1, 0);
  std::shared_ptr<ModelBoxLandmarkInfo> face_info = convertFaceMeta(face_meta);
  int32_t ret = model->inference(image_context->image, face_info, outputs);
  if (ret != 0) {
    return ret;
  }

  for (int i = 0; i < outputs.size(); i++) {
    std::shared_ptr<ModelOutputInfo> output = outputs[i];
    ModelAttributeInfo *box_attribute_output =
        (ModelAttributeInfo *)output.get();
    face_meta->info[i].gender_score =
        box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GENDER];
    face_meta->info[i].age =
        box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_AGE];
    face_meta->info[i].glass_score =
        box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_GLASSES];
    face_meta->info[i].mask_score =
        box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_MASK];
  }

  return ret;
}

int32_t TDL_FaceLandmark(TDLHandle handle, const TDLModel model_id,
                         TDLImage image_handle, TDLFace *face_meta) {
  TDLContext *context = (TDLContext *)handle;
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

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  TDL_InitFaceMeta(face_meta, 1, 0);
  std::shared_ptr<ModelBoxLandmarkInfo> face_info = convertFaceMeta(face_meta);
  int32_t ret = model->inference(image_context->image, face_info, outputs);
  if (ret != 0) {
    return ret;
  }

  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  ModelLandmarksInfo *box_landmark_output = (ModelLandmarksInfo *)output.get();
  TDL_InitFaceMeta(face_meta, 1, box_landmark_output->landmarks_x.size());
  for (int i = 0; i < box_landmark_output->landmarks_x.size(); i++) {
    face_meta->info->landmarks.x[i] = box_landmark_output->landmarks_x[i];
    face_meta->info->landmarks.y[i] = box_landmark_output->landmarks_y[i];
  }
  face_meta->info->landmarks.score = box_landmark_output->landmarks_score[0];
  return 0;
}

int32_t TDL_Keypoint(TDLHandle handle, const TDLModel model_id,
                     TDLImage image_handle, TDLKeypoint *keypoint_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::OBJECT_LANDMARKS) {
    ModelLandmarksInfo *keypoint_output = (ModelLandmarksInfo *)output.get();
    TDL_InitKeypointMeta(keypoint_meta, keypoint_output->landmarks_x.size());
    keypoint_meta->width = keypoint_output->image_width;
    keypoint_meta->height = keypoint_output->image_height;
    for (size_t i = 0; i < keypoint_output->landmarks_x.size(); i++) {
      keypoint_meta->info[i].x = keypoint_output->landmarks_x[i];
      keypoint_meta->info[i].y = keypoint_output->landmarks_y[i];
      if (i < keypoint_output->landmarks_score.size()) {
        keypoint_meta->info[i].score = keypoint_output->landmarks_score[i];
      }
    }
  } else {
    LOGW("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_DetectionKeypoint(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLObject *object_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;

  std::shared_ptr<ModelBoxInfo> obj_info = convertObjMeta(object_meta);
  int32_t ret = model->inference(image_context->image, obj_info, outputs);
  if (ret != 0) {
    return ret;
  }
  for (int i = 0; i < outputs.size(); i++) {
    std::shared_ptr<ModelOutputInfo> output = outputs[i];
    if (output->getType() != ModelOutputType::OBJECT_LANDMARKS) {
      LOGW("Unsupported model output type: %d", output->getType());
      return -1;
    }
    ModelLandmarksInfo *keypoint_output = (ModelLandmarksInfo *)output.get();
    TDL_InitObjectMeta(object_meta, 1, keypoint_output->landmarks_x.size());
    object_meta->width = keypoint_output->image_width;
    object_meta->height = keypoint_output->image_height;
    object_meta->info[i].landmark_size = keypoint_output->landmarks_x.size();
    for (int j = 0; j < keypoint_output->landmarks_x.size(); j++) {
      object_meta->info[i].landmark_properity[j].x =
          keypoint_output->landmarks_x[j];
      object_meta->info[i].landmark_properity[j].y =
          keypoint_output->landmarks_y[j];
    }
  }
  return 0;
}

int32_t TDL_SemanticSegmentation(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle,
                                 TDLSegmentation *seg_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::SEGMENTATION) {
    ModelSegmentationInfo *segmentation_output =
        (ModelSegmentationInfo *)output.get();
    TDL_InitSemanticSegMeta(seg_meta, segmentation_output->output_width *
                                          segmentation_output->output_height);
    seg_meta->height = segmentation_output->image_height;
    seg_meta->width = segmentation_output->image_width;
    seg_meta->output_width = segmentation_output->output_width;
    seg_meta->output_height = segmentation_output->output_height;
    if (segmentation_output->class_id != nullptr) {
      seg_meta->class_id = segmentation_output->class_id;
      segmentation_output->class_id = nullptr;
    }
    if (segmentation_output->class_conf != nullptr) {
      seg_meta->class_conf = segmentation_output->class_conf;
      segmentation_output->class_conf = nullptr;
    }
  } else {
    LOGW("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_InstanceSegmentation(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle,
                                 TDLInstanceSeg *inst_seg_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }

  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() !=
      ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }

  ModelBoxSegmentationInfo *instance_seg_output =
      (ModelBoxSegmentationInfo *)output.get();
  TDL_InitInstanceSegMeta(
      inst_seg_meta, instance_seg_output->box_seg.size(),
      instance_seg_output->mask_width * instance_seg_output->mask_height);
  inst_seg_meta->width = instance_seg_output->image_width;
  inst_seg_meta->height = instance_seg_output->image_height;
  inst_seg_meta->mask_width = instance_seg_output->mask_width;
  inst_seg_meta->mask_height = instance_seg_output->mask_height;

  for (int i = 0; i < instance_seg_output->box_seg.size(); i++) {
    inst_seg_meta->info[i].obj_info->box.x1 =
        instance_seg_output->box_seg[i].x1;
    inst_seg_meta->info[i].obj_info->box.y1 =
        instance_seg_output->box_seg[i].y1;
    inst_seg_meta->info[i].obj_info->box.x2 =
        instance_seg_output->box_seg[i].x2;
    inst_seg_meta->info[i].obj_info->box.y2 =
        instance_seg_output->box_seg[i].y2;
    inst_seg_meta->info[i].obj_info->class_id =
        instance_seg_output->box_seg[i].class_id;
    inst_seg_meta->info[i].obj_info->score =
        instance_seg_output->box_seg[i].score;

    cv::Mat src(instance_seg_output->mask_height,
                instance_seg_output->mask_width, CV_8UC1,
                instance_seg_output->box_seg[i].mask,
                instance_seg_output->mask_width * sizeof(uint8_t));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);

    // find the longest contour
    int longest_index = -1;
    size_t max_length = 0;
    for (size_t i = 0; i < contours.size(); i++) {
      if (contours[i].size() > max_length) {
        max_length = contours[i].size();
        longest_index = i;
      }
    }

    if (longest_index >= 0 && max_length >= 1) {
      float ratio_height = (instance_seg_output->mask_height /
                            static_cast<float>(inst_seg_meta->height));
      float ratio_width = (instance_seg_output->mask_width /
                           static_cast<float>(inst_seg_meta->width));
      int source_y_offset, source_x_offset;
      if (ratio_height > ratio_width) {
        source_x_offset = 0;
        source_y_offset = (instance_seg_output->mask_height -
                           inst_seg_meta->height * ratio_width) /
                          2;
      } else {
        source_x_offset = (instance_seg_output->mask_width -
                           inst_seg_meta->width * ratio_height) /
                          2;
        source_y_offset = 0;
      }
      int source_region_height =
          instance_seg_output->mask_height - 2 * source_y_offset;
      int source_region_width =
          instance_seg_output->mask_width - 2 * source_x_offset;
      // calculate scaling factor
      float height_scale = static_cast<float>(inst_seg_meta->height) /
                           static_cast<float>(source_region_height);
      float width_scale = static_cast<float>(inst_seg_meta->width) /
                          static_cast<float>(source_region_width);
      instance_seg_output->box_seg[i].mask_point_size = max_length;
      instance_seg_output->box_seg[i].mask_point =
          new float[2 * max_length * sizeof(float)];
      size_t j = 0;
      for (const auto &point : contours[longest_index]) {
        instance_seg_output->box_seg[i].mask_point[2 * j] =
            (point.x - source_x_offset) * width_scale;
        instance_seg_output->box_seg[i].mask_point[2 * j + 1] =
            (point.y - source_y_offset) * height_scale;
        j++;
      }
    }

    if (instance_seg_output->box_seg[i].mask != nullptr) {
      inst_seg_meta->info[i].mask = instance_seg_output->box_seg[i].mask;
      instance_seg_output->box_seg[i].mask = nullptr;
    }
    if (instance_seg_output->box_seg[i].mask_point_size > 0) {
      inst_seg_meta->info[i].mask_point_size =
          instance_seg_output->box_seg[i].mask_point_size;
      inst_seg_meta->info[i].mask_point =
          instance_seg_output->box_seg[i].mask_point;
      instance_seg_output->box_seg[i].mask_point = nullptr;
    }
  }
  return 0;
}

int32_t TDL_FeatureExtraction(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLFeature *feature_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  memset(feature_meta, 0, sizeof(TDLFeature));
  if (output->getType() == ModelOutputType::FEATURE_EMBEDDING) {
    ModelFeatureInfo *feature_output = (ModelFeatureInfo *)output.get();
    feature_meta->size = feature_output->embedding_num;
    feature_meta->type = convertDataType(feature_output->embedding_type);
    feature_meta->ptr = (int8_t *)feature_output->embedding;  // transfer
                                                              // ownership to
                                                              // feature_meta
    feature_output->embedding = nullptr;  // set to null to prevent release
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_LaneDetection(TDLHandle handle, const TDLModel model_id,
                          TDLImage image_handle, TDLLane *lane_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
    ModelBoxLandmarkInfo *lane_output = (ModelBoxLandmarkInfo *)output.get();

    TDL_InitLaneMeta(lane_meta, lane_output->box_landmarks.size());
    lane_meta->width = lane_output->image_width;
    lane_meta->height = lane_output->image_height;

    for (size_t j = 0; j < lane_output->box_landmarks.size(); j++) {
      for (int k = 0; k < 2; k++) {
        lane_meta->lane[j].x[k] = lane_output->box_landmarks[j].landmarks_x[k];
        lane_meta->lane[j].y[k] = lane_output->box_landmarks[j].landmarks_y[k];
      }
    }
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }

  return 0;
}

int32_t TDL_CharacterRecognition(TDLHandle handle, const TDLModel model_id,
                                 TDLImage image_handle, TDLOcr *char_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::OCR_INFO) {
    ModelOcrInfo *char_output = (ModelOcrInfo *)output.get();
    TDL_InitCharacterMeta(char_meta, char_output->length);
    char_meta->size = char_output->length;
    char_meta->text_info = char_output->text_info;
    char_output->text_info = NULL;
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }

  return 0;
}

int32_t TDL_Tracking(TDLHandle handle, int frame_id, TDLFace *face_meta,
                     TDLObject *obj_meta, TDLTracker *track_meta) {
  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_MOT_SORT);
  if (tracker == nullptr) {
    LOGE("Failed to create tracker\n");
    return -1;
  }

  std::map<TDLObjectType, TDLObjectType> object_pair_config;
  object_pair_config[TDLObjectType::OBJECT_TYPE_FACE] =
      TDLObjectType::OBJECT_TYPE_PERSON;
  tracker->setPairConfig(object_pair_config);
  std::vector<ObjectBoxInfo> det_results;
  std::vector<TrackerInfo> track_results;

  if (face_meta != nullptr && face_meta->info != nullptr) {
    for (uint32_t i = 0; i < face_meta->size; i++) {
      ObjectBoxInfo box;
      box.x1 = face_meta->info[i].box.x1;
      box.y1 = face_meta->info[i].box.y1;
      box.x2 = face_meta->info[i].box.x2;
      box.y2 = face_meta->info[i].box.y2;
      box.score = face_meta->info[i].score;
      box.class_id = 0;
      det_results.push_back(box);
    }
    tracker->setImgSize(face_meta->width, face_meta->height);
  }

  if (obj_meta != nullptr && obj_meta->info != nullptr) {
    for (uint32_t i = 0; i < obj_meta->size; i++) {
      ObjectBoxInfo box;
      box.x1 = obj_meta->info[i].box.x1;
      box.y1 = obj_meta->info[i].box.y1;
      box.x2 = obj_meta->info[i].box.x2;
      box.y2 = obj_meta->info[i].box.y2;
      box.score = obj_meta->info[i].score;
      box.class_id = obj_meta->info[i].class_id;
      det_results.push_back(box);
    }
    tracker->setImgSize(obj_meta->width, obj_meta->height);
  }

  tracker->track(det_results, frame_id, track_results);

  TDL_InitTrackMeta(track_meta, track_results.size());
  for (int i = 0; i < track_results.size(); i++) {
    TrackerInfo track_info = track_results[i];
    track_meta->info[i].id = track_info.track_id_;
    track_meta->info[i].bbox.x1 = track_info.box_info_.x1;
    track_meta->info[i].bbox.x2 = track_info.box_info_.x2;
    track_meta->info[i].bbox.y1 = track_info.box_info_.y1;
    track_meta->info[i].bbox.y2 = track_info.box_info_.y2;
  }
  return 0;
}
