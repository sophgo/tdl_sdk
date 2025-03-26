#include "tdl_sdk.h"

#include "common/common_types.hpp"
#include "tdl_type_internal.hpp"
#include "tdl_utils.h"
#include "utils/tdl_log.hpp"

std::shared_ptr<BaseModel> get_model(tdl_handle_t handle,
                                     const tdl_model_e model_id) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return nullptr;
  }
  if (context->models.find(model_id) == context->models.end()) {
    LOGW("model %d not found", model_id);
    return nullptr;
  }
  return context->models[model_id];
}

tdl_handle_t TDL_CreateHandle(const int32_t tpu_device_id) {
  tdl_context_t *context = new tdl_context_t();
  return (tdl_handle_t)context;
}

int32_t TDL_DestroyHandle(tdl_handle_t handle) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  for (auto &model : context->models) {
    TDL_CloseModel(handle, model.first);
  }
  delete context;
  return 0;
}

tdl_image_t TDL_WrapVPSSFrame(void *vpss_frame, bool own_memory) {
  if (vpss_frame == nullptr) {
    return nullptr;
  }

  // TODO(fuquan.ke): use own_memory to create VPSSFrame
  tdl_image_context_t *image_context = new tdl_image_context_t();
  image_context->image = ImageFactory::wrapVPSSFrame(vpss_frame, own_memory);
  return (tdl_image_t)image_context;
}

tdl_image_t TDL_ReadImage(const char *path) {
  tdl_image_context_t *image_context = new tdl_image_context_t();
  image_context->image =
      ImageFactory::readImage(path, false, InferencePlatform::CVITEK);
  return (tdl_image_t)image_context;
}

tdl_image_t TDL_ReadAudio(const char *path, int size){
  FILE *file = fopen(path, "rb");
  if (!file) {
      return 0;
  }
  unsigned char *buffer = (unsigned char *)malloc(size);
  size_t read_size = fread(buffer, 1, size, file);
  fclose(file);

  tdl_image_context_t *image_context = new tdl_image_context_t();
  image_context->image =
      ImageFactory::createImage(size, 1, ImageFormat::GRAY, TDLDataType::UINT8, true);
  uint8_t* data_buffer = image_context->image->getVirtualAddress()[0];
  memcpy(data_buffer, buffer, size * sizeof(uint8_t));
  return (tdl_image_t)image_context;
}

int32_t TDL_DestroyImage(tdl_image_t image_handle) {
  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;
  if (image_context == nullptr) {
    return -1;
  }
  delete image_context;
  return 0;
}

int32_t TDL_OpenModel(tdl_handle_t handle, const tdl_model_e model_id,
                          const char *model_path) {
  tdl_context_t *context = (tdl_context_t *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->models.find(model_id) != context->models.end()) {
    return 0;
  }
  ModelType model_type = convert_model_type(model_id);
  std::shared_ptr<BaseModel> model =
      context->model_factory->getModel(model_type, model_path);
  if (model == nullptr) {
    return -1;
  }
  context->models[model_id] = model;
  return 0;
}

int32_t TDL_CloseModel(tdl_handle_t handle,
                           const tdl_model_e model_id) {
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

int32_t TDL_Detection(tdl_handle_t handle, const tdl_model_e model_id,
                          tdl_image_t image_handle,
                          tdl_object_t *object_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
    ModelBoxLandmarkInfo *object_Landmark_output = (ModelBoxLandmarkInfo *)output.get();
    TDL_InitObjectMeta(
        object_meta,
        object_Landmark_output->box_landmarks.size(),
        object_Landmark_output->box_landmarks[0].landmarks_x.size());
    for (int i = 0; i < object_Landmark_output->box_landmarks.size(); i ++) {
      object_meta->info[i].box.x1 = object_Landmark_output->box_landmarks[i].x1;
      object_meta->info[i].box.y1 = object_Landmark_output->box_landmarks[i].y1;
      object_meta->info[i].box.x2 = object_Landmark_output->box_landmarks[i].x2;
      object_meta->info[i].box.y2 = object_Landmark_output->box_landmarks[i].y2;
      object_meta->info[i].class_id = object_Landmark_output->box_landmarks[i].class_id;
      object_meta->info[i].score = object_Landmark_output->box_landmarks[i].score;
      for (int j = 0; j < object_Landmark_output->box_landmarks[i].landmarks_x.size(); j++) {
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
      object_meta->info[i].class_id = object_detection_output->bboxes[i].class_id;
      object_meta->info[i].score = object_detection_output->bboxes[i].score;
    }
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
  return 0;
}

int32_t TDL_FaceDetection(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_face_t *face_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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

int32_t TDL_Classfification(tdl_handle_t handle,
                                const tdl_model_e model_id,
                                tdl_image_t image_handle,
                                tdl_class_info_t *class_info) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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

int32_t TDL_ObjectClassification(tdl_handle_t handle,
                                     const tdl_model_e model_id,
                                     tdl_image_t image_handle,
                                     tdl_object_t *object_meta,
                                     tdl_class_t *class_info) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  // TODO(fuquan.ke): crop object from image
  // TODO(fuquan.ke): inference
  // TODO(fuquan.ke): set class info
  return 0;
}

int32_t TDL_FaceAttribute(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_face_t *face_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;

  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  TDL_InitFaceMeta(face_meta, 1, 0);
  std::shared_ptr<ModelBoxLandmarkInfo> face_info =
      convert_face_meta(face_meta);
  int32_t ret = model->inference(image_context->image, face_info, outputs);
  if (ret != 0) {
    return ret;
  }

  for (int i = 0; i < outputs.size(); i++) {
    std::shared_ptr<ModelOutputInfo> output = outputs[i];
    ModelAttributeInfo *box_attribute_output = (ModelAttributeInfo *)output.get();
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

int32_t TDL_FaceLandmark(tdl_handle_t handle,
                             const tdl_model_e model_id,
                             tdl_image_t image_handle,
                             tdl_face_t *face_meta) {
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

  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;

  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  TDL_InitFaceMeta(face_meta, 1, 0);
  std::shared_ptr<ModelBoxLandmarkInfo> face_info =
      convert_face_meta(face_meta);
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

int32_t TDL_Keypoint(tdl_handle_t handle,
                     const tdl_model_e model_id,
                     tdl_image_t image_handle,
                     tdl_keypoint_t *keypoint_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  if (output->getType() == ModelOutputType::OBJECT_LANDMARKS) {
    ModelLandmarksInfo *keypoint_output = (ModelLandmarksInfo *)output.get();
    TDL_InitKeypointMeta(keypoint_meta,
                             keypoint_output->landmarks_x.size());
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

int32_t TDL_DetectionKeypoint(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_object_t *object_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  tdl_image_context_t *image_context = (tdl_image_context_t *)image_handle;
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;

  std::shared_ptr<ModelBoxInfo> obj_info =
      convert_obj_meta(object_meta);
  int32_t ret = model->inference(image_context->image, obj_info, outputs);
  if (ret != 0) {
    return ret;
  }    
  for (int i = 0; i < outputs.size(); i ++) {
    std::shared_ptr<ModelOutputInfo> output = outputs[i];
    if (output->getType() != ModelOutputType::OBJECT_LANDMARKS) {
      LOGW("Unsupported model output type: %d", output->getType());
      return -1;
    }
    ModelLandmarksInfo *keypoint_output = (ModelLandmarksInfo *)output.get();
    TDL_InitObjectMeta(object_meta, 1,
      keypoint_output->landmarks_x.size());
    object_meta->width = keypoint_output->image_width;
    object_meta->height = keypoint_output->image_height;
    object_meta->info[i].landmark_size = keypoint_output->landmarks_x.size();
    for (int j = 0; j < keypoint_output->landmarks_x.size(); j++) {
      object_meta->info[i].landmark_properity[j].x = keypoint_output->landmarks_x[j];
      object_meta->info[i].landmark_properity[j].y = keypoint_output->landmarks_y[j];
    }
  }
  return 0;
}

int32_t TDL_SemanticSegmentation(tdl_handle_t handle,
                                     const tdl_model_e model_id,
                                     tdl_image_t image_handle,
                                     tdl_seg_t *seg_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  if (output->getType() == ModelOutputType::SEGMENTATION) {
    ModelSegmentationInfo *segmentation_output =
        (ModelSegmentationInfo *)output.get();
    TDL_InitSemanticSegMeta(
        seg_meta,
        segmentation_output->output_width * segmentation_output->output_height);
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

int32_t TDL_InstanceSegmentation(tdl_handle_t handle,
                                     const tdl_model_e model_id,
                                     tdl_image_t image_handle,
                                     tdl_instance_seg_t *inst_seg_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  if (output->getType() !=
      ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }

  ModelBoxSegmentationInfo *instance_seg_output =
    (ModelBoxSegmentationInfo *)output.get();
  TDL_InitInstanceSegMeta(
      inst_seg_meta,
      instance_seg_output->box_seg.size(),
      instance_seg_output->mask_width * instance_seg_output->mask_height);
  inst_seg_meta->width = instance_seg_output->image_width;
  inst_seg_meta->height = instance_seg_output->image_height;
  inst_seg_meta->mask_width = instance_seg_output->mask_width;
  inst_seg_meta->mask_height = instance_seg_output->mask_height;

  for (int i = 0; i < instance_seg_output->box_seg.size(); i++) {
    inst_seg_meta->info[i].obj_info->box.x1 = instance_seg_output->box_seg[i].x1;
    inst_seg_meta->info[i].obj_info->box.y1 = instance_seg_output->box_seg[i].y1;
    inst_seg_meta->info[i].obj_info->box.x2 = instance_seg_output->box_seg[i].x2;
    inst_seg_meta->info[i].obj_info->box.y2 = instance_seg_output->box_seg[i].y2;
    inst_seg_meta->info[i].obj_info->class_id = instance_seg_output->box_seg[i].class_id;
    inst_seg_meta->info[i].obj_info->score = instance_seg_output->box_seg[i].score;

    cv::Mat src(instance_seg_output->mask_height, instance_seg_output->mask_width, CV_8UC1, instance_seg_output->box_seg[i].mask,
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
      float ratio_height = (instance_seg_output->mask_height / static_cast<float>(inst_seg_meta->height));
      float ratio_width = (instance_seg_output->mask_width / static_cast<float>(inst_seg_meta->width));
      int source_y_offset, source_x_offset;
      if (ratio_height > ratio_width) {
        source_x_offset = 0;
        source_y_offset = (instance_seg_output->mask_height - inst_seg_meta->height * ratio_width) / 2;
      } else {
        source_x_offset = (instance_seg_output->mask_width - inst_seg_meta->width * ratio_height) / 2;
        source_y_offset = 0;
      }
      int source_region_height = instance_seg_output->mask_height - 2 * source_y_offset;
      int source_region_width = instance_seg_output->mask_width - 2 * source_x_offset;
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
      inst_seg_meta->info[i].mask_point_size = instance_seg_output->box_seg[i].mask_point_size;
      inst_seg_meta->info[i].mask_point = instance_seg_output->box_seg[i].mask_point;
      instance_seg_output->box_seg[i].mask_point = nullptr;
    }
  }
  return 0;
}

int32_t TDL_FeatureExtraction(tdl_handle_t handle,
                                  const tdl_model_e model_id,
                                  tdl_image_t image_handle,
                                  tdl_feature_t *feature_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  memset(feature_meta, 0, sizeof(tdl_feature_t));
  if (output->getType() == ModelOutputType::FEATURE_EMBEDDING) {
    ModelFeatureInfo *feature_output = (ModelFeatureInfo *)output.get();
    feature_meta->size = feature_output->embedding_num;
    feature_meta->type = convert_data_type(feature_output->embedding_type);
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

int32_t TDL_LaneDetection(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_lane_t *lane_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  if (output->getType() == ModelOutputType::OBJECT_LANDMARKS) {
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

int32_t TDL_CharacterRecognition(tdl_handle_t handle,
                              const tdl_model_e model_id,
                              tdl_image_t image_handle,
                              tdl_ocr_t *char_meta) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
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
  if (output->getType() == ModelOutputType::OCR_INFO) {
    ModelOcrInfo *char_output = (ModelOcrInfo *)output.get();
    TDL_InitCharacterMeta(char_meta, char_output->length);
    char_meta->size = char_output->length;
    char_meta->text_info = char_output->text_info;
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }

  return 0;
}