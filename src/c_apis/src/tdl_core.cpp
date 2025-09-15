#include "tdl_sdk.h"

#include <cstring>
#include <opencv2/opencv.hpp>
#include "app/app_data_types.hpp"
#include "common/common_types.hpp"
#include "consumer_counting/consumer_counting_app.hpp"
#include "tdl_type_internal.hpp"
#include "tdl_utils.h"
#include "tracker/tracker_types.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

static std::shared_ptr<BaseModel> get_model(TDLHandle handle,
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
  if (context->app_task) {
    context->app_task->release();
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

int32_t TDL_WrapImage(TDLImage image, void *frame) {
  if (image == nullptr) {
    LOGE("Invalid input parameters: image or frame is null");
    return -1;
  }

  TDLImageContext *image_context = (TDLImageContext *)image;

  if (image_context->image == nullptr) {
    LOGE("Invalid image context: image is null");
    return -1;
  }

  *(void **)frame = image_context->image->getInternalData();
  if (frame == nullptr) {
    LOGE("Failed to get internal data from image");
    return -1;
  }
  return 0;
}

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
    LOGE("file size %ld is not aligned with data type size %d", file_size,
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

  if (model_config_json != nullptr) {
    factory.loadModelConfig(model_config_json);
  }

  ModelConfig model_config = factory.getModelConfig(model_type);
  std::shared_ptr<BaseModel> model =
      factory.getModel(model_type, str_model_path, model_config);
  if (model == nullptr) {
    return -1;
  }
  context->models[model_id] = model;
  return 0;
}

int32_t TDL_OpenModelFromBuffer(TDLHandle handle, const TDLModel model_id,
                                const uint8_t *model_buffer,
                                uint32_t model_buffer_size,
                                const char *model_config_json) {
  TDLContext *context = (TDLContext *)handle;
  if (context->models.find(model_id) != context->models.end()) {
    return 0;
  }
  ModelType model_type = convertModelType(model_id);
  TDLModelFactory &factory = TDLModelFactory::getInstance();

  if (model_config_json != nullptr) {
    factory.loadModelConfig(model_config_json);
  }

  ModelConfig model_config = factory.getModelConfig(model_type);
  std::shared_ptr<BaseModel> model = factory.getModel(
      model_type, model_buffer, model_buffer_size, model_config);
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
    for (size_t i = 0; i < object_Landmark_output->box_landmarks.size(); i++) {
      object_meta->info[i].box.x1 = object_Landmark_output->box_landmarks[i].x1;
      object_meta->info[i].box.y1 = object_Landmark_output->box_landmarks[i].y1;
      object_meta->info[i].box.x2 = object_Landmark_output->box_landmarks[i].x2;
      object_meta->info[i].box.y2 = object_Landmark_output->box_landmarks[i].y2;
      object_meta->info[i].class_id =
          object_Landmark_output->box_landmarks[i].class_id;
      object_meta->info[i].score =
          object_Landmark_output->box_landmarks[i].score;
      for (size_t j = 0;
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
    for (size_t i = 0; i < object_detection_output->bboxes.size(); i++) {
      object_meta->info[i].box.x1 = object_detection_output->bboxes[i].x1;
      object_meta->info[i].box.y1 = object_detection_output->bboxes[i].y1;
      object_meta->info[i].box.x2 = object_detection_output->bboxes[i].x2;
      object_meta->info[i].box.y2 = object_detection_output->bboxes[i].y2;
      object_meta->info[i].class_id =
          object_detection_output->bboxes[i].class_id;
      object_meta->info[i].score = object_detection_output->bboxes[i].score;
    }
  } else {
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
    return -1;
  }
  return 0;
}

int32_t TDL_Classification(TDLHandle handle, const TDLModel model_id,
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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

int32_t TDL_IspClassification(TDLHandle handle, const TDLModel model_id,
                              TDLImage image_handle, TDLIspMeta *isp_meta,
                              TDLClass *class_info) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }
  std::vector<std::shared_ptr<BaseImage>> images;
  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  images.push_back(image_context->image);
  std::map<std::string, float> isp_data;
  isp_data["awb[0]"] = isp_meta->awb[0];
  isp_data["awb[1]"] = isp_meta->awb[1];
  isp_data["awb[2]"] = isp_meta->awb[2];
  isp_data["ccm[0]"] = isp_meta->ccm[0];
  isp_data["ccm[1]"] = isp_meta->ccm[1];
  isp_data["ccm[2]"] = isp_meta->ccm[2];
  isp_data["ccm[3]"] = isp_meta->ccm[3];
  isp_data["ccm[4]"] = isp_meta->ccm[4];
  isp_data["ccm[5]"] = isp_meta->ccm[5];
  isp_data["ccm[6]"] = isp_meta->ccm[6];
  isp_data["ccm[7]"] = isp_meta->ccm[7];
  isp_data["ccm[8]"] = isp_meta->ccm[8];
  isp_data["blc"] = isp_meta->blc;
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  int32_t ret = model->inference(images, outputs, isp_data);
  if (ret != 0) {
    return ret;
  }
  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  if (output->getType() == ModelOutputType::CLASSIFICATION) {
    ModelClassificationInfo *classification_output =
        (ModelClassificationInfo *)output.get();
    class_info->size = classification_output->topk_class_ids.size();
    for (int i = 0; i < class_info->size; i++) {
      class_info->info[i].class_id = classification_output->topk_class_ids[i];
      class_info->info[i].score = classification_output->topk_scores[i];
    }
  } else {
    LOGE("Unsupported model output type: %d", output->getType());
    return -1;
  }
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

  for (size_t i = 0; i < outputs.size(); i++) {
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
    face_meta->info[i].emotion_score =
        box_attribute_output->attributes[OBJECT_ATTRIBUTE_HUMAN_EMOTION];
  }

  return ret;
}

int32_t TDL_FaceLandmark(TDLHandle handle, const TDLModel model_id,
                         TDLImage image_handle, TDLImage *crop_image_handle,
                         TDLFace *face_meta) {
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
  std::vector<std::shared_ptr<BaseImage>> images;
  std::shared_ptr<BaseImage> target_img;
  int32_t ret = 0;
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  TDL_InitFaceMeta(face_meta, 1, 0);
  std::shared_ptr<ModelBoxLandmarkInfo> face_info = convertFaceMeta(face_meta);

  if (face_meta->info->box.x1 != 0 || face_meta->info->box.x2 != 0 ||
      face_meta->info->box.y1 != 0 || face_meta->info->box.y2 != 0) {
    std::shared_ptr<BasePreprocessor> preprocessor = model->getPreprocessor();
    int32_t img_width = (int32_t)image_context->image->getWidth();
    int32_t img_height = (int32_t)image_context->image->getHeight();
    // 1. 计算原始框的宽高和中心
    float orig_w = face_meta->info->box.x2 - face_meta->info->box.x1;
    float orig_h = face_meta->info->box.y2 - face_meta->info->box.y1;
    float cx = face_meta->info->box.x1 + orig_w * 0.5f;
    float cy = face_meta->info->box.y1 + orig_h * 0.5f;
    // 2. 决定裁剪区域的方形区域
    int32_t crop_w = int32_t(std::max(orig_w, orig_h) + 0.5f);
    int32_t crop_h = crop_w;
    // 3. 如果有最小裁剪限度，也可以在这里保证不小于它
    float scale = crop_w < 128 ? 128.0f / crop_w : 256.0f / crop_w;

    int32_t dst_width = int32_t(crop_w * scale + 0.5f);
    int32_t dst_height = int32_t(crop_h * scale + 0.5f);

    // 4. 以中心为基准，计算左上角
    int32_t crop_x = int32_t(cx - crop_w * 0.5f + 0.5f);
    int32_t crop_y = int32_t(cy - crop_h * 0.5f + 0.5f);

    // 5. 边界控制：保证整个裁剪框在图像内部
    if (crop_x < 0) crop_x = 0;
    if (crop_y < 0) crop_y = 0;
    if (crop_x + crop_w > img_width) crop_x = std::max(0, img_width - crop_w);
    if (crop_y + crop_h > img_height) crop_y = std::max(0, img_height - crop_h);
    target_img =
        preprocessor->cropResize(image_context->image, crop_x, crop_y, crop_w,
                                 crop_w, dst_width, dst_height);
  } else {
    target_img = image_context->image;
  }

  images.push_back(target_img);
  ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }

  std::shared_ptr<ModelOutputInfo> output = outputs[0];
  ModelLandmarksInfo *box_landmark_output = (ModelLandmarksInfo *)output.get();
  int32_t landmarks_cnt = box_landmark_output->landmarks_x.size();
  TDL_InitFaceMeta(face_meta, 1, landmarks_cnt);
  for (size_t i = 0; i < landmarks_cnt; i++) {
    face_meta->info->landmarks.x[i] = box_landmark_output->landmarks_x[i];
    face_meta->info->landmarks.y[i] = box_landmark_output->landmarks_y[i];
  }
  face_meta->info->landmarks.score = box_landmark_output->landmarks_score[0];
  face_meta->info->landmarks.size = landmarks_cnt;

  if (crop_image_handle != NULL) {
    float dst_landmarks[landmarks_cnt * 2];
    for (int i = 0; i < landmarks_cnt; i++) {
      dst_landmarks[2 * i] = face_meta->info->landmarks.x[i];
      dst_landmarks[2 * i + 1] = face_meta->info->landmarks.y[i];
    }
    std::shared_ptr<BaseImage> face_crop = ImageFactory::alignFace(
        target_img, dst_landmarks, nullptr, landmarks_cnt, nullptr);

    TDLImageContext *image_crop_context = new TDLImageContext();
    image_crop_context->image = face_crop;
    *crop_image_handle = reinterpret_cast<TDLImage>(image_crop_context);
  }

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
                              TDLImage image_handle, TDLObject *object_meta,
                              TDLImage *crop_image_handle) {
  std::shared_ptr<BaseModel> model = get_model(handle, model_id);
  if (model == nullptr) {
    return -1;
  }

  TDLImageContext *image_context = (TDLImageContext *)image_handle;
  std::vector<std::shared_ptr<BaseImage>> images;
  std::vector<std::shared_ptr<ModelOutputInfo>> outputs;
  std::shared_ptr<BasePreprocessor> preprocessor = model->getPreprocessor();
  int32_t ret = 0;
  int32_t keypoint_cnt = 0;
  std::shared_ptr<ModelBoxInfo> obj_info = convertObjMeta(object_meta);
  if (object_meta->size != 0) {
    for (int32_t i = 0; i < object_meta->size; i++) {
      std::shared_ptr<BaseImage> target_img;
      int32_t width = (int32_t)object_meta->info[i].box.x2 -
                      (int32_t)object_meta->info[i].box.x1;
      int32_t height = (int32_t)object_meta->info[i].box.y2 -
                       (int32_t)object_meta->info[i].box.y1;
      float expansion_factor = 1.25f;
      int32_t new_width = static_cast<int32_t>(width * expansion_factor);
      int32_t new_height = static_cast<int32_t>(height * expansion_factor);
      int32_t crop_x =
          (int32_t)object_meta->info[i].box.x1 - (new_width - width) / 2;
      int32_t crop_y =
          (int32_t)object_meta->info[i].box.y1 - (new_height - height) / 2;
      target_img = preprocessor->crop(image_context->image, crop_x, crop_y,
                                      new_width, new_height);
      images.push_back(target_img);
    }
  } else {
    images.push_back(image_context->image);
  }
  ret = model->inference(images, outputs);
  if (ret != 0) {
    return ret;
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    std::shared_ptr<ModelOutputInfo> output = outputs[i];
    if (output->getType() != ModelOutputType::OBJECT_LANDMARKS) {
      LOGW("Unsupported model output type: %d", output->getType());
      return -1;
    }
    ModelLandmarksInfo *keypoint_output = (ModelLandmarksInfo *)output.get();
    keypoint_cnt = keypoint_output->landmarks_x.size();
    TDL_InitObjectMeta(object_meta, object_meta->size, keypoint_cnt);
    object_meta->width = keypoint_output->image_width;
    object_meta->height = keypoint_output->image_height;
    object_meta->info[i].landmark_size = keypoint_cnt;
    for (size_t j = 0; j < keypoint_cnt; j++) {
      object_meta->info[i].landmark_properity[j].x =
          keypoint_output->landmarks_x[j];
      object_meta->info[i].landmark_properity[j].y =
          keypoint_output->landmarks_y[j];
    }

    if (crop_image_handle != nullptr) {
      TDLImageContext *image_crop_context = new TDLImageContext();
      if (model_id == TDL_MODEL_KEYPOINT_LICENSE_PLATE) {
        float dst_keypoints[keypoint_cnt * 2];
        for (int32_t k = 0; k < keypoint_cnt; k++) {
          dst_keypoints[2 * k] = object_meta->info[i].landmark_properity[k].x;
          dst_keypoints[2 * k + 1] =
              object_meta->info[i].landmark_properity[k].y;
        }
        std::shared_ptr<BaseImage> license_plate_align =
            ImageFactory::alignLicensePlate(images[i], dst_keypoints, nullptr,
                                            4, nullptr);
        image_crop_context->image = license_plate_align;
      } else {
        image_crop_context->image = images[i];
      }
      crop_image_handle[i] = reinterpret_cast<TDLImage>(image_crop_context);
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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

  for (size_t i = 0; i < instance_seg_output->box_seg.size(); i++) {
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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
    LOGE("Unsupported model output type: %d",
         static_cast<int>(output->getType()));
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
  for (size_t i = 0; i < track_results.size(); i++) {
    TrackerInfo track_info = track_results[i];
    track_meta->info[i].id = track_info.track_id_;
    track_meta->info[i].bbox.x1 = track_info.box_info_.x1;
    track_meta->info[i].bbox.x2 = track_info.box_info_.x2;
    track_meta->info[i].bbox.y1 = track_info.box_info_.y1;
    track_meta->info[i].bbox.y2 = track_info.box_info_.y2;
  }
  return 0;
}

int32_t TDL_SetSingleObjectTracking(TDLHandle handle, TDLImage image_handle,
                                    TDLObject *object_meta, int *set_values,
                                    int size) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->tracker == nullptr) {
    LOGI(" to init context->tracker \n");
    context->tracker = TrackerFactory::createTracker(TrackerType::TDL_SOT);
    std::shared_ptr<BaseModel> sot_model =
        get_model(handle, TDLModel::TDL_MODEL_TRACKING_FEARTRACK);
    context->tracker->setModel(sot_model);
  }

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  if (set_values == nullptr) {
    LOGE("set_values is nullptr \n");
    return -1;
  }

  std::vector<ObjectBoxInfo> bboxes;
  for (int i = 0; i < object_meta->size; i++) {
    ObjectBoxInfo box;
    box.x1 = object_meta->info[i].box.x1;
    box.y1 = object_meta->info[i].box.y1;
    box.x2 = object_meta->info[i].box.x2;
    box.y2 = object_meta->info[i].box.y2;
    box.score = object_meta->info[i].score;
    bboxes.push_back(box);
  }

  if (size == 1) {
    return context->tracker->initialize(image_context->image, bboxes,
                                        set_values[0]);
  }

  if (size == 2) {
    return context->tracker->initialize(image_context->image, bboxes,
                                        set_values[0], set_values[1]);

  } else if (size == 4) {
    ObjectBoxInfo init_bbox;
    init_bbox.x1 = set_values[0];
    init_bbox.y1 = set_values[1];
    init_bbox.x2 = set_values[2];
    init_bbox.y2 = set_values[3];
    init_bbox.score = 1.0f;

    return context->tracker->initialize(image_context->image, bboxes,
                                        init_bbox);
  } else {
    LOGE("set_values size should be 1 or 2 or 4, but got %d", size);
    return -1;
  }
}

int32_t TDL_SingleObjectTracking(TDLHandle handle, TDLImage image_handle,
                                 TDLTracker *track_meta, uint64_t frame_id) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->tracker == nullptr) {
    LOGE("context->tracker is nullptr \n");
    return -1;
  }

  TDLImageContext *image_context = (TDLImageContext *)image_handle;

  TrackerInfo tracker_info;
  context->tracker->track(image_context->image, frame_id, tracker_info);

  if (tracker_info.status_ != TrackStatus::LOST) {
    TDL_InitTrackMeta(track_meta, 1);
    track_meta->info[0].id = tracker_info.track_id_;
    track_meta->info[0].bbox.x1 = tracker_info.box_info_.x1;
    track_meta->info[0].bbox.y1 = tracker_info.box_info_.y1;
    track_meta->info[0].bbox.x2 = tracker_info.box_info_.x2;
    track_meta->info[0].bbox.y2 = tracker_info.box_info_.y2;
  } else {
    LOGI("tracker_info.status_ is LOST");
  }
  return 0;
}

int32_t TDL_IntrusionDetection(TDLHandle handle, TDLPoints *regions,
                               TDLBox *box, bool *is_intrusion) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    LOGE("Invalid handle");
    return -1;
  }

  if (regions == nullptr || box == nullptr || regions->size == 0) {
    LOGE("Invalid input parameters");
    return -1;
  }

  if (context->intrusion_detect == nullptr) {
    context->intrusion_detect = std::make_shared<IntrusionDetection>();
    if (context->intrusion_detect == nullptr) {
      LOGE("Failed to create intrusion detect");
      return -1;
    }
  }

  context->intrusion_detect->clean();

  PointsInfo points_info;
  points_info.x.assign(regions->x, regions->x + regions->size);
  points_info.y.assign(regions->y, regions->y + regions->size);

  int ret = context->intrusion_detect->addRegion(points_info);
  if (ret != 0) {
    LOGE("Failed to add region");
    return -1;
  }

  ObjectBoxInfo bbox;
  bbox.x1 = box->x1;
  bbox.y1 = box->y1;
  bbox.x2 = box->x2;
  bbox.y2 = box->y2;

  *is_intrusion = context->intrusion_detect->isIntrusion(bbox);

  return 0;
}

#if defined(__CV181X__) || defined(__CV184X__)

int32_t TDL_MotionDetection(TDLHandle handle, TDLImage background,
                            TDLImage detect_image, TDLObject *roi,
                            uint8_t threshold, double min_area,
                            TDLObject *obj_meta,
                            uint32_t background_update_interval) {
  TDLContext *context = (TDLContext *)handle;
  int ret = 0;
  if (context == nullptr) {
    return -1;
  }
  // 如果motion detection没有初始化，则初始化
  if (context->md == nullptr) {
    context->md = MotionDetection::getMotionDetection();
    if (context->md == nullptr) {
      LOGE("Failed to create motion detection\n");
      return -1;
    }
  }
  // 背景图和检测图转换为灰度图
  TDLImageContext *background_image_context = (TDLImageContext *)background;
  TDLImageContext *detect_image_context = (TDLImageContext *)detect_image;
  ImageFormat image_format = background_image_context->image->getImageFormat();
  if (image_format != ImageFormat::GRAY &&
      image_format != ImageFormat::BGR_PACKED &&
      image_format != ImageFormat::YUV420SP_VU) {
    LOGE("Invalid background image format: %d\n",
         static_cast<int>(image_format));
    return -1;
  }
  if (image_format == ImageFormat::YUV420SP_VU) {
    TDLImage background_gray_image;
    TDLImage detect_gray_image;
    TDL_NV21ToGray(background, &background_gray_image);
    TDL_NV21ToGray(detect_image, &detect_gray_image);
    background_image_context = (TDLImageContext *)background_gray_image;
    detect_image_context = (TDLImageContext *)detect_gray_image;
  } else if (image_format == ImageFormat::BGR_PACKED) {
    TDLImage background_gray_image;
    TDLImage detect_gray_image;
    TDL_BGRPACKEDToGray(background, &background_gray_image);
    TDL_BGRPACKEDToGray(detect_image, &detect_gray_image);
    background_image_context = (TDLImageContext *)background_gray_image;
    detect_image_context = (TDLImageContext *)detect_gray_image;
  }
  if (context->md->background_update_count_ != background_update_interval) {
    context->md->background_update_count_ += 1;
  } else {
    ret = context->md->setBackground(background_image_context->image);
    if (ret != 0) {
      LOGE("Failed to set background image\n");
      return -1;
    }
    context->md->background_update_count_ = 0;
  }

  if (roi->size > 0 && context->md->isROIEmpty()) {
    std::vector<ObjectBoxInfo> roi_s;
    for (size_t i = 0; i < roi->size; i++) {
      ObjectBoxInfo box;
      box.x1 = roi->info[i].box.x1;
      box.y1 = roi->info[i].box.y1;
      box.x2 = roi->info[i].box.x2;
      box.y2 = roi->info[i].box.y2;
      roi_s.push_back(box);
    }

    ret = context->md->setROI(roi_s);
    if (ret != 0) {
      LOGE("Failed to set roi\n");
      return -1;
    }
  }
  std::vector<ObjectBoxInfo> objs;
  ret = context->md->detect(detect_image_context->image, threshold, min_area,
                            objs);
  if (ret != 0) {
    LOGE("Failed to detect\n");
    return -1;
  }
  TDL_InitObjectMeta(obj_meta, objs.size(), 0);
  for (size_t i = 0; i < objs.size(); i++) {
    obj_meta->info[i].box.x1 = objs[i].x1;
    obj_meta->info[i].box.y1 = objs[i].y1;
    obj_meta->info[i].box.x2 = objs[i].x2;
    obj_meta->info[i].box.y2 = objs[i].y2;
  }
  return 0;
}

#endif

/*******************************************
 *          APP API implementation         *
 *******************************************/

int32_t TDL_APP_Init(TDLHandle handle, const char *task,
                     const char *config_file, char ***channel_names,
                     uint8_t *channel_size) {
  TDLContext *context = (TDLContext *)handle;
  int ret = 0;
  if (context == nullptr) {
    return -1;
  }

  if (context->app_task == nullptr) {
    context->app_task = AppFactory::createAppTask(task, config_file);
    if (context->app_task == nullptr) {
      LOGE("Failed to create app_task\n");
      return -1;
    }

    int32_t ret = context->app_task->init();
    if (ret != 0) {
      LOGE("app_task init failed\n");
      return -1;
    }

    std::vector<std::string> ch_names = context->app_task->getChannelNames();

    uint8_t n = ch_names.size();
    *channel_names = (char **)malloc(sizeof(char *) * n);
    if (!*channel_names) {
      LOGE("malloc failed\n");
      return -1;
    }

    for (size_t i = 0; i < n; ++i) {
      const std::string &str = ch_names[i];
      (*channel_names)[i] = (char *)malloc(str.size() + 1);  // +1 for '\0'
      if (!(*channel_names)[i]) {
        for (size_t j = 0; j < i; ++j) {
          free((*channel_names)[j]);
        }
        free(*channel_names);
        *channel_names = nullptr;
        LOGE("malloc failed\n");
        return -1;
      }
      strcpy((*channel_names)[i], str.c_str());
    }

    *channel_size = n;
    return 0;
  }

  return 0;
}

int32_t TDL_APP_SetFrame(TDLHandle handle, const char *channel_name,
                         TDLImage image_handle, uint64_t frame_id,
                         int buffer_size) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->app_task == nullptr) {
    LOGE("app_task is not init\n");
    return -1;
  }

  if (image_handle != NULL) {
    int channel_max_processing_num =
        context->app_task->getChannelMaxProcessingNum(
            std::string(channel_name));

    if (channel_max_processing_num > buffer_size) {
      context->app_task->send_interval += 1;
    } else if (channel_max_processing_num < buffer_size) {
      context->app_task->send_interval =
          std::max(context->app_task->send_interval - 1, 1);
    }

    if (channel_max_processing_num > buffer_size ||
        frame_id % context->app_task->send_interval != 0) {
      LOGI("to skip frame %ld", frame_id);
      return 0;
    }

    std::shared_ptr<BaseImage> image = ((TDLImageContext *)image_handle)->image;
    return context->app_task->setFrame(std::string(channel_name), image,
                                       frame_id);

  } else {
    LOGE("image_handle is NULL!");
    return -1;
  }
}

int32_t TDL_APP_Capture(TDLHandle handle, const char *channel_name,
                        TDLCaptureInfo *capture_info) {
  TDLContext *context = (TDLContext *)handle;
  int ret = 0;
  if (context == nullptr) {
    return -1;
  }
  if (context->app_task == nullptr) {
    LOGE("app_task is not init\n");
    return -1;
  }

  int processing_channel_num = context->app_task->getProcessingChannelNum();
  if (processing_channel_num == 0) {
    printf("no processing channel\n");
    return 2;
  }
  if ((context->app_task->getChannelNodeName(std::string(channel_name), 0) ==
       "video_node") ==
      context->app_task->isExternalFrameChannel(std::string(channel_name))) {
    LOGE("only one of TDLImage and video_node should be set!");
    return -1;
  }

  Packet result;
  ret = context->app_task->getResult(std::string(channel_name), result);
  if (ret != 0) {
    printf("get result failed\n");
    context->app_task->removeChannel(std::string(channel_name));
    return 1;
  }

  std::shared_ptr<FacePetCaptureResult> ori_capture_info =
      result.get<std::shared_ptr<FacePetCaptureResult>>();
  if (ori_capture_info == nullptr) {
    printf("capture_info is nullptr\n");
    return 1;
  }

  capture_info->frame_id = ori_capture_info->frame_id;
  capture_info->frame_width = ori_capture_info->frame_width;
  capture_info->frame_height = ori_capture_info->frame_height;

  if (ori_capture_info->image) {
    TDLImageContext *image_context = new TDLImageContext();
    image_context->image = ori_capture_info->image;
    capture_info->image = (TDLImage)image_context;
  }

  if (ori_capture_info->face_boxes.size() >
      0) {  // face_meta from object detection without landmarks
    TDL_InitFaceMeta(&capture_info->face_meta,
                     ori_capture_info->face_boxes.size(), 0);
    capture_info->face_meta.width = capture_info->frame_width;
    capture_info->face_meta.height = capture_info->frame_height;
    for (size_t i = 0; i < ori_capture_info->face_boxes.size(); i++) {
      capture_info->face_meta.info[i].box.x1 =
          ori_capture_info->face_boxes[i].x1;
      capture_info->face_meta.info[i].box.y1 =
          ori_capture_info->face_boxes[i].y1;
      capture_info->face_meta.info[i].box.x2 =
          ori_capture_info->face_boxes[i].x2;
      capture_info->face_meta.info[i].box.y2 =
          ori_capture_info->face_boxes[i].y2;
      capture_info->face_meta.info[i].score =
          ori_capture_info->face_boxes[i].score;
    }
  }

  TDL_InitObjectMeta(&capture_info->person_meta,
                     ori_capture_info->person_boxes.size(), 0);
  for (int i = 0; i < ori_capture_info->person_boxes.size(); i++) {
    capture_info->person_meta.info[i].box.x1 =
        ori_capture_info->person_boxes[i].x1;
    capture_info->person_meta.info[i].box.y1 =
        ori_capture_info->person_boxes[i].y1;
    capture_info->person_meta.info[i].box.x2 =
        ori_capture_info->person_boxes[i].x2;
    capture_info->person_meta.info[i].box.y2 =
        ori_capture_info->person_boxes[i].y2;
    capture_info->person_meta.info[i].class_id =
        ori_capture_info->person_boxes[i].class_id;
    capture_info->person_meta.info[i].score =
        ori_capture_info->person_boxes[i].score;
  }

  TDL_InitObjectMeta(&capture_info->pet_meta,
                     ori_capture_info->pet_boxes.size(), 0);
  for (int i = 0; i < ori_capture_info->pet_boxes.size(); i++) {
    capture_info->pet_meta.info[i].box.x1 = ori_capture_info->pet_boxes[i].x1;
    capture_info->pet_meta.info[i].box.y1 = ori_capture_info->pet_boxes[i].y1;
    capture_info->pet_meta.info[i].box.x2 = ori_capture_info->pet_boxes[i].x2;
    capture_info->pet_meta.info[i].box.y2 = ori_capture_info->pet_boxes[i].y2;
    capture_info->pet_meta.info[i].class_id =
        ori_capture_info->pet_boxes[i].class_id;
    capture_info->pet_meta.info[i].score = ori_capture_info->pet_boxes[i].score;
  }

  TDL_InitTrackMeta(&capture_info->track_meta,
                    ori_capture_info->track_results.size());
  for (int i = 0; i < ori_capture_info->track_results.size(); i++) {
    TrackerInfo track_info = ori_capture_info->track_results[i];
    capture_info->track_meta.info[i].id = track_info.track_id_;
    capture_info->track_meta.info[i].bbox.x1 = track_info.box_info_.x1;
    capture_info->track_meta.info[i].bbox.x2 = track_info.box_info_.x2;
    capture_info->track_meta.info[i].bbox.y1 = track_info.box_info_.y1;
    capture_info->track_meta.info[i].bbox.y2 = track_info.box_info_.y2;

    if (track_info.obj_idx_ != -1) {
      if (track_info.box_info_.object_type == TDLObjectType::OBJECT_TYPE_FACE) {
        capture_info->face_meta.info[track_info.obj_idx_].track_id =
            track_info.track_id_;
      } else if (track_info.box_info_.object_type ==
                 TDLObjectType::OBJECT_TYPE_PERSON) {
        capture_info->person_meta
            .info[track_info.obj_idx_ - ori_capture_info->face_boxes.size()]
            .track_id = track_info.track_id_;
      }
    }
  }

  capture_info->snapshot_size = ori_capture_info->face_snapshots.size();
  if (capture_info->snapshot_size > 0) {
    capture_info->snapshot_info = (TDLSnapshotInfo *)malloc(
        capture_info->snapshot_size * sizeof(TDLSnapshotInfo));
    memset(capture_info->snapshot_info, 0,
           capture_info->snapshot_size * sizeof(TDLSnapshotInfo));
    capture_info->features =
        (TDLFeature *)malloc(capture_info->snapshot_size * sizeof(TDLFeature));

    for (int i = 0; i < capture_info->snapshot_size; i++) {
      capture_info->snapshot_info[i].quality =
          ori_capture_info->face_snapshots[i].quality;
      capture_info->snapshot_info[i].snapshot_frame_id =
          ori_capture_info->face_snapshots[i].snapshot_frame_id;
      capture_info->snapshot_info[i].track_id =
          ori_capture_info->face_snapshots[i].track_id;
      std::vector<float> feature = ori_capture_info->face_features.at(
          ori_capture_info->face_snapshots[i].track_id);

      if (ori_capture_info->face_snapshots[i].object_image) {
        TDLImageContext *object_image_context = new TDLImageContext();
        object_image_context->image =
            ori_capture_info->face_snapshots[i].object_image;
        capture_info->snapshot_info[i].object_image =
            (TDLImage)object_image_context;
      }

      auto face_attribute = ori_capture_info->face_attributes[i];
      capture_info->snapshot_info[i].male =
          face_attribute
                      [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER] >
                  0.5
              ? 1
              : 0;
      capture_info->snapshot_info[i].glass =
          face_attribute
                      [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES] >
                  0.5
              ? 1
              : 0;
      capture_info->snapshot_info[i].age =
          (int)(face_attribute
                    [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE] *
                100);
      capture_info->snapshot_info[i].emotion = (int)face_attribute
          [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_EMOTION];

      if (feature.size() == 0) {
        LOGE("face feature size = 0!\n");
        return -1;
      }

      capture_info->features[i].size = feature.size();
      capture_info->features[i].type = TDL_TYPE_INT8;
      capture_info->features[i].ptr =
          (int8_t *)malloc(feature.size() * sizeof(int8_t));
      for (int j = 0; j < feature.size(); j++) {
        capture_info->features[i].ptr[j] = (int)feature[j];
      }
    }
  }
  return 0;
}

int32_t TDL_APP_ObjectCounting(TDLHandle handle, const char *channel_name,
                               TDLObjectCountingInfo *object_counting_info) {
  TDLContext *context = (TDLContext *)handle;
  int ret = 0;
  if (context == nullptr) {
    return -1;
  }
  if (context->app_task == nullptr) {
    LOGE("app_task is not init\n");
    return -1;
  }

  if (strstr(channel_name, "consumer_counting") == NULL &&
      strstr(channel_name, "cross_detection") == NULL) {
    LOGE("channel_name should contain consumer_counting or cross_detection\n");
    return -1;
  }

  int processing_channel_num = context->app_task->getProcessingChannelNum();
  if (processing_channel_num == 0) {
    printf("no processing channel\n");
    return 2;
  }
  if ((context->app_task->getChannelNodeName(std::string(channel_name), 0) ==
       "video_node") ==
      context->app_task->isExternalFrameChannel(std::string(channel_name))) {
    LOGE("only one of TDLImage and video_node should be set!");
    return -1;
  }

  Packet result;
  ret = context->app_task->getResult(std::string(channel_name), result);
  if (ret != 0) {
    printf("get result failed\n");
    context->app_task->removeChannel(std::string(channel_name));
    return 1;
  }

  std::shared_ptr<ConsumerCountingResult> consumer_counting_result =
      result.get<std::shared_ptr<ConsumerCountingResult>>();
  if (consumer_counting_result == nullptr) {
    printf("consumer_counting_result is nullptr\n");
    return 1;
  }

  object_counting_info->frame_id = consumer_counting_result->frame_id;
  object_counting_info->frame_width = consumer_counting_result->frame_width;
  object_counting_info->frame_height = consumer_counting_result->frame_height;
  if (consumer_counting_result->image) {
    TDLImageContext *image_context = new TDLImageContext();
    image_context->image = consumer_counting_result->image;
    object_counting_info->image = (TDLImage)image_context;
  }

  if (consumer_counting_result->object_boxes.size() > 0) {
    TDL_InitObjectMeta(&object_counting_info->object_meta,
                       consumer_counting_result->object_boxes.size(), 0);
    object_counting_info->object_meta.width =
        consumer_counting_result->frame_width;
    object_counting_info->object_meta.height =
        consumer_counting_result->frame_height;
    for (size_t i = 0; i < consumer_counting_result->object_boxes.size(); i++) {
      object_counting_info->object_meta.info[i].box.x1 =
          consumer_counting_result->object_boxes[i].x1;
      object_counting_info->object_meta.info[i].box.y1 =
          consumer_counting_result->object_boxes[i].y1;
      object_counting_info->object_meta.info[i].box.x2 =
          consumer_counting_result->object_boxes[i].x2;
      object_counting_info->object_meta.info[i].box.y2 =
          consumer_counting_result->object_boxes[i].y2;
      object_counting_info->object_meta.info[i].score =
          consumer_counting_result->object_boxes[i].score;
      object_counting_info->object_meta.info[i].is_cross = false;
    }
  }

  for (int i = 0; i < consumer_counting_result->track_results.size(); i++) {
    TrackerInfo track_info = consumer_counting_result->track_results[i];

    if (track_info.obj_idx_ != -1) {
      object_counting_info->object_meta.info[track_info.obj_idx_].track_id =
          track_info.track_id_;

      if (std::find(consumer_counting_result->cross_id.begin(),
                    consumer_counting_result->cross_id.end(),
                    track_info.track_id_) !=
          consumer_counting_result->cross_id.end()) {
        object_counting_info->object_meta.info[track_info.obj_idx_].is_cross =
            true;
      }
    }
  }

  for (size_t i = 0; i < consumer_counting_result->counting_line.size(); i++) {
    object_counting_info->counting_line[i] =
        consumer_counting_result->counting_line[i];
  }

  object_counting_info->enter_num = consumer_counting_result->enter_num;
  object_counting_info->miss_num = consumer_counting_result->miss_num;

  return 0;
}

int32_t TDL_APP_ObjectCountingSetLine(TDLHandle handle,
                                      const char *channel_name, int x1, int y1,
                                      int x2, int y2, int mode) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }
  if (context->app_task == nullptr) {
    LOGE("app_task is not init\n");
    return -1;
  }

  if (strstr(channel_name, "consumer_counting") != NULL) {
    return (std::dynamic_pointer_cast<ConsumerCountingAPP>(context->app_task))
        ->setLine(std::string(channel_name),
                  std::string("consumer_counting_node"), x1, y1, x2, y2, mode);
  } else if (strstr(channel_name, "cross_detection") != NULL) {
    return (std::dynamic_pointer_cast<ConsumerCountingAPP>(context->app_task))
        ->setLine(std::string(channel_name),
                  std::string("cross_detection_node"), x1, y1, x2, y2, mode);
  } else {
    LOGE("channel_name should contain consumer_counting or cross_detection\n");
    return -1;
  }
}
