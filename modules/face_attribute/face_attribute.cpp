#include "face_attribute.hpp"

#include "cviai_types_free.h"
#include "face/cvai_face_helper.h"
#include "face_attribute_types.hpp"

#include "core_utils.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define FACE_ATTRIBUTE_MEAN (-0.99609375)
#define FACE_ATTRIBUTE_INPUT_THRESHOLD (1 / 128.0)

#define FACE_OUT_NAME "BMFace_dense_MatMul_folded"
#define AGE_OUT_NAME "Age_Conv_Conv2D"
#define EMOTION_OUT_NAME "Emotion_Conv_Conv2D"
#define GENDER_OUT_NAME "Gender_Conv_Conv2D"
#define RACE_OUT_NAME "Race_Conv_Conv2D"

#define RACE_OUT_THRESH (16.9593105)
#define GENDER_OUT_THRESH (6.53386879)
#define AGE_OUT_THRESH (35.0413513)
#define EMOTION_OUT_THRESH (9.25036811)

namespace cviai {

FaceAttribute::FaceAttribute() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;

  attribute_buffer = new float[ATTR_AGE_FEATURE_DIM];
}

FaceAttribute::~FaceAttribute() {
  if (attribute_buffer != nullptr) {
    delete[] attribute_buffer;
    attribute_buffer = nullptr;
  }
}

int FaceAttribute::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta) {
  uint32_t img_width = stOutFrame->stVFrame.u32Width;
  uint32_t img_height = stOutFrame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_8UC3);
  stOutFrame->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(stOutFrame->stVFrame.u64PhyAddr[0], stOutFrame->stVFrame.u32Length[0]);
  char *va_rgb = (char *)stOutFrame->stVFrame.pu8VirAddr[0];
  uint32_t dst_width = image.cols;
  uint32_t dst_height = image.rows;

  for (size_t i = 0; i < (size_t)dst_height; i++) {
    memcpy(image.ptr(i, 0), va_rgb + stOutFrame->stVFrame.u32Stride[0] * i, dst_width * 3);
  }

  CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);

  for (int i = 0; i < meta->size; ++i) {
    cvai_face_info_t face_info = bbox_rescale(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height,
                                              meta, i);

    prepareInputTensor(image, face_info);
    run(stOutFrame);
    outputParser(meta, i);
    CVI_AI_FreeCpp(&face_info);
  }
  return CVI_RC_SUCCESS;
}

void FaceAttribute::prepareInputTensor(cv::Mat src_image, cvai_face_info_t &face_info) {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  cv::Mat image(input->shape.dim[2], input->shape.dim[3], src_image.type());

  face_align(src_image, image, face_info, input->shape.dim[3], input->shape.dim[2]);

  cv::Mat tmpchannels[3];
  cv::split(image, tmpchannels);

  for (int i = 0; i < 3; ++i) {
    tmpchannels[i].convertTo(tmpchannels[i], CV_32F, FACE_ATTRIBUTE_INPUT_THRESHOLD,
                             FACE_ATTRIBUTE_MEAN);

    int size = tmpchannels[i].rows * tmpchannels[i].cols;
    for (int r = 0; r < tmpchannels[i].rows; ++r) {
      memcpy((float *)CVI_NN_TensorPtr(input) + size * i + tmpchannels[i].cols * r,
             tmpchannels[i].ptr(r, 0), tmpchannels[i].cols * sizeof(float));
    }
  }
}

template <typename U, typename V>
std::pair<U, V> ExtractFeatures(float *out, int size) {
  SoftMaxForBuffer(out, out, size);
  size_t max_idx = 0;
  float max_val = -1e3;

  for (int i = 0; i < size; i++) {
    if (out[i] > max_val) {
      max_val = out[i];
      max_idx = i;
    }
  }

  V features;
  memcpy(features.features.data(), out, sizeof(float) * size);
  return std::make_pair(U(max_idx + 1), features);
}

std::pair<float, AgeFeature> ExtractAge(float *out, const int age_prob_size) {
  float expect_age = 0;
  if (age_prob_size == 1) {
    expect_age = out[0];
  } else if (age_prob_size == 101) {
    SoftMaxForBuffer(out, out, age_prob_size);
    for (size_t i = 0; i < 101; i++) {
      expect_age += (i * out[i]);
    }
  } else {
    expect_age = -1.0;
  }

  AgeFeature features;
  memcpy(features.features.data(), out, sizeof(float) * age_prob_size);
  return std::make_pair(expect_age, features);
}

void FaceAttribute::outputParser(cvai_face_t *meta, int meta_i) {
  FaceAttributeInfo result;

  // feature
  CVI_TENSOR *out = CVI_NN_GetTensorByName(FACE_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *face_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t face_feature_size = CVI_NN_TensorCount(out);
  // Create feature
  CVI_AI_FreeCpp(&meta->face_info[meta_i].face_feature);
  meta->face_info[meta_i].face_feature.ptr = (int8_t *)malloc(sizeof(int8_t) * face_feature_size);
  meta->face_info[meta_i].face_feature.size = face_feature_size;
  meta->face_info[meta_i].face_feature.type = TYPE_INT8;
  memcpy(meta->face_info[meta_i].face_feature.ptr, face_blob, face_feature_size);

  // race
  out = CVI_NN_GetTensorByName(RACE_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *race_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t race_prob_size = CVI_NN_TensorCount(out);
  Dequantize(race_blob, attribute_buffer, RACE_OUT_THRESH, race_prob_size);
  auto race = ExtractFeatures<cvai_face_race_e, RaceFeature>(attribute_buffer, race_prob_size);
  result.race = race.first;
  result.race_prob = std::move(race.second);

  // gender
  out = CVI_NN_GetTensorByName(GENDER_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *gender_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t gender_prob_size = CVI_NN_TensorCount(out);
  Dequantize(gender_blob, attribute_buffer, GENDER_OUT_THRESH, gender_prob_size);
  auto gender =
      ExtractFeatures<cvai_face_gender_e, GenderFeature>(attribute_buffer, gender_prob_size);
  result.gender = gender.first;
  result.gender_prob = std::move(gender.second);

  // age
  out = CVI_NN_GetTensorByName(AGE_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *age_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t age_prob_size = CVI_NN_TensorCount(out);
  Dequantize(age_blob, attribute_buffer, AGE_OUT_THRESH, age_prob_size);
  auto age = ExtractAge(attribute_buffer, age_prob_size);
  result.age = age.first;
  result.age_prob = std::move(age.second);

  // emotion
  out = CVI_NN_GetTensorByName(EMOTION_OUT_NAME, mp_output_tensors, m_output_num);
  int8_t *emotion_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t emotion_prob_size = CVI_NN_TensorCount(out);
  Dequantize(emotion_blob, attribute_buffer, EMOTION_OUT_THRESH, emotion_prob_size);
  auto emotion =
      ExtractFeatures<cvai_face_emotion_e, EmotionFeature>(attribute_buffer, emotion_prob_size);
  result.emotion = emotion.first;
  result.emotion_prob = std::move(emotion.second);

  meta->face_info[meta_i].race = result.race;
  meta->face_info[meta_i].gender = result.gender;
  meta->face_info[meta_i].emotion = result.emotion;
  meta->face_info[meta_i].age = result.age;

#ifdef ENABLE_FACE_ATTRIBUTE_DEBUG
  std::cout << "Emotion   |" << getEmotionString(result.emotion) << std::endl;
  std::cout << "Age       |" << result.age << std::endl;
  std::cout << "Gender    |" << getGenderString(result.gender) << std::endl;
  std::cout << "Race      |" << getRaceString(result.race) << std::endl;
#endif
}

}  // namespace cviai