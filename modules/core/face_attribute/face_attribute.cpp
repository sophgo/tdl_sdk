#include "face_attribute.hpp"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/face/cvai_face_helper.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cviai_log.hpp"
#include "face_attribute_types.hpp"
#include "face_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define ENABLE_HW_WRAP_TEST 0

#define FACE_ATTRIBUTE_QUANTIZE_SCALE (0.996098577)
#define FACE_ATTRIBUTE_MEAN (-0.99609375)

#define ATTRIBUTE_OUT_NAME "BMFace_dense_MatMul_folded"
#define RECOGNITION_OUT_NAME "pre_fc1"
#define AGE_OUT_NAME "Age_Conv_Conv2D"
#define EMOTION_OUT_NAME "Emotion_Conv_Conv2D"
#define GENDER_OUT_NAME "Gender_Conv_Conv2D"
#define RACE_OUT_NAME "Race_Conv_Conv2D"

#define RACE_OUT_THRESH (16.9593105)
#define GENDER_OUT_THRESH (6.53386879)
#define AGE_OUT_THRESH (35.0413513)
#define EMOTION_OUT_THRESH (9.25036811)

namespace cviai {

FaceAttribute::FaceAttribute(bool use_wrap_hw) : m_use_wrap_hw(use_wrap_hw) {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->skip_preprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
  attribute_buffer = new float[ATTR_AGE_FEATURE_DIM];
}

int FaceAttribute::initAfterModelOpened() {
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  VPSS_CHN_ATTR_S vpssChnAttr;
  const float factor[] = {FACE_ATTRIBUTE_QUANTIZE_SCALE, FACE_ATTRIBUTE_QUANTIZE_SCALE,
                          FACE_ATTRIBUTE_QUANTIZE_SCALE};
  const float mean[] = {(-1) * FACE_ATTRIBUTE_MEAN * 128 / FACE_ATTRIBUTE_QUANTIZE_SCALE,
                        (-1) * FACE_ATTRIBUTE_MEAN * 128 / FACE_ATTRIBUTE_QUANTIZE_SCALE,
                        (-1) * FACE_ATTRIBUTE_MEAN * 128 / FACE_ATTRIBUTE_QUANTIZE_SCALE};
  VPSS_CHN_SQ_HELPER(&vpssChnAttr, input->shape.dim[3], input->shape.dim[2],
                     PIXEL_FORMAT_RGB_888_PLANAR, factor, mean, false);
  m_vpss_chn_attr.push_back(vpssChnAttr);
  PIXEL_FORMAT_E format = m_use_wrap_hw ? PIXEL_FORMAT_RGB_888_PLANAR : PIXEL_FORMAT_RGB_888;
  if (CREATE_VBFRAME_HELPER(&m_gdc_blk, &m_wrap_frame, vpssChnAttr.u32Width, vpssChnAttr.u32Height,
                            format) != CVI_SUCCESS) {
    return -1;
  }
  if (!m_use_wrap_hw) {
    m_wrap_frame.stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
        m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.u32Length[0]);
  }
  return 0;
}

FaceAttribute::~FaceAttribute() {
  if (attribute_buffer != nullptr) {
    delete[] attribute_buffer;
    attribute_buffer = nullptr;
  }
  if (m_gdc_blk != (VB_BLK)-1) {
    CVI_SYS_Munmap((void *)m_wrap_frame.stVFrame.pu8VirAddr[0], m_wrap_frame.stVFrame.u32Length[0]);
    m_wrap_frame.stVFrame.pu8VirAddr[0] = NULL;
    CVI_VB_ReleaseBlock(m_gdc_blk);
  }
}

int FaceAttribute::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta, int face_idx) {
  if (m_use_wrap_hw) {
    if (stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR &&
        stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
      LOGE(
          "Supported format are PIXEL_FORMAT_RGB_888_PLANAR, PIXEL_FORMAT_YUV_PLANAR_420. Current: "
          "%x\n",
          stOutFrame->stVFrame.enPixelFormat);
      return -1;
    }
    for (uint32_t i = 0; i < meta->size; ++i) {
      if (face_idx != -1 && i != (uint32_t)face_idx) continue;

      cvai_face_info_t face_info =
          bbox_rescale(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height, meta, i);
      face_align_gdc(stOutFrame, &m_wrap_frame, face_info);
      run(&m_wrap_frame);
      outputParser(meta, i);
      CVI_AI_FreeCpp(&face_info);
    }
  } else {
    if (stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
      LOGE("Supported format is PIXEL_FORMAT_RGB_888. Current: %x\n",
           stOutFrame->stVFrame.enPixelFormat);
      return -1;
    }
    uint32_t img_width = stOutFrame->stVFrame.u32Width;
    uint32_t img_height = stOutFrame->stVFrame.u32Height;
    bool do_unmap = false;
    if (stOutFrame->stVFrame.pu8VirAddr[0] == NULL) {
      stOutFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(
          stOutFrame->stVFrame.u64PhyAddr[0], stOutFrame->stVFrame.u32Length[0]);
      do_unmap = true;
    }
    cv::Mat image(img_height, img_width, CV_8UC3, stOutFrame->stVFrame.pu8VirAddr[0],
                  stOutFrame->stVFrame.u32Stride[0]);

    for (uint32_t i = 0; i < meta->size; ++i) {
      if (face_idx != -1 && i != (uint32_t)face_idx) continue;

      cvai_face_info_t face_info =
          bbox_rescale(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height, meta, i);
      cv::Mat warp_image(cv::Size(m_wrap_frame.stVFrame.u32Width, m_wrap_frame.stVFrame.u32Height),
                         image.type(), m_wrap_frame.stVFrame.pu8VirAddr[0],
                         m_wrap_frame.stVFrame.u32Stride[0]);
      face_align(image, warp_image, face_info);
      CVI_SYS_IonFlushCache(m_wrap_frame.stVFrame.u64PhyAddr[0],
                            m_wrap_frame.stVFrame.pu8VirAddr[0],
                            m_wrap_frame.stVFrame.u32Length[0]);
      run(&m_wrap_frame);
      outputParser(meta, i);
      CVI_AI_FreeCpp(&face_info);
    }
    if (do_unmap) {
      CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);
      stOutFrame->stVFrame.pu8VirAddr[0] = NULL;
    }
  }
  return CVI_SUCCESS;
}

void FaceAttribute::setWithAttribute(bool with_attr) { m_with_attribute = with_attr; }

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
  std::string feature_out_name = (m_with_attribute) ? ATTRIBUTE_OUT_NAME : RECOGNITION_OUT_NAME;
  CVI_TENSOR *out =
      CVI_NN_GetTensorByName(feature_out_name.c_str(), mp_output_tensors, m_output_num);
  int8_t *face_blob = (int8_t *)CVI_NN_TensorPtr(out);
  size_t face_feature_size = CVI_NN_TensorCount(out);
  // Create feature
  CVI_AI_MemAlloc(sizeof(int8_t), face_feature_size, TYPE_INT8, &meta->info[meta_i].face_feature);
  memcpy(meta->info[meta_i].face_feature.ptr, face_blob, face_feature_size);

  if (!m_with_attribute) {
    return;
  }

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

  meta->info[meta_i].race = result.race;
  meta->info[meta_i].gender = result.gender;
  meta->info[meta_i].emotion = result.emotion;
  meta->info[meta_i].age = result.age;

#ifdef ENABLE_FACE_ATTRIBUTE_DEBUG
  std::cout << "Emotion   |" << getEmotionString(result.emotion) << std::endl;
  std::cout << "Age       |" << result.age << std::endl;
  std::cout << "Gender    |" << getGenderString(result.gender) << std::endl;
  std::cout << "Race      |" << getRaceString(result.race) << std::endl;
#endif
}

}  // namespace cviai