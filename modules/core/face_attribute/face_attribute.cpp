#include "face_attribute.hpp"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/face/cvai_face_helper.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cviai_log.hpp"
#include "face_attribute_types.hpp"
#include "face_utils.hpp"

#include "core/core/cvai_errno.h"
#include "cvi_sys.h"

#define FACE_ATTRIBUTE_FACTOR (1 / 128.f)
#define FACE_ATTRIBUTE_MEAN (0.99609375)

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

FaceAttribute::FaceAttribute(bool with_attr)
    : Core(CVI_MEM_DEVICE), m_use_wrap_hw(false), m_with_attribute(with_attr) {
  attribute_buffer = new float[ATTR_AGE_FEATURE_DIM];
}

int FaceAttribute::setupInputPreprocess(std::vector<InputPreprecessSetup> *data) {
  if (data->size() != 1) {
    LOGE("Face attribute only has 1 input.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }
  for (uint32_t i = 0; i < 3; i++) {
    (*data)[0].factor[i] = FACE_ATTRIBUTE_FACTOR;
    (*data)[0].mean[i] = FACE_ATTRIBUTE_MEAN;
  }
  (*data)[0].use_quantize_scale = true;

  return CVIAI_SUCCESS;
}

int FaceAttribute::onModelOpened() { return allocateION(); }

int FaceAttribute::onModelClosed() {
  releaseION();
  return CVIAI_SUCCESS;
}

CVI_S32 FaceAttribute::allocateION() {
  CVI_SHAPE shape = getInputShape(0);
  PIXEL_FORMAT_E format = m_use_wrap_hw ? PIXEL_FORMAT_RGB_888_PLANAR : PIXEL_FORMAT_RGB_888;
  if (CREATE_ION_HELPER(&m_wrap_frame, shape.dim[3], shape.dim[2], format, "tpu") != CVI_SUCCESS) {
    LOGE("Cannot allocate ion for preprocess\n");
    return CVIAI_ERR_ALLOC_ION_FAIL;
  }
  return CVIAI_SUCCESS;
}

void FaceAttribute::releaseION() {
  if (m_wrap_frame.stVFrame.u64PhyAddr[0] != 0) {
    CVI_SYS_IonFree(m_wrap_frame.stVFrame.u64PhyAddr[0], m_wrap_frame.stVFrame.pu8VirAddr[0]);
    m_wrap_frame.stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    m_wrap_frame.stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    m_wrap_frame.stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    m_wrap_frame.stVFrame.pu8VirAddr[0] = NULL;
    m_wrap_frame.stVFrame.pu8VirAddr[1] = NULL;
    m_wrap_frame.stVFrame.pu8VirAddr[2] = NULL;
  }
}

FaceAttribute::~FaceAttribute() {
  if (attribute_buffer != nullptr) {
    delete[] attribute_buffer;
    attribute_buffer = nullptr;
  }
}

void FaceAttribute::setHardwareGDC(bool use_wrap_hw) {
  if (isInitialized()) {
    LOGW("Please invoke CVI_AI_EnableGDC before opening model\n");
    return;
  }

  m_use_wrap_hw = use_wrap_hw;
}

int FaceAttribute::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta, int face_idx) {
  if (m_use_wrap_hw) {
    if (stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR &&
        stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
      LOGE(
          "Supported format are PIXEL_FORMAT_RGB_888_PLANAR, PIXEL_FORMAT_YUV_PLANAR_420. Current: "
          "%x\n",
          stOutFrame->stVFrame.enPixelFormat);
      return CVIAI_ERR_INVALID_ARGS;
    }

    for (uint32_t i = 0; i < meta->size; ++i) {
      if (face_idx != -1 && i != (uint32_t)face_idx) continue;

      cvai_face_info_t face_info =
          info_rescale_c(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height, *meta, i);
      face_align_gdc(stOutFrame, &m_wrap_frame, face_info);
      std::vector<VIDEO_FRAME_INFO_S *> frames = {&m_wrap_frame};
      int ret = run(frames);
      if (ret != CVIAI_SUCCESS) {
        return ret;
      }
      outputParser(meta, i);
      CVI_AI_FreeCpp(&face_info);
    }
  } else {
    if (stOutFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
      LOGE("Supported format is PIXEL_FORMAT_RGB_888. Current: %x\n",
           stOutFrame->stVFrame.enPixelFormat);
      return CVIAI_ERR_INVALID_ARGS;
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
          info_rescale_c(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height, *meta, i);
      cv::Mat warp_image(cv::Size(m_wrap_frame.stVFrame.u32Width, m_wrap_frame.stVFrame.u32Height),
                         image.type(), m_wrap_frame.stVFrame.pu8VirAddr[0],
                         m_wrap_frame.stVFrame.u32Stride[0]);
      face_align(image, warp_image, face_info);
      std::vector<VIDEO_FRAME_INFO_S *> frames = {&m_wrap_frame};
      int ret = run(frames);
      if (ret != CVIAI_SUCCESS) {
        return ret;
      }

      outputParser(meta, i);
      CVI_AI_FreeCpp(&face_info);
    }
    if (do_unmap) {
      CVI_SYS_Munmap((void *)stOutFrame->stVFrame.pu8VirAddr[0], stOutFrame->stVFrame.u32Length[0]);
      stOutFrame->stVFrame.pu8VirAddr[0] = NULL;
    }
  }
  return CVIAI_SUCCESS;
}

template <typename U, typename V>
struct ExtractFeatures {
  std::pair<U, V> operator()(float *out, const int size) {
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
};

template <>
struct ExtractFeatures<float, AgeFeature> {
  std::pair<float, AgeFeature> operator()(float *out, const int age_prob_size) {
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
};

template <typename U, typename V>
std::pair<U, V> getDequantTensor(const TensorInfo &tinfo, float threshold, float *buffer,
                                 ExtractFeatures<U, V> functor) {
  int8_t *blob = tinfo.get<int8_t>();
  size_t prob_size = tinfo.tensor_elem;
  Dequantize(blob, buffer, threshold, prob_size);
  return functor(buffer, prob_size);
}

void FaceAttribute::outputParser(cvai_face_t *meta, int meta_i) {
  FaceAttributeInfo result;

  // feature
  std::string feature_out_name = (m_with_attribute) ? ATTRIBUTE_OUT_NAME : RECOGNITION_OUT_NAME;
  const TensorInfo &tinfo = getOutputTensorInfo(feature_out_name);
  int8_t *face_blob = tinfo.get<int8_t>();
  size_t face_feature_size = tinfo.tensor_elem;
  // Create feature
  CVI_AI_MemAlloc(sizeof(int8_t), face_feature_size, TYPE_INT8, &meta->info[meta_i].feature);
  memcpy(meta->info[meta_i].feature.ptr, face_blob, face_feature_size);

  if (!m_with_attribute) {
    return;
  }

  // race
  auto race = getDequantTensor(getOutputTensorInfo(RACE_OUT_NAME), RACE_OUT_THRESH,
                               attribute_buffer, ExtractFeatures<cvai_face_race_e, RaceFeature>());
  result.race = race.first;
  result.race_prob = std::move(race.second);

  // gender
  auto gender =
      getDequantTensor(getOutputTensorInfo(GENDER_OUT_NAME), GENDER_OUT_THRESH, attribute_buffer,
                       ExtractFeatures<cvai_face_gender_e, GenderFeature>());
  result.gender = gender.first;
  result.gender_prob = std::move(gender.second);

  // age
  auto age = getDequantTensor(getOutputTensorInfo(AGE_OUT_NAME), AGE_OUT_THRESH, attribute_buffer,
                              ExtractFeatures<float, AgeFeature>());
  result.age = age.first;
  result.age_prob = std::move(age.second);

  // emotion
  auto emotion =
      getDequantTensor(getOutputTensorInfo(EMOTION_OUT_NAME), EMOTION_OUT_THRESH, attribute_buffer,
                       ExtractFeatures<cvai_face_emotion_e, EmotionFeature>());
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