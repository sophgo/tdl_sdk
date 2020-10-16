#include "osnet.hpp"

#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define STD_R (255.f * 0.229f)
#define STD_G (255.f * 0.224f)
#define STD_B (255.f * 0.225f)
#define MODEL_MEAN_R (0.485f * 255.f)
#define MODEL_MEAN_G (0.456f * 255.f)
#define MODEL_MEAN_B (0.406f * 255.f)

#define OSNET_OUT_NAME "feature"

namespace cviai {

OSNet::OSNet() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->skip_postprocess = true;
  mp_config->skip_preprocess = true;
  mp_config->input_mem_type = CVI_MEM_DEVICE;
  m_use_vpss_crop = true;
}

int OSNet::initAfterModelOpened(float *factor, float *mean, bool &pad_reverse,
                                bool &keep_aspect_ratio, bool &use_model_threshold) {
  factor[0] = 1 / STD_R;
  factor[1] = 1 / STD_G;
  factor[2] = 1 / STD_B;
  mean[0] = MODEL_MEAN_R / STD_R;
  mean[1] = MODEL_MEAN_G / STD_G;
  mean[2] = MODEL_MEAN_B / STD_B;
  keep_aspect_ratio = false;
  use_model_threshold = true;
  return 0;
}

int OSNet::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_object_t *meta, int obj_idx) {
  for (uint32_t i = 0; i < meta->size; ++i) {
    if (obj_idx != -1 && i != (uint32_t)obj_idx) continue;
    cvai_bbox_t box =
        box_rescale(stOutFrame->stVFrame.u32Width, stOutFrame->stVFrame.u32Height, meta->width,
                    meta->height, meta->info[i].bbox, BOX_RESCALE_TYPE::CENTER);
    m_crop_attr.bEnable = true;
    m_crop_attr.enCropCoordinate = VPSS_CROP_ABS_COOR;
    m_crop_attr.stCropRect = {(int32_t)box.x1, (int32_t)box.y1, (uint32_t)(box.x2 - box.x1),
                              (uint32_t)(box.y2 - box.y1)};
    run(stOutFrame);

    // feature
    CVI_TENSOR *out = CVI_NN_GetTensorByName(OSNET_OUT_NAME, mp_output_tensors, m_output_num);
    int8_t *feature_blob = (int8_t *)CVI_NN_TensorPtr(out);
    size_t feature_size = CVI_NN_TensorCount(out);
    // Create feature
    CVI_AI_MemAlloc(sizeof(int8_t), feature_size, TYPE_INT8, &meta->info[i].feature);
    memcpy(meta->info[i].feature.ptr, feature_blob, feature_size);
  }
  return CVI_SUCCESS;
}

}  // namespace cviai