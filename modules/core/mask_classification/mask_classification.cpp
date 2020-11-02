#include "mask_classification.hpp"

#include "core/cviai_types_mem.h"
#include "rescale_utils.hpp"

#include "cvi_sys.h"
#include "opencv2/opencv.hpp"

#define R_SCALE (1 / (256.0 * 0.229))
#define G_SCALE (1 / (256.0 * 0.224))
#define B_SCALE (1 / (256.0 * 0.225))
#define R_MEAN (0.485 / 0.229)
#define G_MEAN (0.456 / 0.224)
#define B_MEAN (0.406 / 0.225)
#define CROP_PCT 0.875
#define MASK_OUT_NAME "logits_dequant"

namespace cviai {

MaskClassification::MaskClassification() {
  mp_config = std::make_unique<ModelConfig>();
  mp_config->input_mem_type = CVI_MEM_DEVICE;
}

MaskClassification::~MaskClassification() {}

int MaskClassification::initAfterModelOpened(std::vector<initSetup> *data) {
  if (data->size() != 1) {
    LOGE("Mask classification only has 1 input.\n");
    return CVI_FAILURE;
  }
  (*data)[0].factor[0] = R_SCALE;
  (*data)[0].factor[1] = G_SCALE;
  (*data)[0].factor[2] = B_SCALE;
  (*data)[0].mean[0] = R_MEAN;
  (*data)[0].mean[1] = G_MEAN;
  (*data)[0].mean[2] = B_MEAN;
  (*data)[0].use_quantize_scale = true;

  return 0;
}

int MaskClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta) {
  uint32_t img_width = stOutFrame->stVFrame.u32Width;
  uint32_t img_height = stOutFrame->stVFrame.u32Height;
  for (uint32_t i = 0; i < meta->size; i++) {
    cvai_face_info_t face_info = info_rescale_c(img_width, img_height, *meta, i);
    int box_x1 = face_info.bbox.x1;
    int box_y1 = face_info.bbox.y1;
    uint32_t box_w = face_info.bbox.x2 - face_info.bbox.x1;
    uint32_t box_h = face_info.bbox.y2 - face_info.bbox.y1;
    CVI_AI_FreeCpp(&face_info);
    uint32_t min_edge = std::min(box_w, box_h);
    float new_edge = min_edge * CROP_PCT;
    int box_new_x1 = (box_w - new_edge) / 2.f + box_x1;
    int box_new_y1 = (box_h - new_edge) / 2.f + box_y1;

    m_vpss_config[0].crop_attr.enCropCoordinate = VPSS_CROP_RATIO_COOR;
    m_vpss_config[0].crop_attr.stCropRect = {box_new_x1, box_new_y1, (uint32_t)new_edge,
                                             (uint32_t)new_edge};
    run(stOutFrame);

    CVI_TENSOR *out = CVI_NN_GetTensorByName(MASK_OUT_NAME, mp_output_tensors, m_output_num);
    float *out_data = (float *)CVI_NN_TensorPtr(out);

    float max = std::max(out_data[0], out_data[1]);
    float f0 = std::exp(out_data[0] - max);
    float f1 = std::exp(out_data[1] - max);
    float score = f0 / (f0 + f1);

    meta->info[i].mask_score = score;
  }

  return CVI_SUCCESS;
}

}  // namespace cviai